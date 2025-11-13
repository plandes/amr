"""AMR container classes that fit a document/sentence hierarchy.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Union, List, Tuple, Dict, Iterable, Any, Optional, Type
from dataclasses import dataclass, field
import sys
import re
import itertools as it
from io import TextIOBase, StringIO
from pathlib import Path
from penman.graph import Graph
from zensols.config import Writable
from zensols.install import Installer
from zensols.persist import PersistableContainer, persisted
from amrlib.graph_processing.amr_loading import load_amr_entries
from amrlib.graph_processing.amr_loading_raw import load_raw_amr
from . import AmrSentence, AmrGeneratedSentence
from .varidx import VariableIndexer


@dataclass(init=False)
class AmrDocument(PersistableContainer, Writable):
    """A document of AMR graphs, which is indexible and iterable.

    """
    _COMMENT_REGEX = re.compile(r'^\s*#\s*[^:]{2}.*')
    _DOC_ID_REGEX = re.compile(r'^(.+)\.\d+$')

    sents: Tuple[AmrSentence, ...] = field()
    """The AMR sentences that make up the document."""

    path: Optional[Path, ...] = field(default=None)
    """If set, the file the sentences were parsed from in Penman notation."""

    def __init__(self, sents: Iterable[Union[str, Graph, AmrSentence]],
                 path: Optional[Path] = None,
                 model: str = None):
        """Initialize.

        :param sents: the document's sentences

        :param path: the path to file containing the Penman notation sentence
                     graphs used in ``sents``

        :param model: the model to initailize :class:`.AmrSentence` when
                      ``sents`` is a list of string Penman graphs

        """
        sents = tuple(sents)
        if len(sents) > 0 and not isinstance(sents[0], AmrSentence):
            self.sents = tuple(
                map(lambda s: AmrSentence(s, model=model), sents))
        else:
            self.sents = sents
        self.path = path

    @property
    @persisted('_text', transient=True)
    def text(self) -> str:
        """The text of the natural language form of the document.  This is the
        concatenation of all the sentinel text.

        """
        return ' '.join(map(lambda s: s.text, self.sents))

    @property
    def graph_string(self) -> str:
        """The graph of all sentences with two newlines as a separator as a
        string in Penman format.

        """
        n_sents = len(self.sents)
        sio = StringIO()
        for i, sent in enumerate(self.sents):
            sio.write(sent.graph_string)
            if i < n_sents - 1:
                sio.write('\n\n')
        return sio.getvalue()

    def get_doc_id(self) -> Optional[str]:
        """Get the ID of the document from the first sentence's ID, if there is
        one.  For example, if the first sentence's ID is ``liu-example.0``, the
        string ``liu-example`` is returned.

        :return: the porition of the first sentence's metadata ``id`` of the
                 characters before the sentence index, or ``None`` if the
                 metadata is not available

        """
        if len(self.sents) > 0:
            sent: AmrSentence = self.sents[0]
            sid: str = sent.metadata.get('id')
            m: re.Match = self._DOC_ID_REGEX.match(sid)
            if m is not None:
                return m.group(1)

    def normalize(self):
        """Normalize the graph string to standard notation per the Penman API.

        :see: :meth:`.AmrSentence.normalize`

        """
        sent: AmrSentence
        for sent in self.sents:
            sent.normalize()

    def reindex_variables(self):
        """Reindexes all variables for sentences of a :class:`.AmrDocument` so
        all node variables are unique in the document.

        """
        vi = VariableIndexer()
        vi.reindex(self.sents)

    @staticmethod
    def resolve_source(source: Union[Path, Installer]) -> Path:
        """Coerce a an (optionally) installer to a path."""
        if isinstance(source, Installer):
            source = source.get_singleton_path()
        return source

    @classmethod
    def is_comment(cls, doc: str) -> bool:
        """Return whethor or not ``doc`` is all comments."""
        is_comment = True
        for ln in doc.split('\n'):
            if cls._COMMENT_REGEX.match(ln) is None:
                is_comment = False
                break
        return is_comment

    @classmethod
    def from_source(cls, source: Union[Path, Installer],
                    transform_ascii: bool = False, **kwargs) -> AmrDocument:
        """Return a new document created for ``source``.

        :param source: either a double newline list of AMR graphs or an
                       installer that has a singleton path to a like file

        :param transform_ascii: whether to replace non-ASCII characters to their
                                ASCII equivalents (i.e. removes umlauts)

        :param kwargs: additional keyword arguments given to the initializer of
                       the document

        """
        source: Path = cls.resolve_source(source)
        entries: List[str]
        if transform_ascii:
            entries = tuple(filter(
                lambda s: not cls.is_comment(s),
                load_raw_amr(str(source))))
        else:
            entries = tuple(filter(
                lambda s: not cls.is_comment(s),
                load_amr_entries(str(source), False)))
        return cls(sents=entries, path=source, **kwargs)

    def clone(self, **kwargs) -> AmrDocument:
        """Return a deep copy of this instance."""
        params = dict(kwargs)
        if 'sents' not in params:
            params['sents'] = tuple(map(lambda s: s.clone(), self.sents))
        if 'path' not in params:
            params['path'] = getattr(self, 'path')
        cls = params.pop('cls', self.__class__)
        return cls(**params)

    def from_sentences(self, sents: Iterable[AmrSentence],
                       deep: bool = False) -> AmrDocument:
        """Return a new cloned document using the given sentences.

        :param sents: the sentences to add to the new cloned document

        :param deep: whether or not to clone the sentences

        :see: :meth:`clone`

        """
        if deep:
            sents = tuple(map(lambda s: s.clone(), sents))
        return self.clone(sents=sents)

    @classmethod
    def to_document(cls: Type, sents: Iterable[AmrSentence]):
        """Create a new document from an iterable of sentences."""
        return cls(sents=tuple(sents))

    def remove_wiki_attribs(self):
        """Removes the ``:wiki`` roles from all sentence graphs."""
        sent: AmrSentence
        for sent in self.sents:
            sent.remove_wiki_attribs()

    def remove_alignments(self):
        """Remove text-to-graph alignments in all sentence graphs."""
        sent: AmrSentence
        for sent in self.sents:
            sent.remove_alignments()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              limit_sent: int = sys.maxsize, add_sent_id: bool = False,
              include_metadata: bool = True, text_only: bool = False):
        """
        :param limit_sent: the max number of sentences to write

        :param add_sent_id: add the sentence ID to the output

        :param include_metadata: whether to add graph metadata to the output

        :param text_ony: whether to only write the sentence text rather than the
                         AMR Penman notation

        """
        sent: AmrSentence
        for i, sent in enumerate(it.islice(self.sents, limit_sent)):
            if add_sent_id:
                self._write_line(f'# ::id {i+1}', depth, writer)
            if text_only:
                self._write_line(sent.text, depth, writer)
            else:
                sent.write(depth, writer, include_metadata=include_metadata)
                if i < len(self.sents) - 1:
                    self._write_empty(writer)

    def __eq__(self, other: AmrDocument) -> bool:
        return all(map(lambda s: s[0] == s[1], zip(self.sents, other.sents)))

    def __iter__(self) -> Iterable[AmrSentence]:
        return iter(self.sents)

    def __getitem__(self, index: int):
        return self.sents[index]

    def __len__(self) -> int:
        return len(self.sents)

    def __str__(self):
        return f'sents: {len(self)}, path: {self.path}'

    def __repr(self):
        return self.__str__()


@dataclass
class AmrGeneratedDocument(Writable):
    """A sentence generated by the graph-to-text model.

    """
    sents: Tuple[AmrGeneratedSentence, ...] = field()
    """The predicted sentences using :obj:`amr` as observation."""

    amr: AmrDocument = field()
    """The input document to the model used to predict :obj:`sents`."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              clipped_inline: bool = True,
              amr_doc_kwargs: Dict[str, Any] = None,
              amr_sent_kwargs: Dict[str, Any] = None):
        """
        :param clipped_inline: whether to render clipped sentences in the text
               output line

        :param amr_kwargs: the keyword arguments to pass on to the rendering
                           of :obj:`amr`
        """
        self._write_line('sentences:', depth, writer)
        sent: AmrGeneratedSentence
        for sent in self.sents:
            sent.write(
                depth + 1, writer,
                clipped_inline=clipped_inline,
                amr_kwargs=amr_sent_kwargs)

    def __eq__(self, other: AmrGeneratedSentence) -> bool:
        return all(map(lambda s: s[0] == s[1], zip(self.sents, other.sents)))

    def __iter__(self) -> Iterable[AmrSentence]:
        return iter(self.sents)

    def __getitem__(self, index: int):
        return self.sents[index]

    def __len__(self) -> int:
        return len(self.sents)
