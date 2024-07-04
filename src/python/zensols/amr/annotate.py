"""AMR annotated corpus utility classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, List, Iterable, Set, Tuple, Type, Union, Any, Sequence
from enum import Enum
from dataclasses import dataclass, field
import sys
import logging
import re
from itertools import chain as ch
import textwrap as tw
from io import TextIOBase
from pathlib import Path
import json
import pandas as pd
from spacy.tokens import Doc
from penman import Graph
from zensols.persist import persisted, PersistedWork, Stash, PrimeableStash
from zensols.config import Writable
from zensols.install import Installer
from zensols.nlp import FeatureDocumentParser
from . import (
    AmrError, AmrDocument, AmrSentence, AmrFeatureSentence, AmrFeatureDocument
)
from .model import AmrParser
from .coref import CoreferenceResolver
from .docparser import AnnotationFeatureDocumentParser
from .docfac import AmrFeatureDocumentFactory

logger = logging.getLogger(__name__)


class SentenceType(Enum):
    """The type of sentence in relation to its function in the document.

    """
    TITLE = 't'
    BODY = 'b'
    SUMMARY = 'a'
    SECTION = 's'
    FIGURE_TITLE = 'ft'
    FIGURE = 'f'
    OTHER = 'o'

    def __lt__(self, other: SentenceType) -> bool:
        return self.value < other.value


class AnnotatedAmrSentence(AmrSentence):
    """A sentence containing its index in the document and the funtional type.

    """
    def __init__(self, data: Union[str, Graph], model: str,
                 doc_sent_idx: int, sent_type: SentenceType):
        super().__init__(data, model)
        self.doc_sent_idx = doc_sent_idx
        self.sent_type = sent_type

    def clone(self, cls: Type[AmrSentence] = None, **kwargs) -> AmrSentence:
        """Return a deep copy of this instance."""
        params = dict(
            cls=self.__class__ if cls is None else cls,
            data=self.graph_string,
            model=self._model,
            doc_sent_idx=self.doc_sent_idx,
            sent_type=self.sent_type)
        params.update(kwargs)
        return super().clone(**params)


@dataclass(eq=False, repr=False)
class AnnotatedAmrSectionDocument(AmrDocument):
    """Represents a section from an annotated document.

    """
    section_sents: Tuple[AmrSentence] = field(default=())
    """The sentences that make up the section title (usually just one)."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        for i, sec in enumerate(self.section_sents):
            sec_text = self._trunc(sec.text)
            self._write_line(f'section ({i}): {sec_text}', depth, writer)
        if len(self.section_sents) == 0:
            self._write_line('no section sentences', depth, writer)
        for sent in self.sents:
            sec_text = self._trunc(sent.text)
            self._write_line(f'{sec_text}', depth + 1, writer)

    def __str__(self) -> str:
        return super().__str__() + f', sections: {len(self.section_sents)}'


@dataclass(eq=False, repr=False)
class AnnotatedAmrDocument(AmrDocument):
    """An AMR document containing a unique document identifier from the corpus.

    """
    doc_id: str = field(default=None)
    """The unique document identifier."""

    def _filter_by_sentence_type(self, stypes: Set[SentenceType]) -> \
            Iterable[AmrSentence]:
        return filter(lambda s: s.sent_type in stypes, self.sents)

    @property
    def summary(self) -> AmrDocument:
        """The sentences that make up the summary of the document."""
        sents = self._filter_by_sentence_type({SentenceType.SUMMARY})
        return self.from_sentences(tuple(sents))

    @property
    def body(self) -> AmrDocument:
        """The sentences that make up the body of the document."""
        sents = self._filter_by_sentence_type({SentenceType.BODY})
        return self.from_sentences(tuple(sents))

    @property
    def sections(self) -> Tuple[AnnotatedAmrSectionDocument]:
        """The sections of the document."""
        stypes = {SentenceType.SECTION, SentenceType.BODY}
        secs: List[AnnotatedAmrSectionDocument] = []
        sec_sents: List[AmrSentence] = []
        body_sents: List[AmrSentence] = []
        last_body = False
        sec: AnnotatedAmrSectionDocument
        sent: AmrSentence
        for sent in self._filter_by_sentence_type(stypes):
            if sent.sent_type == SentenceType.SECTION:
                if last_body and (len(sec_sents) > 0 or len(body_sents) > 0):
                    sec = AnnotatedAmrSectionDocument(
                        sents=body_sents,
                        section_sents=sec_sents)
                    secs.append(sec)
                    sec_sents = []
                    body_sents = []
                sec_sents.append(sent)
                last_body = False
            elif sent.sent_type == SentenceType.BODY:
                body_sents.append(sent)
                last_body = True
            else:
                raise ValueError(f'Unknown type: {sent.type}')
        if len(sec_sents) > 0 or len(body_sents) > 0:
            sec = AnnotatedAmrSectionDocument(
                sents=tuple(body_sents),
                section_sents=tuple(sec_sents))
            secs.append(sec)
        return tuple(secs)

    @staticmethod
    def get_feature_sentences(feature_doc: AmrFeatureDocument,
                              amr_docs: Iterable[AmrDocument]) -> \
            Iterable[AmrFeatureSentence]:
        """Return the feature sentences of those that refer to the AMR
        sentences, but starting from the AMR side.

        :param feature_doc: the document having the
                            :class:`~zensols.nlp.container.FeatureSentence`
                            instances

        :param amr_docs: the documents having the sentences, such as
                         :obj:`summary`

        """
        asents: Iterable[AmrSentence] = map(lambda s: s.sents, amr_docs)
        sent_ids: Set[int] = set(map(id, ch.from_iterable(asents)))
        return filter(lambda s: id(s.amr) in sent_ids, feature_doc)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_summary: bool = True, include_sections: bool = True,
              include_body: bool = False, include_amr: bool = True,
              **kwargs):
        """Write the contents of this instance to ``writer`` using indention
        ``depth``.

        :param include_summary: whether to include the summary sentences

        :param include_sectional: whether to include the sectional sentences

        :param include_body: whether to include the body sentences

        :param include_amr: whether to include the super class AMR output

        :param kwargs: arguments given to the super classe's write, such as
                       ``limit_sent=0`` to effectively disable it

        """
        summary = self.summary
        body = self.body
        sections = self.sections
        if include_amr:
            super().write(depth, writer, **kwargs)
        if include_summary and len(summary) > 0:
            self._write_line('summary:', depth, writer)
            for sent in summary:
                self._write_line(self._trunc(sent.text), depth + 1, writer)
        if include_sections and len(sections) > 0:
            self._write_line('sections:', depth, writer)
            for sec in self.sections:
                sec.write(depth + 1, writer)
        if include_body and len(body) > 0:
            self._write_line('body:', depth, writer)
            for sent in self.body:
                self._write_line(self._trunc(sent.text), depth + 1, writer)

    def clone(self, **kwargs) -> AmrDocument:
        return super().clone(doc_id=self.doc_id, **kwargs)

    def __eq__(self, other: AmrDocument) -> bool:
        return self.doc_id == other.doc_id and super().__eq__(other)

    def __str__(self):
        return super().__str__() + f", doc_id: '{self.doc_id}'"


@dataclass
class AnnotatedAmrDocumentStash(Stash):
    """A factory stash that creates :class:`.AnnotatedAmrDocument` instances of
    annotated documents from a single text file containing a corpus of AMR
    Penman formatted graphs.

    """
    _SENT_TYPE_NAME = 'sent-types.csv'
    """The default sentence type file name."""

    installer: Installer = field()
    """The installer containing the AMR annotated corpus."""

    doc_dir: Path = field()
    """The directory containing sentence type mapping for documents or ``None``
    if there are no sentence type alignments.

    """
    corpus_cache_dir: Path = field()
    """A directory to store pickle cache files of the annotated corpus."""

    id_name: str = field()
    """The ID used in the graph string comments containing the document ID."""

    id_regexp: re.Pattern = field(default=re.compile(r'([^.]+)\.(\d+)'))
    """The regular expression used to create the :obj:`id_name` if it exists.
    The regular expression must have with two groups: the first the ID and the
    second is the sentence index.

    """
    sent_type_col: str = field(default='snt-type')
    """The AMR metadata ID used for the sentence type."""

    sent_type_mapping: Dict[str, str] = field(default=None)
    """Used to map what's in the corpus to a value of :class:`SentenceType` if
    given.

    """
    doc_parser: FeatureDocumentParser = field(default=None)
    """If provided, AMR metadata is added to sentences, which is needed by the
    AMR populator.

    """
    amr_sent_model: str = field(default=None)
    """The model set in the :class:`.AmrSentence` initializer."""

    amr_sent_class: Type[AnnotatedAmrSentence] = field(
        default=AnnotatedAmrSentence)
    """The class used to create new instances of :class:`.AmrSentence`."""

    amr_doc_class: Type[AnnotatedAmrDocument] = field(
        default=AnnotatedAmrDocument)
    """The class used to create new instances of :class:`.AmrDocument`."""

    doc_annotator: AnnotationFeatureDocumentParser = field(default=None)
    """Used to annotated AMR documents if not ``None``."""

    def __post_init__(self):
        if self.doc_dir is not None:
            self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_cache_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_doc = PersistedWork(
            self.corpus_cache_dir / 'doc.dat', self, mkdir=True)
        self._corpus_df = PersistedWork(
            self.corpus_cache_dir / 'df.dat', self, mkdir=True)
        if self.doc_annotator.alignment_populator is not None and \
           self.doc_parser is None:
            logger.warning(
                ("Alignment will be accomplished without providing tokens " +
                 "and other metadata needed as 'doc_parser' is not provided"))

    @property
    @persisted('_corpus_doc')
    def corpus_doc(self) -> AmrDocument:
        """A document containing all the sentences from the corpus."""
        self.installer()
        corp_path: Path = self.installer.get_singleton_path()
        return AmrDocument.from_source(corp_path, model=self.amr_sent_model)

    def parse_id(self, id: str) -> Tuple[str, str]:
        """Parse an AMR ID and return it as ``(doc_id, sent_id)``, both strings.

        """
        m: re.Match = self.id_regexp.match(id)
        if m is not None:
            return m.groups()

    @property
    @persisted('_corpus_df')
    def corpus_df(self) -> pd.DataFrame:
        """A data frame containing the identifier, text of the sentences and the
        annotated sentence types of the corpus.

        """
        id_name: str = self.id_name
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'creating corpus dataframe for: {id_name}')
        metas: List[Dict[str, str]] = []
        for six, doc in enumerate(self.corpus_doc):
            meta = dict(doc.metadata)
            meta['sent_idx'] = six
            meta.pop('preferred', None)
            if self.id_regexp is None:
                logger.warning(f"ID mismatch: {meta['id']}: {self.id_regexp}")
            else:
                key_data: Tuple[str, str] = self.parse_id(meta['id'])
                if key_data is not None:
                    id, dsix = key_data
                    meta[id_name] = id
                    meta['doc_sent_idx'] = dsix
            metas.append(meta)
        return pd.DataFrame(metas)

    @property
    @persisted('_doc_counts')
    def doc_counts(self) -> pd.DataFrame:
        """A data frame of the counts by unique identifier."""
        id_name = self.id_name
        existing = set(self.keys())
        df = self.corpus_df
        dfc = df.groupby(id_name)[id_name].agg('count').to_frame().\
            rename(columns={id_name: 'count'}).sort_values(
                'count', ascending=False)
        dfc['exist'] = dfc.index.to_series().apply(lambda i: i in existing)
        return dfc

    def export_sent_type_template(self, doc_id: str, out_path: Path = None):
        """Create a CSV file that contains the sentences and other metadata of
        an annotated document used to annotated sentence types.

        """
        if out_path is None:
            out_path = self.doc_dir / doc_id / self._SENT_TYPE_NAME
        if out_path.exists():
            raise AmrError(
                f'Refusing to overwrite export sentence type file: {out_path}')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.corpus_df
        df = df[df[self.id_name] == doc_id]
        df['sent_type'] = SentenceType.BODY.value
        df = df['doc_sent_idx sent_type snt'.split()]
        df.to_csv(out_path)
        logger.info(f'wrote: {out_path}')

    def _doc_id_to_path(self, doc_id: str) -> Path:
        """Return a path that contains the annotated sentence type mappings for
        a document.

        :param doc_id: the document unique identifier

        """
        return self.doc_dir / doc_id

    def _get_sent_types_from_doc(self, doc_id: str,
                                 doc_path: Path) -> Dict[int, str]:
        if not doc_path.is_dir():
            raise AmrError(f'No metadata document found: {doc_id}')
        st_df: pd.DataFrame = pd.read_csv(doc_path / self._SENT_TYPE_NAME)
        sent_types = dict(zip(st_df['doc_sent_idx'].astype(int).to_list(),
                              st_df['sent_type'].to_list()))
        assert len(sent_types) == len(st_df)
        return sent_types

    def _map_sent_types(self, sent_types: Dict[int, str]):
        stm: Dict[str, str] = self.sent_type_mapping
        if stm is not None:
            ntypes = {}
            for k, v in sent_types.items():
                if v not in stm:
                    raise AmrError(
                        f'Not sentence type: {v} in {stm} from {sent_types}')
                ntypes[k] = stm[v]
            sent_types = ntypes
        return sent_types

    def _get_doc_from_path(self, doc_id: str,
                           sent_types: Dict[int, str]) -> AnnotatedAmrDocument:
        cdoc: AmrDocument = self.corpus_doc
        df: pd.DataFrame = self.corpus_df
        df = df[df[self.id_name] == doc_id]
        if len(df) == 0:
            raise AmrError(f'No corpus document found: {doc_id}')
        if sent_types is None:
            df = df.copy()
            if 'doc_sent_idx' not in df.columns:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'setting doc idx to df len: {len(df)}')
                df['doc_sent_idx'] = tuple(range(len(df)))
            if self.sent_type_col not in df.columns:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'set sent types to all body, len={len(df)}')
                df[self.sent_type_col] = [SentenceType.BODY] * len(df)
            sids: List[int] = df['doc_sent_idx'].astype(int).to_list()
            stypes: List[str] = df[self.sent_type_col].to_list()
            sent_types = dict(zip(sids, stypes))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sides: {sids} ({len(sids)}), ' +
                             f'stypes: {stypes} ({len(stypes)}), ' +
                             f'sent_types: {sent_types} {len(sent_types)}')
        sent_types = self._map_sent_types(sent_types)
        if len(sent_types) != len(df):
            raise AmrError(f'Expected {len(df)} sentence types ' +
                           f'but got {len(sent_types)}')
        sents: List[AnnotatedAmrSentence] = []
        sent_cls: Type[AnnotatedAmrSentence] = self.amr_sent_class
        for _, row in df.iterrows():
            sent: AmrSentence = cdoc[row['sent_idx']]
            sent_text: str = row['snt']
            assert sent.text == sent_text
            doc_sent_idx = int(row['doc_sent_idx'])
            sent_type_nom: str = sent_types[doc_sent_idx]
            sent_type: SentenceType = SentenceType(sent_type_nom)
            sents.append(sent_cls(
                data=sent.graph_string,
                model=self.amr_sent_model,
                doc_sent_idx=doc_sent_idx,
                sent_type=sent_type))
        return self.amr_doc_class(sents=sents, path=cdoc.path, doc_id=doc_id)

    def _add_metadata(self, doc: AnnotatedAmrDocument):
        sent: AnnotatedAmrSentence
        for sent in doc.sents:
            doc: Doc = self.doc_parser.parse_spacy_doc(sent.text)
            AmrParser.add_metadata(sent, doc)

    def load(self, doc_id: str) -> AnnotatedAmrDocument:
        """
        :param doc_id: the document unique identifier
        """
        sent_types: Dict[int, str] = None
        if self.doc_dir is not None:
            doc_path: Path = self._doc_id_to_path(doc_id)
            if doc_path.is_dir():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'getting sent type {doc_id}: {doc_path}')
                sent_types = self._get_sent_types_from_doc(doc_id, doc_path)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'get doc from path: id={doc_id}, stypes={sent_types}')
        doc: AnnotatedAmrDocument = self._get_doc_from_path(doc_id, sent_types)
        if self.doc_annotator is not None:
            if self.doc_parser is not None:
                self._add_metadata(doc)
            if self.doc_annotator.alignment_populator is not None:
                self.doc_annotator.alignment_populator.align(doc)
        return doc

    def keys(self) -> Iterable[str]:
        def filter_doc_path(p: Path) -> bool:
            return (p / self._SENT_TYPE_NAME).is_file()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loading keys from {self.doc_dir}')

        if self.doc_dir is not None:
            keys = map(lambda p: p.name, filter(
                filter_doc_path, self.doc_dir.iterdir()))
        else:
            keys = self.corpus_df[self.id_name].drop_duplicates().to_list()
        return keys

    def exists(self, doc_id: str) -> bool:
        """
        :param doc_id: the document unique identifier
        """
        if self.doc_dir:
            return self._doc_id_to_path(doc_id).is_dir()
        else:
            return doc_id in self.corpus_df[self.id_name].values

    def dump(self, name: str, inst: Any):
        pass

    def delete(self, name: str = None):
        pass

    def clear(self):
        """Remove all corpus cache files."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cleaning: {type(self)}')
        self._corpus_doc.clear()
        self._corpus_df.clear()
        if self.doc_annotator is not None:
            self.doc_annotator.clear()


@dataclass
class AnnotatedAmrFeatureDocumentStash(PrimeableStash):
    """A stash that persists :class:`.AmrFeatureDocument` instances using AMR
    annotates from :class:`.AnnotatedAmrDocumentStash` as a source.  The key set
    and *exists* behavior is identical between to two stashes.  However, the
    instances of :class:`.AmrFeatureDocument` (and its constituent sentences)
    are generated from the AMR annotated sentences (i.e. from the ``::snt`
    metadata field).

    This stash keeps the persistance of the :class:`.AmrDocument` separate from
    instance of the feature document to avoid persisting it twice across
    :obj:`doc_stash` and :obj:`amr_stash`.  On load, these two data structures
    are *stitched* together.

    """
    feature_doc_factory: AmrFeatureDocumentFactory = field()
    """Creates :class:`.AmrFeatureDocument` from :class:`.AmrDocument`
    instances.

    """
    doc_stash: Stash = field()
    """The stash used to persist instances of :class:`.AmrFeatureDocument`.  It
    does not persis the :class:`.AmrDocument` (see class docs).

    """
    amr_stash: AnnotatedAmrDocumentStash = field()
    """The stash used to persist :class:`.AmrDocument` instances that are
    *stitched* together with the :class:`.AmrFeatureDocument` (see class docs).

    """
    coref_resolver: CoreferenceResolver = field(default=None)
    """Adds coreferences between the sentences of the document."""

    def load(self, doc_id: str) -> AmrFeatureDocument:
        amr_doc: AnnotatedAmrDocument = self.amr_stash.load(doc_id)
        doc: AmrFeatureDocument = self.doc_stash.load(doc_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'loaded: {doc_id}: {doc}')
        if doc is None:
            doc = self.feature_doc_factory.to_feature_doc(amr_doc)
            # clear the amr document so it isn't persisted; this is set in
            # :meth:`to_feature_doc` for client use
            doc.amr = None
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'dumping {doc_id}: {doc}')
            self.doc_stash.dump(doc_id, doc)
        # set the document and the respective AmrSentences
        doc.amr = amr_doc
        # optionally add coreference; we could persist res (move after the
        # `to_feature_doc` call) to save with the doc; but better to let the
        # coref_resolver cache it, which is configured to persist
        if self.coref_resolver is not None:
            self.coref_resolver(doc)
        return doc

    def keys(self) -> Iterable[str]:
        return self.amr_stash.keys()

    def exists(self, doc_id: str) -> bool:
        return self.amr_stash.exists(doc_id)

    def dump(self, name: str, inst: Any):
        pass

    def delete(self, name: str = None):
        pass

    def clear(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cleaning: {type(self)}')
        self.doc_stash.clear()
        self.amr_stash.clear()

    def prime(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'priming {type(self)}...')
        for stash in (self.doc_stash, self.amr_stash):
            if isinstance(stash, PrimeableStash):
                stash.prime()


@dataclass
class AnnotatedAmrFeatureDocumentFactory(object):
    """Creates instances of :class:`.AmrFeatureDocument` each with
    :obj:`.AmrFeatureDocument.amr` instance of :class:`.AnnotatedAmrDocument`
    and :obj:`.AmrFeatureSentence.amr` with :class:`.AnnotatedAmrSentence`.
    This is created using a JSON file or a list of :class:`dict`.

    The keys each dictionary are the case-insensitive enumeration values of
    :class:`.SentenceType`.  Keys ``id`` and ``comment`` are the unique document
    identifier and a comment that is added to the AMR sentence metadata.  Both
    are optional, and if ``id`` is missing, :obj:``doc_id``.

    An example JSON creates a document with ID ``ex1``, a ``comment`` metadata,
    one :obj:`.SentenceType.SUMMARY` and two :obj:`.SentenceType.BODY`
    sentences::

        [{
            "id": "ex1",
            "comment": "very short",
            "body": "The man ran to make the train. He just missed it.",
            "summary": "A man got caught in the door of a train he just missed."
        }]

    :see: :meth:`from_dict`

    """
    doc_parser: FeatureDocumentParser = field()
    """The feature document parser used to create the Penman formatted graphs.

    """
    remove_wiki_attribs: bool = field(default=False)
    """Whether to remove the ``:wiki`` roles from all sentence graphs after
    parsing.

    """
    remove_alignments: bool = field(default=False)
    """Whether to remove text-to-graph alignments in all sentence graphs after
    parsing.

    """
    doc_id: int = field(default=0)
    """An instance based enumerated value is used, which is enumerated for each
    document missing an ID.

    """
    def to_annotated_sent(self, sent: AmrFeatureSentence,
                          sent_type: SentenceType = None) -> AmrFeatureSentence:
        """Clone ``sent.amr`` into an :class:`.AnnotatedAmrSentence`.

        :param sent: the sentence to convert to an
                     :class:`.AnnotatedAmrSentence`

        :param sent_type: the type of sentence to set on

        :return: a feature sentence with a new
                 :obj:`~zensols.amr.container.AmrFeatureSentence.amr` to new
                 :class:`AnnotatedAmrSentence`, which is a new instance if
                 ``sent`` isn't an annotated AMR sentence

        """
        if sent_type is None:
            stype_name: str = sent.amr.metadata['snt-type']
            sent_type = SentenceType[stype_name.upper()]
        if not isinstance(sent.amr, AnnotatedAmrSentence):
            asent = sent.amr.clone(
                cls=AnnotatedAmrSentence,
                sent_type=sent_type,
                doc_sent_idx=0)
            if sent_type is not None:
                asent.set_metadata('snt-type', sent_type.name.lower())
            sent = sent.clone()
            sent.amr = asent
        return sent

    def to_annotated_doc(self, doc: AmrFeatureDocument) -> AmrFeatureDocument:
        """Clone ``doc.amr`` into an :class:`.AnnotatedAmrDocument`.

        :param sent: the document to convert to an
                     :class:`.AnnotatedAmrDocument`

        :return: a feature document with a new
                 :obj:`~zensols.amr.container.AmrFeatureDocument.amr` to new
                 :class:`AnnotatedAmrDocument`, which is a new instance if
                 ``sent`` isn't an annotated AMR document

        """
        if not isinstance(doc.amr, AnnotatedAmrDocument):
            fsents: Tuple[AmrFeatureSentence, ...] = \
                tuple(map(self.to_annotated_sent, doc))
            adoc = doc.amr.clone(
                cls=AnnotatedAmrDocument,
                sents=tuple(map(lambda s: s.amr, fsents)))
            if adoc.doc_id is None:
                adoc.doc_id = adoc.get_doc_id()
            doc = doc.clone(amr=adoc, sents=fsents)
        return doc

    def from_str(self, sents: str, stype: SentenceType) -> \
            Iterable[AmrFeatureSentence]:
        """Parse and create AMR sentences from a string.

        :param sents: the string containing a space separated list of sentences

        :param stype: the sentence type assigned to each new AMR sentence

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsing: <{tw.shorten(sents, 60)}>')
        doc: AmrFeatureDocument = self.doc_parser(sents)
        return map(lambda s: self.to_annotated_sent(s, stype), doc)

    def from_dict(self, data: Dict[str, str]) -> AmrFeatureDocument:
        """Parse and create an AMR document from a :class:`dict`.

        :param data: the AMR text to be parsed each entry having keys
                     ``summary`` and ``body``

        :param doc_id: the document ID to set as
                       :obj:`.AmrFeatureDocument.doc_id`

        """
        def map_sent_type(stype: str) -> Iterable[SentenceType]:
            stkey: str = stype.upper()
            if stkey not in SentenceType._member_map_:
                mems: List[str] = SentenceType._member_names_
                raise AmrError(f"No such sentence type '{stkey}' in {mems}")
            return SentenceType[stype.upper()]

        doc_id: str = data.pop('id', None)
        comment: str = data.pop('comment', None)
        if doc_id is None:
            doc_id = str(self.doc_id)
            self.doc_id += 1
        sents: Tuple[AmrFeatureSentence] = tuple(ch.from_iterable(
            map(lambda t: self.from_str(t[1], t[0]),
                sorted(map(lambda t: (map_sent_type(t[0]), t[1]), data.items()),
                       key=lambda t: t[0]))))

        for sid, sent in enumerate(sents):
            sent.doc_sent_idx = sid
            sent.amr.set_metadata('id', f'{doc_id}.{sid}')
            if comment is not None:
                sent.amr.set_metadata('comment', comment)
        fdoc = AmrFeatureDocument(sents=tuple(sents))
        fdoc.amr = AnnotatedAmrDocument(
            sents=tuple(map(lambda s: s.amr, sents)))
        fdoc.amr.doc_id = doc_id
        if self.remove_wiki_attribs:
            fdoc.amr.remove_wiki_attribs()
        if self.remove_alignments:
            fdoc.amr.remove_alignments()
        return fdoc

    def from_dicts(self, data: List[Dict[str, str]]) -> \
            Iterable[AmrFeatureDocument]:
        """Parse and create an AMR documents from a list of :class:`dict`.

        :param data: the list of :class:`dict` processed by :meth:`from_dict`

        :param doc_ids: a document Id for each data processed

        :see: :meth:`from_dict`

        """
        return map(self.from_dict, data)

    def from_file(self, input_file: Path) -> Iterable[AmrFeatureDocument]:
        """Read annotated documents from a file and create AMR documents.

        :param input_file: the JSON file to read the doc text

        """
        with open(input_file) as f:
            return self.from_dicts(json.load(f))

    def from_data(self, data: Union[Path, Dict, Sequence]) -> \
            Iterable[AmrFeatureDocument]:
        """Create AMR documents based on the type of ``data``.

        :param data: the data that contains the annotated AMR document

        :see: :meth:`from_file`

        :see: :meth:`from_dicts`

        :see: :meth:`from_dict`

        """
        if isinstance(data, Path):
            return self.from_file(data)
        elif isinstance(data, Sequence):
            return self.from_dicts(data)
        elif isinstance(data, Dict):
            return self.from_dict(data)

    def __call__(self, data: Union[Path, Dict, Sequence]) -> \
            Iterable[AmrFeatureDocument]:
        """See :meth:`from_data`."""
        return self.from_data(data)


@dataclass
class CorpusWriter(Writable):
    """Writes :class:`.AmrDocument` instances to a file.  To use, first add
    documents either directly with :obj:`docs` or using the :meth:`add`.

    """
    anon_doc_factory: AnnotatedAmrFeatureDocumentFactory = field()
    """The factory used to create the :class:`.AmrFeatureDocument` instances
    that are in turn used to format that graphs as Penman text output.

    """
    def __post_init__(self):
        self._docs: List[AmrFeatureDocument] = []

    @property
    def docs(self) -> List[AmrDocument]:
        """The document to write."""
        return self._docs

    def add(self, data: Union[Path, Dict, Sequence]):
        """Add document(s) to this corpus writer.  This uses the
        :meth:`.AnnotatedAmrFeatureDocumentFactory.from_data` and adds the
        instances of :class:`.AmrFeatureDocument`.

        """
        self._docs.extend(self.anon_doc_factory.from_data(data))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """Write the contents of the documents added to this writer to
        ``writer`` as flat formatted Penman AMRs.

        :param depth: the starting indentation depth

        :param writer: the writer to dump the content of this writable

        """
        did: int
        doc: AmrFeatureDocument
        for did, doc in enumerate(self._docs):
            for sid, sent in enumerate(doc):
                if did > 0 or sid > 0:
                    writer.write('\n')
                sent.amr.write(depth, writer)


@dataclass
class FileCorpusWriter(CorpusWriter):
    """A corpus writer that parses a JSON file for its source input, then uses a
    the configured AMR parser to generate the graphs.

    """
    input_file: Path = field()
    """The JSON file as formatted per
    :class:`.AnnotatedAmrFeatureDocumentFactory`.

    """
    output_file: Path = field()
    """The file path to write the AMR sentences."""

    def __call__(self):
        par_dir: Path = self.output_file.parent
        self.add(self.input_file)
        if not par_dir.is_dir():
            logger.info(f'creating directory: {par_dir}')
            par_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            self.write(writer=f)
        logger.info(f'wrote: {self.output_file}')
