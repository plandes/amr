"""Prepare and compile AMR corpora for training.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Iterable
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import random
from pathlib import Path
from io import TextIOBase
from zensols.util.time import time
from zensols.config import Dictable
from zensols.install import Installer
from . import AmrSentence, AmrDocument

logger = logging.getLogger(__name__)


@dataclass
class CorpusPrepper(Dictable, metaclass=ABCMeta):
    """Subclasses know where to download, install, and split the corpus in to
    train and dev data sets.  Each subclass generates only the training and
    dev/validation datasets, which is an aspect of AMR parser and text
    generation models.  Both the input and outupt are Penman encoded AMR graphs.

    """
    name: str = field()
    """Used for logging and directory naming."""

    installer: Installer = field()
    """The location and decompression details."""

    stage_dir: Path = field()
    """The location of where to copy the finished files."""

    transform_ascii: bool = field(default=True)
    """Whether to replace non-ASCII characters for models."""

    shuffle: bool = field(default=True)
    """Whether to shuffle the AMR sentences in the files added to
    :obj:`stage_dir`.

    """
    @abstractmethod
    def _read_files(self) -> Iterable[Tuple[Path, AmrDocument]]:
        """Read and return tuples of where to write the output of the sentences
        of the corresponding document.

        """
        pass

    def _load_doc(self, path: Path) -> AmrDocument:
        """Load text from ``path`` and return the sentences as a document."""
        doc = AmrDocument.from_source(
            path, transform_ascii=self.transform_ascii)
        if self.shuffle:
            sents = list(doc.sents)
            random.shuffle(sents)
            doc.sents = tuple(sents)
        return doc

    def _write(self, writer: TextIOBase, doc: AmrDocument):
        """Write the ``doc`` to data sink ``writer`` with newlines between each
        in a format for AMR parser/generators trainers.

        """
        dlen_m1: int = len(doc) - 1
        sent: AmrSentence
        for i, sent in enumerate(doc):
            writer.write(sent.graph_string)
            if i < dlen_m1:
                writer.write('\n\n')
        writer.write('\n')

    def prepare(self):
        """Download, install and write the corpus to disk in the
        :obj:`stage_dir` directory.  The data is then ready for AMR parser and
        generator trainers.

        """
        self.installer.install()
        with time(f'prepared corpus {self}'):
            path: Path
            doc: AmrDocument
            for path, doc in self._read_files():
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'writing: {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    self._write(f, doc)

    def __str__(self):
        return str(self.name)


@dataclass
class SingletonCorpusPrepper(CorpusPrepper):
    """Prepares the corpus training files from a single AMR Penman encoded file.

    """
    dev_portion: float = field(default=0.15)
    """The portion of the dev/validation set in sentences of the single input
    file.

    """
    def _read_files(self) -> Iterable[Tuple[Path, AmrDocument]]:
        corp_file: Path = self.installer.get_singleton_path()
        doc: AmrDocument = self._load_doc(corp_file)
        sents: Tuple[AmrSentence] = doc.sents
        n_sents: int = len(doc)
        n_dev: int = round(self.dev_portion * n_sents)
        dev: AmrDocument = doc.from_sentences(sents[:n_dev])
        train: AmrDocument = doc.from_sentences(sents[n_dev:])
        assert (len(dev) + len(train)) == n_sents
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'{self}, file={corp_file.name}: dev={len(dev)}, ' +
                        f'train={len(train)}, total={n_sents}')
        yield (self.stage_dir / 'training' / f'{self.name}.txt', train)
        yield (self.stage_dir / 'dev' / f'{self.name}.txt', dev)


@dataclass
class AmrReleaseCorpusPrepper(CorpusPrepper):
    """Writes the `AMR 3 release`_ corpus files.

    .. AMR 3 release: https://catalog.ldc.upenn.edu/LDC2020T02

    """
    def _read_files(self) -> Iterable[Tuple[Path, AmrDocument]]:
        splits_path: Path = self.installer.get_singleton_path()
        for amr_file in splits_path.glob('**/*.txt'):
            split: str = amr_file.parent.name
            split = 'training' if split == 'test' else split
            out_file: Path = self.stage_dir / split / amr_file.name
            doc: AmrDocument = self._load_doc(amr_file)
            yield (out_file, doc)


@dataclass
class CorpusPrepperManager(Dictable):
    """Aggregates and applies corpus prepare instances.

    """
    preppers: Tuple[CorpusPrepper] = field()
    """The corpus prepare instances used to create the training files."""

    def prepare(self):
        """Download, install and write the corpus to disk from all
        :obj:`preppers`.  The output of each is placed in the corresponding
        ``training`` or ``dev`` directories in :obj:`stage_dir`.  The data is
        then ready for AMR parser and generator trainers.

        """
        for prepper in self.preppers:
            prepper.prepare()
