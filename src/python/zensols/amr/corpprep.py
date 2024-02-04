"""Prepare and compile AMR corpora for training.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Set, Iterable, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import random
from pathlib import Path
from io import TextIOBase
import shutil
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
    TRAINING_SUBDIR: ClassVar[str] = 'training'
    """The training dataset subdirectory name."""

    DEV_SUBDIR: ClassVar[str] = 'dev'
    """The development / validation dataset subdirectory name."""

    name: str = field()
    """Used for logging and directory naming."""

    installer: Installer = field(repr=False)
    """The location and decompression details."""

    transform_ascii: bool = field(default=True)
    """Whether to replace non-ASCII characters for models."""

    shuffle: bool = field(default=True)
    """Whether to shuffle the AMR sentences before writing to the target
    directory.

    """
    @abstractmethod
    def _read_files(self, target: Path) -> Iterable[Tuple[Path, AmrDocument]]:
        """Read and return tuples of where to write the output of the sentences
        of the corresponding document.

        :param target: the location of where to copy the finished files

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

    def prepare(self, target: Path):
        """Download, install and write the corpus to disk.  The data is then
        ready for AMR parser and generator trainers.

        :param target: the location of where to copy the finished files

        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'preparing {self}')
        self.installer.install()
        with time(f'prepared corpus {self}'):
            path: Path
            doc: AmrDocument
            for path, doc in self._read_files(target):
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
    def _read_files(self, target: Path) -> Iterable[Tuple[Path, AmrDocument]]:
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
        yield (target / self.TRAINING_SUBDIR / f'{self.name}.txt', train)
        yield (target / self.DEV_SUBDIR / f'{self.name}.txt', dev)


@dataclass
class AmrReleaseCorpusPrepper(CorpusPrepper):
    """Writes the `AMR 3 release`_ corpus files.

    .. AMR 3 release: https://catalog.ldc.upenn.edu/LDC2020T02

    """
    def _read_files(self, target: Path) -> Iterable[Tuple[Path, AmrDocument]]:
        split_names: Set[str] = {self.TRAINING_SUBDIR, self.DEV_SUBDIR}
        splits_path: Path = self.installer.get_singleton_path()
        for amr_file in splits_path.glob('**/*.txt'):
            split: str = amr_file.parent.name
            split = self.TRAINING_SUBDIR if split == 'test' else split
            assert split in split_names
            out_file: Path = target / split / amr_file.name
            doc: AmrDocument = self._load_doc(amr_file)
            yield (out_file, doc)


@dataclass
class CorpusPrepperManager(Dictable):
    """Aggregates and applies corpus prepare instances.

    """
    stage_dir: Path = field()
    """The location of where to copy the finished files."""

    preppers: Tuple[CorpusPrepper] = field()
    """The corpus prepare instances used to create the training files."""

    @property
    def is_done(self) -> bool:
        """Whether or not the preparation is already complete."""
        return self.stage_dir.is_dir()

    @property
    def training_dir(self) -> Path:
        """The training dataset directory."""
        return self.stage_dir / CorpusPrepper.TRAINING_SUBDIR

    @property
    def dev_dir(self) -> Path:
        """The deveopment / validation dataset directory."""
        return self.stage_dir / CorpusPrepper.DEV_SUBDIR

    def prepare(self):
        """Download, install and write the corpus to disk from all
        :obj:`preppers`.  The output of each is placed in the corresponding
        ``training`` or ``dev`` directories in :obj:`stage_dir`.  The data is
        then ready for AMR parser and generator trainers.

        """
        if self.is_done:
            logger.info('corpus preparation is already complete')
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'preparing corpus in {self.stage_dir}')
            for prepper in self.preppers:
                prepper.prepare(self.stage_dir)

    def clear(self):
        if self.is_done:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'removing corpus prep staging: {self.stage_dir}')
            shutil.rmtree(self.stage_dir)
        else:
            logger.info('no corpus preparation found')
