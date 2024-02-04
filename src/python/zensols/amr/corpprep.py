"""Prepare and compile AMR corpora for training.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Set, Dict, List, Iterable, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import random
import collections
import json
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
    TRAINING_SPLIT_NAME: ClassVar[str] = 'training'
    """The training dataset name."""

    DEV_SPLIT_NAME: ClassVar[str] = 'dev'
    """The development / validation dataset name."""

    name: str = field()
    """Used for logging and directory naming."""

    installer: Installer = field(repr=False)
    """The location and decompression details."""

    transform_ascii: bool = field(default=True)
    """Whether to replace non-ASCII characters for models."""

    @abstractmethod
    def read_docs(self, target: Path) -> Iterable[Tuple[str, AmrDocument]]:
        """Read and return tuples of where to write the output of the sentences
        of the corresponding document.

        :param target: the location of where to copy the finished files

        :return: tuples of the dataset name and the read document

        """
        pass

    def _load_doc(self, path: Path) -> AmrDocument:
        """Load text from ``path`` and return the sentences as a document."""
        return AmrDocument.from_source(
            path, transform_ascii=self.transform_ascii)

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
    def read_docs(self, target: Path) -> Iterable[Tuple[str, AmrDocument]]:
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
        yield (self.TRAINING_SPLIT_NAME, train)
        yield (self.DEV_SPLIT_NAME, dev)


@dataclass
class AmrReleaseCorpusPrepper(CorpusPrepper):
    """Writes the `AMR 3 release`_ corpus files.

    .. AMR 3 release: https://catalog.ldc.upenn.edu/LDC2020T02

    """
    def read_docs(self, target: Path) -> Iterable[Tuple[str, AmrDocument]]:
        split_names: Set[str] = {self.TRAINING_SPLIT_NAME, self.DEV_SPLIT_NAME}
        splits_path: Path = self.installer.get_singleton_path()
        for amr_file in splits_path.glob('**/*.txt'):
            split: str = amr_file.parent.name
            split = self.TRAINING_SPLIT_NAME if split == 'test' else split
            assert split in split_names
            doc: AmrDocument = self._load_doc(amr_file)
            yield (split, doc)


@dataclass
class CorpusPrepperManager(Dictable):
    """Aggregates and applies corpus prepare instances.

    """
    preppers: Tuple[CorpusPrepper] = field()
    """The corpus prepare instances used to create the training files."""

    stage_dir: Path = field()
    """The location of where to copy the finished files."""

    shuffle: bool = field(default=True)
    """Whether to shuffle the AMR sentences before writing to the target
    directory.

    """
    key_splits: Path = field(default=None)
    """The AMR ``id``s from the sentence metadatas for each split are written to
    this JSON file if specified.

    """
    @property
    def is_done(self) -> bool:
        """Whether or not the preparation is already complete."""
        return self.stage_dir.is_dir()

    @property
    def training_file(self) -> Path:
        """The training dataset directory."""
        name: str = CorpusPrepper.TRAINING_SPLIT_NAME
        return self.stage_dir / name / f'{name}.txt'

    @property
    def dev_file(self) -> Path:
        """The deveopment / validation dataset directory."""
        name: str = CorpusPrepper.DEV_SPLIT_NAME
        return self.stage_dir / name / f'{name}.txt'

    def _split_to_file(self, split_name: str):
        return {
            CorpusPrepper.TRAINING_SPLIT_NAME: self.training_file,
            CorpusPrepper.DEV_SPLIT_NAME: self.dev_file,
        }[split_name]

    def _write(self, writer: TextIOBase, sents: List[AmrSentence]):
        """Write the ``doc`` to data sink ``writer`` with newlines between each
        in a format for AMR parser/generators trainers.

        """
        dlen_m1: int = len(sents) - 1
        for i, sent in enumerate(sents):
            writer.write(sent.graph_string)
            if i < dlen_m1:
                writer.write('\n\n')
        writer.write('\n')

    def _prepare(self) -> Dict[str, int]:
        """Prepare the corpus (see :meth:`prepare`) and return the number of
        sentences added for each split.

        """
        def map_keys(sent: AmrSentence) -> Tuple[str]:
            meta: Dict[str, str] = sent.metadata
            if 'id' in meta:
                return meta['id']

        sents: Dict[str, List[AmrSentence]] = collections.defaultdict(list)
        prepper: CorpusPrepper
        for prepper in self.preppers:
            prepper.installer.install()
            split_name: str
            doc: AmrDocument
            for split_name, doc in prepper.read_docs(self.stage_dir):
                sents[split_name].extend(doc.sents)
        if self.shuffle:
            sent_set: List[AmrSentence]
            for sent_set in sents.values():
                random.shuffle(sent_set)
        split_name: str
        sent_set: List[AmrSentence]
        for split_name, sent_set in sents.items():
            out_path: Path = self._split_to_file(split_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                self._write(f, sent_set)
            logger.info(f'wrote: {out_path}')
        if self.key_splits is not None:
            keys: Dict[str, Tuple[str]] = {}
            for split_name, sent_set in sents.items():
                keys[split_name] = tuple(filter(
                    lambda t: t is not None,
                    map(map_keys, sent_set)))
            with open(self.key_splits, 'w') as f:
                json.dump(keys, f, indent=4)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'wrote: {self.key_splits}')
        return {k: len(sents[k]) for k in sents.keys()}

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
            with time('wrote {total} sentences ({sstr})'):
                stats: Dict[str, int] = self._prepare()
                sstr: str = ', '.join(map(
                    lambda t: f'{t[0]}: {t[1]}', stats.items()))
                total: int = sum(stats.values())

    def clear(self):
        if self.is_done:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'removing corpus prep staging: {self.stage_dir}')
            shutil.rmtree(self.stage_dir)
        else:
            logger.info('no corpus preparation found')
