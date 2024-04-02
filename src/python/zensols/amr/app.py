"""Adapts :mod:`amrlib` in the Zensols framework.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Dict, Any, List, Set, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import sys
import re
from pathlib import Path
import json
import itertools as it
import pandas as pd
from zensols.introspect import ClassImporter
from zensols.config import Settings, ConfigFactory, DictionaryConfig
from zensols.persist import Stash
from zensols.cli import LogConfigurator, ApplicationError
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from . import AmrError, AmrSentence, AmrDocument, AmrFeatureSentence

logger = logging.getLogger(__name__)


class Format(Enum):
    """Format output type for AMR corpous documents.

    """
    txt = auto()
    json = auto()
    csv = auto()

    @classmethod
    def to_ext(cls, f: Format) -> str:
        return f.name


@dataclass
class BaseApplication(object):
    """Base class for applications.

    """
    log_config: LogConfigurator = field()
    """Used to update logging levels based on the ran action."""

    def _normalize_huggingface_logging(self):
        """Make HF APIs using the logging system rather than to stdout."""
        from transformers import logging
        logging.disable_default_handler()
        logging.enable_propagation()

    def _set_level(self, level: int, verbose: bool = False):
        self.log_config.level = level
        self.log_config()
        if verbose:
            # this doesn't cross over to (multi) sub-processes; for that set
            # log configuration in app.conf or train configuration
            for n in 'persist multi install'.split():
                logging.getLogger(f'zensols.{n}').setLevel(logging.INFO)

    def _key_split(self, keys: str) -> List[str]:
        return re.split(r'\s*,\s*', keys)


@dataclass
class Application(BaseApplication):
    """Parse and plot AMR graphs in Penman notation.

    """
    config_factory: ConfigFactory = field()
    """Application context used by programmatic clients of this class."""

    doc_parser: FeatureDocumentParser = field()
    """The feature document parser for the app.  This is not done via the
    application config to allow overriding of the defaults.

    """
    anon_doc_stash: Stash = field()
    """The annotated document stash."""

    dumper: 'Dumper' = field()
    """Plots and writes AMR content in human readable formats."""

    def count(self, input_file: Path):
        """Provide counts on an AMR corpus file.

        :param input_file: a file with newline separated AMR Penman graphs

        """
        amr_doc: AmrDocument = AmrDocument.from_source(input_file)
        print(f'{len(amr_doc)} {input_file}')

    def write_metadata(self, input_file: Path, output_dir: Path = None):
        """Write the metadata of each AMR in a corpus file.

        :param input_file: a file with newline separated AMR Penman graphs

        :param output_dir: the output directory

        """
        metas: List[Dict[str, str]] = []
        if output_dir is None:
            output_dir = input_file.parent
        output_file: Path = output_dir / f'{input_file.stem}.csv'
        sent: AmrSentence
        for sent in AmrDocument.from_source(input_file):
            try:
                metas.append(sent.metadata)
            except AmrError as e:
                logger.error(f'could not parse AMR: {e}--skipping')
        df = pd.DataFrame(metas)
        cols: List[str] = df.columns.tolist()
        if 'id' in cols:
            cols.remove('id')
            df = df[['id'] + cols]
        df.to_csv(output_file, index=False)
        logger.info(f'wrote: {output_file}')
        return df

    def plot(self, text: str, output_dir: Path = None):
        """Parse a sentence in to an AMR graph.

        :param text: the sentence(s) to parse or a number a pre-written
                     sentence

        :param output_dir: the output directory

        """
        if output_dir is not None:
            self.dumper.target_dir = output_dir
        doc: FeatureDocument = self.doc_parser(text)
        amr_doc: AmrDocument = doc.amr
        for path in self.dumper(amr_doc):
            logger.debug(f'wrote: {path}')

    @staticmethod
    def _to_paths(input_path: Path) -> Tuple[Path, ...]:
        paths: Tuple[Path, ...]
        if not input_path.exists():
            raise ApplicationError(
                f'File or directory does not exist: {input_path}')
        if input_path.is_dir():
            paths = tuple(input_path.iterdir())
        else:
            paths = (input_path,)
        return paths

    def _plot_sents(self, pfile: Path, output_dir: Path):
        logger.debug(f'plotting plan: {pfile}')
        with open(pfile) as f:
            plans: List[Dict[str, Any]] = json.load(f)
        plan: Dict[str, Any]
        for plan in plans:
            config: Dict[str, Any] = plan.get('config')
            if config is not None:
                conf = DictionaryConfig(dict(default=config))
                config = conf.populate({})
            sents: Dict[str, Any] = plan.get('sents')
            if sents is None:
                raise ApplicationError(
                    f"Expecting 'sents' for each plan: {pfile}")
            for sent in sents:
                text: str = sent.get('text')
                path_str: str = sent.get('path')
                if text is None:
                    gstr: str = sent.get('penman')
                    if gstr is None:
                        raise ApplicationError(
                            f"Need 'text' or 'penman' in each plan: {pfile}")
                    doc: AmrDocument = AmrDocument.from_source(pfile)
                    assert len(doc.sents) == 1
                    sent = doc.sents[0]
                else:
                    fdoc: FeatureDocument = self.doc_parser(text)
                    sent = fdoc.amr.sents[0]
                path: Path
                if path_str is None:
                    path = Path(f'{sent.short_name}.pdf')
                else:
                    path = Path(path_str)
                if output_dir is not None:
                    path = output_dir / path
            self.dumper.write_text = False
            self.dumper.overwrite_dir = False
            self.dumper.overwrite_sent_file = True
            self.dumper.target_dir = path.parent
            self.dumper.extension = path.suffix[1:]
            if config is not None:
                self.dumper.__dict__.update(config)
            self.dumper.plot_sent(sent, path.stem)

    def plot_file(self, input_file: Path, output_dir: Path = None):
        """Render a Penman files or a JSON formatted sentence list.

        :param input_file: a file with newline separated AMR Penman graphs

        :param output_dir: the output directory

        """
        self.dumper.robust = True
        path: Path
        for path in self._to_paths(input_file):
            if path.suffix == '.json':
                self._plot_sents(path, output_dir)
            else:
                logger.info(f'rendering {path}')
                doc: AmrDocument = AmrDocument.from_source(path)
                if output_dir is not None:
                    self.dumper.target_dir = output_dir
                self.dumper(doc)

    def parse(self, text: str):
        """Parse the natural language text to an AMR graphs.

        :param text: the sentence(s) to parse

        """
        doc: FeatureDocument = self.doc_parser(text)
        amr_doc: AmrDocument = doc.amr
        print(amr_doc.graph_string)

    def clear(self):
        """Clear all cached parsed AMR documents and data."""
        self.anon_doc_stash.clear()


@dataclass
class ScorerApplication(BaseApplication):
    """Creates parsed files for comparing, and scores.

    """
    config_factory: ConfigFactory = field()
    """Application context."""

    doc_factory: 'AmrFeatureDocumentFactory' = field()
    """Creates :class:`.AmrFeatureDocument` from :class:`.AmrDocument`
    instances.

    """
    @staticmethod
    def _to_alt_path(path: Path, output_dir: Path, suffix: str) -> Path:
        output_file: str = f'{path.stem}{suffix}{path.suffix}'
        output_path: Path
        if output_dir is None:
            output_path = path.parent / output_file
        else:
            output_dir = output_dir.expanduser()
            # the user specified a file
            if not output_dir.is_dir():
                output_path = output_dir
            else:
                output_path = output_dir / output_file
        return output_path

    def parse_penman(self, input_file: Path, output_dir: Path = None,
                     meta_keys: str = 'id,snt',
                     limit: int = None) -> List[Path]:
        """Parse Penman sentence(s) by ``id`` and write a parsed AMR.

        :param input_file: a file with newline separated AMR Penman graphs

        :param output_dir: the output directory

        :param meta_keys: a comma separated list of metadata keys

        :param limit: the max of items to process

        """
        from zensols.amr.score import AmrScoreParser
        score_parser: AmrScoreParser = \
            self.config_factory('amr_score_parser')
        if meta_keys is not None:
            score_parser.keep_keys = self._key_split(meta_keys)
        output_paths: List[Path] = []
        path: Path
        for path in Application._to_paths(input_file):
            doc: AmrDocument = AmrDocument.from_source(path)
            output_path: Path = self._to_alt_path(path, output_dir, '-parsed')
            with open(output_path, 'w') as f:
                first_sent: bool = True
                sent: AmrSentence
                for sent in it.islice(doc.sents, limit):
                    amr_parse_sent: AmrSentence = score_parser.parse(sent)
                    if first_sent:
                        first_sent = False
                    else:
                        f.write('\n')
                    f.write(amr_parse_sent.graph_string)
                    f.write('\n')
            logger.info(f'wrote {output_path}')
            output_paths.append(output_path)
        return output_paths

    def remove_wiki(self, input_file: Path, output_dir: Path = None):
        """Remove wiki attributes necessary for scoring.

        :param input_file: a file with newline separated AMR Penman graphs

        :param output_dir: the output directory

        """
        path: Path
        for path in Application._to_paths(input_file):
            doc: AmrDocument = AmrDocument.from_source(path)
            output_path: Path = self._to_alt_path(path, output_dir, '-rmwiki')
            logger.info(f'removing wiki attributes: {path} -> {output_path}')
            with open(output_path, 'w') as f:
                first_sent: bool = True
                sent: AmrSentence
                for sent in doc.sents:
                    sent.remove_wiki_attribs()
                    if first_sent:
                        first_sent = False
                    else:
                        f.write('\n')
                    f.write(sent.graph_string)
                    f.write('\n')
            logger.info(f'wrote {output_path}')

    def _sentences_by_id(self, path: Path) -> Dict[str, AmrSentence]:
        by_id: Dict[str, AmrSentence] = {}
        if not path.is_file():
            raise ApplicationError(f'No such AMR file: {path}')
        sent: AmrSentence
        for sent in AmrDocument.from_source(path):
            try:
                meta: Dict[str, str] = sent.metadata
                sid: str = meta.get('id')
                if sid is None:
                    text: str = meta.get('snt')
                    logger.warning(f'no sentence ID for: <{text}>--skipping')
                    continue
                by_id[sid] = sent
            except AmrError as e:
                logger.error(f'could not parse AMR: {e}--skipping')
        return by_id

    def _to_feature_sents(self, sents: Iterable[AmrSentence]) -> \
            Iterable[AmrSentence]:
        def map_sent(s: AmrSentence) -> AmrFeatureSentence:
            doc = AmrDocument.to_document([s])
            return self.doc_factory.to_feature_doc(doc).sents[0]

        return map(map_sent, sents)

    def score(self, input_gold: Path, input_parsed: Path = None,
              output_dir: Path = None, output_format: Format = Format.csv,
              limit: int = None, methods: str = None) -> 'ScoreSet':
        """Score AMRs by ID and dump the results to a file or directory.

        :param input_gold: the file containing the gold AMR graphs

        :param input_parsed: the file containing the parser output graphs,
                             defaults to ``gold-parsed.txt``

        :param output_dir: the output directory

        :param output_format: the output format

        :param limit: the max of items to process

        :param methods: a comma separated list of scoring methods

        """
        from zensols.nlp.score import Scorer, ScoreContext, ScoreSet
        scorer: Scorer = self.config_factory('nlp_scorer')
        if input_parsed is None:
            input_parsed: Path = self._to_alt_path(
                input_gold, output_dir, '-parsed')
        if output_dir is None:
            output_dir = Path('.')
        output_file: Path = self._to_alt_path(
            input_gold, output_dir, '-scored')
        output_file = output_file.parent / \
            f'{output_file.stem}.{Format.to_ext(output_format)}'
        limit = sys.maxsize if limit is None else limit
        gold: Dict[str, AmrSentence] = self._sentences_by_id(input_gold)
        parsed: Dict[str, AmrSentence] = self._sentences_by_id(input_parsed)
        ukeys: Set[str] = set(gold.keys()) | set(parsed.keys())
        ikeys: Set[str] = set(gold.keys()) & set(parsed.keys())
        key_diff: Set[str] = ukeys - ikeys
        if len(key_diff) > 0:
            if len(key_diff) > 100:
                logger.warning(f'skipping {len(key_diff)} disjoint ids')
            else:
                logger.warning(f'skipping disjoint IDs: {key_diff}')
        ikeys: List[str] = sorted(ikeys)
        gold_sents = self._to_feature_sents(map(lambda i: gold[i], ikeys))
        parsed_sents = self._to_feature_sents(map(lambda i: parsed[i], ikeys))
        logger.info(f'scoring <{input_gold}>:<{input_parsed}> -> {output_file}')
        sctx = ScoreContext(
            pairs=tuple(it.islice(zip(gold_sents, parsed_sents), limit)),
            methods=None if methods is None else set(self._key_split(methods)),
            correlation_ids=tuple(it.islice(ikeys, limit)))
        sset: ScoreSet = scorer.score(sctx)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            {
                Format.txt: lambda: sset.write(writer=f),
                Format.json: lambda: f.write(sset.asjson()),
                Format.csv: lambda: sset.as_dataframe().to_csv(f, index=False),
            }[output_format]()
        logger.info(f'wrote {output_file}')
        return sset


@dataclass
class TrainerApplication(BaseApplication):
    """Trains and evaluates models.

    """
    config_factory: ConfigFactory = field()
    """Application context."""

    @property
    def trainer(self) -> 'Trainer':
        """Interface in to the :mod:`amrlib` package's trainer.  This is not
        done via the application config to allow overriding of the defaults.

        """
        from .trainer import Trainer
        tdefs: Settings = self.config_factory.config['amr_trainer_default']
        trainer_type: str = tdefs['trainer_type']
        gp: bool = 'generate' if tdefs['trainer_is_generator'] else 'parse'
        sec: str = f'amr_{gp}_{trainer_type}_trainer'
        trainer: Trainer = self.config_factory(sec)
        sup_classes: Set[str] = set(map(
            ClassImporter.full_classname,
            trainer.__class__.__mro__))
        # compare by class names for prototyping app config reloads
        if not ClassImporter.full_classname(Trainer) in sup_classes:
            raise AmrError(
                f"Wrong trainer configuration: '{sec}' ({type(trainer)})")
        return trainer

    def _get_text(self, text_or_file: str):
        path = Path(text_or_file)
        if path.is_file():
            with open(path) as f:
                return f.read().strip()
        return text_or_file

    def write_corpus(self, text_or_file: str, out_file: Path = None):
        """Write a corpus from ad hoc text.

        :param text_or_file: if the file exists, use the contents of the file,
                             otherwise, the sentence(s) to parse

        :param out_file: the output file

        """
        from .annotate import CorpusWriter
        out_file = Path('corpus.txt') if out_file is None else out_file
        writer: CorpusWriter = self.config_factory.new_instance(
            'amr_corpus_writer', path=out_file)
        for text in self._get_text(text_or_file).split('\n'):
            writer.add(text)
        writer()
        logger.info(f'wrote: {out_file}')

    def train(self, dry_run: bool = False):
        """Continue fine tuning on additional corpora.

        :param dry_run: don't do anything; just act like it

        """
        self._set_level(logging.INFO, True)
        self.trainer.train(dry_run)

    def prep_corpus(self):
        """Download and install the training corpus."""
        from .corpprep import CorpusPrepperManager
        prepper: CorpusPrepperManager = self.config_factory('amr_prep_manager')
        prepper.prepare()

    def restore_splits(self, output_dir: Path = None, id_pattern: str = None):
        """Restore corpus splits used for training.

        :param output_dir: the output directory

        :param id_pattern: the AMR metadata ID regular expression to match

        """
        output_dir = Path('corpus-splits') if output_dir is None else output_dir
        if id_pattern is not None:
            id_pattern = re.compile(id_pattern)
        self.trainer.corpus_prep_manager.restore_splits(output_dir, id_pattern)
