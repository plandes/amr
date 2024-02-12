
"""Continues training on an AMR model.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, List, Tuple, Set, Any, Type, Union, ClassVar, Callable
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import json
import random
import re
import os
import copy as cp
import shutil
from datetime import date
from pathlib import Path
import tarfile as tf
from zensols.config import Dictable
from zensols.persist import persisted
from zensols.install import Installer
from zensols.introspect import ClassImporter
from . import AmrError
from .corpprep import CorpusPrepperManager

logger = logging.getLogger(__name__)


@dataclass
class Trainer(Dictable, metaclass=ABCMeta):
    """Interface in to the :mod:`amrlib` package's trainers

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'pretrained_path_or_model', 'trainer_class', 'training_config'}

    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    _INFERENCE_MOD_REGEX: ClassVar[re.Pattern] = re.compile(
        r'.*((parse|generate)_[a-z\d]+).*')

    corpus_prep_manager: CorpusPrepperManager = field()
    """Aggregates and applies corpus prepare instances."""

    model_name: str = field()
    """Some human readable string identifying the model, and ends up in the
    ``amrlib_meta.json``.

    """
    output_model_dir: Path = field()
    """The path where the model is copied and metadata files generated."""

    temporary_dir: Path = field()
    """The path where the trained model is saved."""

    version: str = field(default='0.1.0')
    """The version used in the ``amrlib_meta.json`` output metadata file."""

    model_installer: Installer = field(default=None, repr=False)
    """The installer for the model used to train the model previously (i.e. by
    :mod:`amrlib`).

    """
    training_config_file: Path = field(default=None)
    """The path to the JSON configuration file in the ``amrlib`` repo in such as
    ``amrlib/configs/model_parse_*.json``.  If ``None``, then try to find the
    configuration file genereted by the last pretrained model.

    """
    training_config_overrides: Dict[str, Any] = field(default_factory=dict)
    """More configuration that overrides/clobbers from the contents found in
    :obj:`training_config_file`.

    """
    pretrained_path_or_model: Union[Path, str] = field(default=None)
    """The path to the checkpoint file or the string ``scratch`` if starting
    from scratch.

    """
    package_dir: Path = field(default=Path('.'))
    """The directory to install the compressed distribution file."""

    def __post_init__(self):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # reduce voluminous "Unhandled token X" and "ignoring epigraph data..."
        for n in ('amrlib.models.parse_xfm.penman_serializer',
                  'penman.layout'):
            log: logging.Logger = logging.getLogger(n)
            log.setLevel(logging.ERROR)

    def _get_pg(self) -> str:
        return 'parse'

    @property
    @persisted('_output_dir')
    def output_dir(self) -> Path:
        ver: str = self.version.replace('.', '_')
        pg: str = self._get_pg()
        model_name: str = f'model_{pg}_{self.model_name}-v{ver}'
        return self.output_model_dir / model_name

    @property
    def _pretrained_path_or_model(self) -> Union[str, Path]:
        """The path to the pretrained ``pytorch_model.bin`` file."""
        if self._pretrained_path_or_model_val is None:
            self.model_installer()
            return self.model_installer.get_singleton_path()
        else:
            return self._pretrained_path_or_model_val

    @_pretrained_path_or_model.setter
    def _pretrained_path_or_model(self, path: Path):
        self._pretrained_path_or_model_val = path

    @persisted('_input_metadata')
    def _get_input_metadata(self) -> str:
        self.model_installer()
        model_dir: Path = self.model_installer.get_singleton_path()
        meta_file: Path = model_dir / 'amrlib_meta.json'
        if not meta_file.is_file():
            raise AmrError(f'No metadata file: {meta_file}')
        else:
            with open(meta_file) as f:
                content = json.load(f)
            return content

    def _get_trainer_class(self, submod: str) -> Type:
        class_name = f'amrlib.models.{submod}.trainer.Trainer'
        return ClassImporter(class_name, False).get_class()

    @property
    @persisted('_training_class')
    def trainer_class(self) -> Type:
        """The AMR API class used for the training."""
        meta: Dict[str, str] = self._get_input_metadata()
        inf_mod: str = meta['inference_module']
        if inf_mod is not None:
            m: re.Match = self._INFERENCE_MOD_REGEX.match(inf_mod)
            if m is None:
                raise AmrError(
                    f'Can not parse amrlib training class module: {inf_mod}')
            submod: str = m.group(1)
            return self._get_trainer_class(submod)

    def _guess_training_config_file(self) -> Path:
        pt_path: Path = self.pretrained_path_or_model
        paths: Tuple[Path, ...] = tuple(pt_path.iterdir())
        path: Path = None
        cans: Tuple[Path, ...] = tuple(filter(
            lambda p: p.name.startswith('model') and p.suffix == '.json',
            paths))
        if len(cans) != 1:
            paths: str = ', '.join(map(lambda p: f"'{p.name}'", paths))
            logger.warning(
                f"expecting a single in '{pt_path}' file that starts " +
                f"with 'model' but got files: {paths}")
        else:
            path = cans[0]
        return path

    @property
    def _training_config_file(self) -> Path:
        path: Path = self._training_config_file_val
        if path is None:
            path = self._guess_training_config_file()
        if path is None:
            logger.warning('missing training config file')
            return path
        return path

    @_training_config_file.setter
    def _training_config_file(self, path: Path):
        self._training_config_file_val = path

    def _massage_training_config(self, config: Dict[str, Any]):
        overrides: Dict[str, Any] = self.training_config_overrides
        config.update(overrides)

    @persisted('_training_config_content')
    def _get_training_config_content(self) -> Dict[str, Any]:
        train_file: Path = self.training_config_file
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'loading train config from: {train_file}')
        if train_file is not None:
            config: Dict[str, Any]
            with open(train_file) as f:
                config = json.load(f)
            self._massage_training_config(config)
            return config

    @abstractmethod
    def _populate_training_config(self, config: Dict[str, Any]):
        pass

    @property
    @persisted('_training_config')
    def training_config(self) -> Dict[str, Any]:
        """The parameters given to the instance of the trainer, which is the
        class derived with :obj:`trainer_class`.

        """
        config: Dict[str, Any] = self._get_training_config_content()
        if config is not None:
            self._populate_training_config(config)
            return config

    def _get_base_model(self) -> str:
        pass

    def _write_metadata(self):
        meta: Dict[str, str] = self._get_input_metadata()
        path = self.output_dir / 'amrlib_meta.json'
        base_model: str = meta.get('base_model', self._get_base_model())
        content = {
            'model_type': 'stog',
            'version': self.version,
            'date': date.today().isoformat(),
            'inference_module': meta['inference_module'],
            'inference_class': 'Inference',
            'model_fn': 'pytorch_model.bin',
            'base_model': base_model,
            'kwargs': {}}
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f'writing metadata to {path}')
        with open(path, 'w') as f:
            json.dump(content, f, indent=4)

    def _package(self):
        """Create a compressed file with the model and metadata used by the
        :class:`~zensols.install.installer.Installer` using resource library
        ``amr_parser:installer``.  This is downloaded and used when this
        generated model is first use.

        """
        out_tar_file: Path = self.package_dir / f'{self.output_dir.stem}.tar.gz'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'compressing model: {self.output_dir}')
        out_tar_file.parent.mkdir(parents=True, exist_ok=True)
        with tf.open(out_tar_file, "w:gz") as tar:
            tar.add(self.output_dir, arcname=self.output_dir.name)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote compressed model: {out_tar_file}')

    @abstractmethod
    def _copy_model_files(self):
        pass

    @abstractmethod
    def _compile_model(self):
        pass

    @abstractmethod
    def _get_train_method(self) -> Callable:
        pass

    def _init_random(self, seed: int = 0):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(0)

    def _prepare_train(self):
        self.write_to_log(logger)
        dir_path: Path
        for dir_path in (self.temporary_dir, self.output_dir):
            logger.debug(f'removing directory: {dir_path}')
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
        self._init_random()
        self.corpus_prep_manager.prepare()

    def train(self, dry_run: bool = False):
        """Train the model (see class docs).

        :param dry_run: when ``True``, don't do anything, just act like it.

        """
        self._prepare_train()
        train: Callable = self._get_train_method()
        if not dry_run:
            logger.info(f'training model: {self.model_name}')
            train()
            self._compile_model()
            self._copy_model_files()
            self._package()


Trainer.pretrained_path_or_model = Trainer._pretrained_path_or_model
Trainer.training_config_file = Trainer._training_config_file


@dataclass
class HFTrainer(Trainer):
    """Interface in to the :mod:`amrlib` package's HuggingFace model trainers.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'token_model_name'} | Trainer._DICTABLE_ATTRIBUTES

    @property
    @persisted('_token_model_name')
    def token_model_name(self) -> str:
        """The name of the tokenziation model as the pretrained AMR model files
        do not have these files.

        """
        train_conf: Dict[str, Any] = self._get_training_config_content()
        if train_conf is not None:
            ga: Dict[str, str] = train_conf['gen_args']
            return ga['model_name_or_path']

    def _get_relative_paths(self) -> Tuple[Path, Path, Path]:
        stage_dir: Path = self.corpus_prep_manager.stage_dir
        training_file: Path = self.corpus_prep_manager.training_file
        dev_file: Path = self.corpus_prep_manager.dev_file
        training_file = training_file.relative_to(stage_dir)
        dev_file = dev_file.relative_to(stage_dir)
        return stage_dir, training_file, dev_file

    def _populate_training_config(self, config: Dict[str, Any]):
        paths: Tuple[Path, Path, Path] = self._get_relative_paths()
        ga: Dict[str, str] = config['gen_args']
        hf: Dict[str, str] = config['hf_args']
        model_or_path: Union[str, Path] = self.pretrained_path_or_model
        if isinstance(model_or_path, Path):
            model_or_path = str(model_or_path.absolute())
        ga['model_name_or_path'] = model_or_path
        ga['corpus_dir'] = str(paths[0])
        ga['train_fn'] = str(paths[1])
        ga['eval_fn'] = str(paths[2])
        ga['tok_name_or_path'] = self.token_model_name
        hf['output_dir'] = str(self.temporary_dir)

    def _get_base_model(self) -> str:
        config: Dict[str, Any] = self.training_config
        return config['gen_args']['model_name_or_path']

    def _write_config(self, config: Dict[str, any]):
        meta: Dict[str, str] = self._get_input_metadata()
        base_model: str = meta.get('base_model', self._get_base_model())
        pg: str = self._get_pg()
        cfile: Path = self.output_dir / f'model_{pg}_{self.model_name}.json'
        if base_model is None:
            raise AmrError('Missing base model name')
        config = cp.deepcopy(config)
        config['gen_args']['corpus_dir'] = \
            str(self.corpus_prep_manager.stage_dir)
        config['gen_args']['model_name_or_path'] = base_model
        cfile.parent.mkdir(parents=True)
        with open(cfile, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f'wrote model config: {cfile}')

    def _get_checkpoint_dir(self) -> Path:
        paths: Tuple[Path, ...] = tuple(self.temporary_dir.iterdir())
        cps = tuple(filter(lambda p: p.name.startswith('checkpoint'), paths))
        if len(cps) != 1:
            raise AmrError(
                f'Expecting 1 path at {self.temporary_dir} but got: {paths}')
        return cps[0]

    def _copy_model_files(self):
        fname: str = 'pytorch_model.bin'
        cp_dir: Path = self._get_checkpoint_dir()
        src: Path = cp_dir / fname
        dst: Path = self.output_dir / fname
        shutil.copy(src, dst)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'copied model weights and state: {dst}')

    def _rewrite_config(self):
        meta: Dict[str, str] = self._get_input_metadata()
        base_model: str = meta.get('base_model', self._get_base_model())
        config_file: Path = self._get_checkpoint_dir() / 'config.json'
        if base_model is None:
            raise AmrError('Missing base model name')
        with open(config_file) as f:
            content = json.load(f)
        content['_name_or_path'] = base_model
        pa: Dict[str, Any] = content['task_specific_params']['parse_amr']
        pa['corpus_dir'] = str(self.corpus_prep_manager.stage_dir)
        pa['model_name_or_path'] = base_model
        new_config: Path = self.output_dir / 'config.json'
        with open(new_config, 'w') as f:
            json.dump(content, f, indent=4)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote updated config: {new_config}')

    def _massage_training_config(self, config: Dict[str, Any]):
        overrides: Dict[str, Any] = self.training_config_overrides
        for k in 'gen_args hf_args model_args'.split():
            if k in self.training_config_overrides:
                config[k] = config[k] | overrides[k]
        # by 4.35 HF defaults to safe tensors, but amrlib models were
        # trained before, this
        config['hf_args']['save_safetensors'] = False

    def _get_train_method(self) -> Callable:
        config: Dict[str, Any] = self.training_config
        return self.trainer_class(config).train

    def _compile_model(self):
        config: Dict[str, Any] = self.training_config
        self._write_config(config)
        self._write_metadata()
        self._rewrite_config()


@dataclass
class XfmTrainer(HFTrainer):
    """Trainer for XFM and T5 models.

    """
    pass


@dataclass
class T5Trainer(XfmTrainer):
    """T5 model trainer.

    Citation:

      `Colin Raffel et al. 2020`_. Exploring the limits of transfer learning
      with a unified text-to-text transformer. The Journal of Machine Learning
      Research, 21(1):140:5485-140:5551, January.

    .. _Colin Raffel et al. 2020: https://jmlr.org/papers/volume21/20-074/20-074.pdf

    """
    def _get_trainer_class(self, submod: str) -> Type:
        # amrlib 7.1 uses the Xfm parser for the older T5 model
        return super()._get_trainer_class('parse_xfm')


@dataclass
class T5WithTenseGeneratorTrainer(XfmTrainer):
    nltk_lib_dir: Path = field(default=None)
    """Where to install the punkt tokenizer used by the trainer."""

    annotate_dir: Path = field(default=None)
    """The directory to add the annotated graphs."""

    annotate_model: str = field(default='en_core_web_sm')
    """The spaCy model used to annotate graphs as features to the model."""

    def __post_init__(self):
        super().__post_init__()
        self._anon_train: Path = self.annotate_dir / 'training' / 'training.txt'
        self._anon_eval: Path = self.annotate_dir / 'dev' / 'dev.txt'
        self._tag_train: Path = self.annotate_dir / 'training' / 'tagged.txt'

    def _get_pg(self) -> str:
        return 'generate'

    def _get_trainer_class(self, submod: str) -> Type:
        from amrlib.models.generate_xfm.trainer import Trainer
        return Trainer

    def _rewrite_config(self):
        config_file: Path = self._get_checkpoint_dir() / 'config.json'
        new_config: Path = self.output_dir / 'config.json'
        shutil.copy(config_file, new_config)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote updated config: {new_config}')

    def _install_deps(self):
        import nltk
        self.nltk_lib_dir.mkdir(parents=True, exist_ok=True)
        nltk.download('punkt', download_dir=str(self.nltk_lib_dir.absolute()))
        nltk.data.path.append(str(self.nltk_lib_dir.absolute()))

    def _annotate(self):
        """Annotate AMR graphs with tokens, POS tags and lemmas as done in
        :mod:`amrlib` ``10_Annotate_Corpus.py``.

        """
        from tqdm import tqdm
        from amrlib.graph_processing.annotator import annotate_file, load_spacy
        raw_paths: List[Path, Path] = self._get_relative_paths()[1:]
        stage_dir: Path = self.corpus_prep_manager.stage_dir
        # load the spacy model with the desired model
        logger.info(f'annotating graphs with model: {self.annotate_model}')
        load_spacy(self.annotate_model)
        # run the pipeline
        path: Path
        for path in tqdm(raw_paths):
            raw_path: Path = stage_dir / path
            anon_path: Path = self.annotate_dir / path
            anon_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'annotating {raw_path} -> {anon_path}')
            annotate_file('.', str(raw_path), '.', str(anon_path))

    def _tag(self):
        """From :mod:`amrlib` ``14_Create_Training_Data.py``: Take graphs that
        are annotated (tokens, pos, ...) and align them then tag the graphs.
        Save files with the tagged and untagged data together in a single
        training file.

        """
        from tqdm import tqdm
        from amrlib.graph_processing.amr_loading import load_amr_entries
        from amrlib.models.generate_xfm.model_input_helper \
            import ModelInputHelper
        entries = load_amr_entries(str(self._anon_train))
        tagged_entries = []
        logger.info(f'tagging {len(entries)} entries from: {self._anon_train}')
        for entry in tqdm(entries):
            tagged_entry = ModelInputHelper(entry).get_tagged_with_meta()
            tagged_entries.append(tagged_entry)
        # save the tagged and untagged entries into a single file, shuffled
        # together
        all_entries = entries + tagged_entries
        random.shuffle(all_entries)
        with open(self._tag_train, 'w') as f:
            for entry in all_entries:
                f.write(entry + '\n\n')
        logger.info(f'wrote: {self._tag_train}')

    def _populate_training_config(self, config: Dict[str, Any]):
        anon_dir: Path = self.annotate_dir
        super()._populate_training_config(config)
        config['gen_args']['corpus_dir'] = str(anon_dir)
        config['gen_args']['train_fn'] = \
            str(self._tag_train.relative_to(anon_dir))
        config['gen_args']['eval_fn'] = \
            str(self._anon_eval.relative_to(anon_dir))
        return config

    def _prepare_train(self):
        super()._prepare_train()
        self._install_deps()
        if not self.annotate_dir.is_dir():
            self._annotate()
            self._tag()


@dataclass
class SpringTrainer(Trainer):
    """SPRING model trainer.

    Citation:

        `Michele Bevilacqua et al. 2021`_. One SPRING to Rule Them Both:
        Symmetric AMR Semantic Parsing and Generation without a Complex
        Pipeline. In Proceedings of the AAAI Conference on Artificial
        Intelligence, volume 35, pages 12564–12573, Virtual, May.

    .. _Michele Bevilacqua et al. 2021: https://ojs.aaai.org/index.php/AAAI/article/view/17489

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'train_files', 'dev_files'} | Trainer._DICTABLE_ATTRIBUTES

    _SMATCH_RE: ClassVar[re.Pattern] = re.compile(
        r'^checkpoint.+smatch_([0-9]+)\.pt$')
    train_files: str = field(default=None)
    dev_files: str = field(default=None)

    def _populate_training_config(self, config: Dict[str, Any]):
        config['train'] = str(self.corpus_prep_manager.training_file)
        config['dev'] = str(self.corpus_prep_manager.dev_file)
        config['model_dir'] = str(self.temporary_dir.absolute())

    def _guess_training_config_file(self) -> Path:
        pt_path: Path = self.pretrained_path_or_model
        path: Path = pt_path / 'config.json'
        if path.is_file():
            return path

    @persisted('_training_config_content')
    def _get_training_config_content(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super()._get_training_config_content()
        pt_path: Union[str, Path] = self.pretrained_path_or_model
        if isinstance(pt_path, str):
            if pt_path == 'scratch':
                logger.info('training from scratch')
            else:
                config['model'] = pt_path
                logger.info(f'training from model: {pt_path}')
        return config

    def _write_config(self, config: Dict[str, any]):
        src_conf_path: Path = self.temporary_dir / 'config.json'
        dst_conf_path: Path = self.output_dir / 'config.json'
        meta: Dict[str, str] = dict(self._get_input_metadata())
        meta_file: Path = self.output_dir / 'amrlib_meta.json'
        meta_file.parent.mkdir(parents=True)
        meta['date'] = date.today().isoformat()
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=4)
        logger.info(f'wrote amrlib file {meta_file}')
        shutil.copy(src_conf_path, dst_conf_path)
        logger.info(f'copied spring {dst_conf_path}')

    def _compile_model(self):
        config: Dict[str, Any] = self.training_config
        self._write_config(config)

    def _get_checkpoint_file(self) -> Path:
        def map_smatch(p: Path):
            m: re.Match = self._SMATCH_RE.match(p.name)
            if m is not None:
                return (int(m.group(1)), p)

        by_smatch: Tuple[Path, ...] = tuple(map(
            lambda t: t[1],
            sorted(
                filter(
                    lambda t: t is not None,
                    map(map_smatch, self.temporary_dir.iterdir())),
                key=lambda t: t[0])))
        if len(by_smatch) < 1:
            raise AmrError(
                f'Expecting at least one one path in {self.temporary_dir} ' +
                f'with pattern: {self._SMATCH_RE}')
        return by_smatch[0]

    def _copy_model_files(self):
        cp_file: Path = self._get_checkpoint_file()
        dst: Path = self.output_dir / 'model.pt'
        shutil.copy(cp_file, dst)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'copied model weights and state: {dst}')

    def _get_train_method(self) -> Callable:
        config: Dict[str, Any] = self.training_config
        pt_path: Union[str, Path] = self.pretrained_path_or_model
        cp: str = None
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'setup spring config, pretrain path: {pt_path}')
        if isinstance(pt_path, Path):
            cp = str(self.pretrained_path_or_model.absolute() / 'model.pt')
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'using check point: {cp}')
        trainer: Callable = self.trainer_class(config)
        train_fn: Callable = trainer.train
        train = (lambda: train_fn(checkpoint=cp))
        return train


SpringTrainer.training_config_file = SpringTrainer._training_config_file
