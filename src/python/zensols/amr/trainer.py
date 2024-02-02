"""Continues training on an AMR model.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Tuple, Set, Any, Type, Union, ClassVar, Callable
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import json
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
from . import AmrError, AmrParser

logger = logging.getLogger(__name__)


@dataclass
class Trainer(Dictable, metaclass=ABCMeta):
    """Interface in to the :mod:`amrlib` package's trainers

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'pretrained_path_or_model', 'trainer_class', 'training_config'}

    _INFERENCE_MOD_REGEX: ClassVar[re.Pattern] = re.compile(
        r'.*(parse_[a-z]+).*')

    corpus_installer: Installer = field(repr=False)
    """Points to the AMR corpus file(s)."""

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

    parser: AmrParser = field(default=None, repr=False)
    """The parser that contains the :mod:`amrlib` pretrained model, which is
    loaded and used to continue fine-tuning.  If not set, the training starts
    from the T5 HuggingFace pretrained model.

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

    @property
    @persisted('_output_dir')
    def output_dir(self) -> Path:
        ver = self.version.replace('.', '_')
        model_name = f'model_parse_{self.model_name}-v{ver}'
        return self.output_model_dir / model_name

    @property
    def _pretrained_path_or_model(self) -> Union[str, Path]:
        """The path to the pretrained ``pytorch_model.bin`` file."""
        if self._pretrained_path_or_model_val is None:
            return self.parser.installer.get_singleton_path()
        else:
            return self._pretrained_path_or_model_val

    @_pretrained_path_or_model.setter
    def _pretrained_path_or_model(self, path: Path):
        self._pretrained_path_or_model_val = path

    @persisted('_input_metadata')
    def _get_input_metadata(self) -> str:
        model_dir: Path = self.parser.installer.get_singleton_path()
        meta_file: Path = model_dir / 'amrlib_meta.json'
        self.parser.installer()
        if not meta_file.is_file():
            raise AmrError(f'No metadata file: {meta_file}')
        else:
            with open(meta_file) as f:
                content = json.load(f)
            return content

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
            mod = m.group(1)
            class_name = f'amrlib.models.{mod}.trainer.Trainer'
            return ClassImporter(class_name, False).get_class()

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

    def _write_metadata(self):
        meta: Dict[str, str] = self._get_input_metadata()
        path = self.output_dir / 'amrlib_meta.json'
        content = {
            'model_type': 'stog',
            'version': self.version,
            'date': date.today().isoformat(),
            'inference_module': meta['inference_module'],
            'inference_class': 'Inference',
            'model_fn': 'pytorch_model.bin',
            'base_model': meta['base_model'],
            'kwargs': {}}
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f'writing metadata to {path}')
        with open(path, 'w') as f:
            json.dump(content, f, indent=4)

    def _package(self):
        """Create a compressed file with the model and metadata used by the
        :class:`~zensols.install.installer.Installer` using resource library
        ``amr_parser:installer``.

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

    def train(self, dry_run: bool = False):
        """Train the model (see class docs).

        :param dry_run: when ``True``, don't do anything, just act like it.

        """
        self.write_to_log(logger)
        dir_path: Path
        for dir_path in (self.temporary_dir, self.output_dir):
            logger.debug(f'removing directory: {dir_path}')
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
        train: Callable = self._get_train_method()
        if not dry_run:
            train()
            self._compile_model()
            self._copy_model_files()
            self._package()


Trainer.pretrained_path_or_model = Trainer._pretrained_path_or_model
Trainer.training_config_file = Trainer._training_config_file


@dataclass
class XfmTrainer(Trainer):
    """Interface in to the :mod:`amrlib` package's HuggingFace T5 model trainer.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'corpus_file', 'token_model_name'} | Trainer._DICTABLE_ATTRIBUTES

    @property
    @persisted('_corpus_file')
    def corpus_file(self) -> Path:
        self.corpus_installer()
        return self.corpus_installer.get_singleton_path()

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

    def _populate_training_config(self, config: Dict[str, Any]):
        corpus_file: Path = self.corpus_file
        ga: Dict[str, str] = config['gen_args']
        hf: Dict[str, str] = config['hf_args']
        model_or_path: Union[str, Path] = self.pretrained_path_or_model
        if isinstance(model_or_path, Path):
            model_or_path = str(model_or_path.absolute())
        ga['corpus_dir'] = str(corpus_file.parent.absolute())
        ga['train_fn'] = corpus_file.name
        ga['tok_name_or_path'] = self.token_model_name
        ga['model_name_or_path'] = model_or_path
        ga['eval_fn'] = corpus_file.name
        hf['output_dir'] = str(self.temporary_dir)

    def _write_config(self, config: Dict[str, any]):
        meta: Dict[str, str] = self._get_input_metadata()
        base_model: str = meta.get('base_model', config.get('model'))
        cfile: Path = self.output_dir / f'model_parse_{self.model_name}.json'
        config = cp.deepcopy(config)
        config['gen_args']['corpus_dir'] = str(self.corpus_file.parent)
        config['gen_args']['model_name_or_path'] = base_model
        cfile.parent.mkdir(parents=True)
        with open(cfile, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f'wrote: {cfile}')

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
        base_model: str = meta['base_model']
        cp_dir: Path = self._get_checkpoint_dir() / 'config.json'
        with open(cp_dir) as f:
            content = json.load(f)
        content['_name_or_path'] = base_model
        pa = content['task_specific_params']['parse_amr']
        pa['corpus_dir'] = str(self.corpus_file.parent)
        pa['model_name_or_path'] = base_model
        new_config = self.output_dir / 'config.json'
        with open(new_config, 'w') as f:
            json.dump(content, f, indent=4)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {new_config}')

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
class SpringTrainer(Trainer):
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'train_files', 'dev_files'} | Trainer._DICTABLE_ATTRIBUTES

    _SMATCH_RE: ClassVar[re.Pattern] = re.compile(
        r'^checkpoint.+smatch_([0-9]+)\.pt$')
    train_files: str = field(default=None)
    dev_files: str = field(default=None)

    @persisted('_corpus_file')
    def _get_corpus_file(self) -> Path:
        self.corpus_installer()
        return self.corpus_installer.get_singleton_path()

    @property
    def _train_files(self) -> str:
        if self._train_files_val is None:
            return str(self._get_corpus_file().parent.absolute()) + '/*.txt'
        return self._train_files_val

    @_train_files.setter
    def _train_files(self, _train_files: str):
        self._train_files_val = _train_files

    @property
    def _dev_files(self) -> str:
        if self._dev_files_val is None:
            return str(self._get_corpus_file().parent.absolute()) + '/*.txt'
        return self._dev_files_val

    @_dev_files.setter
    def _dev_files(self, _dev_files: str):
        self._dev_files_val = _dev_files

    def _populate_training_config(self, config: Dict[str, Any]):
        corpus_file: Path = self.corpus_installer.get_singleton_path()
        train_files: str = self.train_files
        dev_files: str = self.dev_files
        if train_files is None:
            train_files = str(corpus_file.parent.absolute()) + '/*.txt'
        if dev_files is None:
            dev_files = str(corpus_file.parent.absolute()) + '/*.txt'
        config['train'] = train_files
        config['dev'] = dev_files
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
            # amrlib expects this
            config['smatch_dev'] = -1
            config['last_epoch'] = -1
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
        output_dir: Path = Path(config['model_dir'])
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'output dir: {output_dir}')
        conf_file = output_dir / 'config.json'
        output_dir.mkdir(parents=True, exist_ok=True)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'create {self.training_config_file} -> {conf_file}')
            assert not conf_file.exists()
            with open(conf_file, 'w') as f:
                json.dump(config, f, indent=4)
        train_fn: Callable = trainer.train
        train = (lambda: train_fn(checkpoint=cp))
        return train


SpringTrainer.train_files = SpringTrainer._train_files
SpringTrainer.dev_files = SpringTrainer._dev_files
SpringTrainer.training_config_file = SpringTrainer._training_config_file
