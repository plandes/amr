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
    """Interface in to the :mod:`amrlib` package's HuggingFace T5 model trainer.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'corpus_file', 'pretrained_path_or_model',
        'trainer_class', 'training_config'}

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
    @persisted('_corpus_file')
    def corpus_file(self) -> Path:
        self.corpus_installer()
        return self.corpus_installer.get_singleton_path()

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

    @persisted('_training_config_file')
    def _get_training_config_file(self) -> Path:
        path: Path = self.training_config_file
        if path is None:
            paths: Tuple[Path, ...] = tuple(self.pretrained_path_or_model.iterdir())
            cans: Tuple[Path, ...] = tuple(filter(
                lambda p: p.name.startswith('model') and p.suffix == '.json',
                paths))
            if len(cans) != 1:
                logger.warning(
                    "expecting a single file starts with 'model' " +
                    f"but got {', '.join(map(lambda p: p.name, paths))}")
            else:
                path = cans[0]
        if path is None:
            logger.warning(
                f'missing training config file: {self.training_config_file}')
        return path

    def _massage_training_config(self, config: Dict[str, Any]):
        overrides: Dict[str, Any] = self.training_config_overrides
        config.update(overrides)

    @persisted('_training_config_content')
    def _get_training_config_content(self) -> Dict[str, Any]:
        train_file: Path = self._get_training_config_file()
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

    def _get_checkpoint_dir(self) -> Path:
        paths: Tuple[Path, ...] = tuple(self.temporary_dir.iterdir())
        cps = tuple(filter(lambda p: p.name.startswith('checkpoint'), paths))
        if len(cps) != 1:
            raise AmrError(
                f'Expecting 1 path at {self.temporary_dir} but got: {paths}')
        return cps[0]

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

    def _copy_model_files(self):
        fname: str = 'pytorch_model.bin'
        cp_dir: Path = self._get_checkpoint_dir()
        src: Path = cp_dir / fname
        dst: Path = self.output_dir / fname
        shutil.copy(src, dst)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'copied model weights and state: {dst}')

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
        for dir_path in (self.temporary_dir, self.output_dir,):
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
        train = self._get_train_method()
        if not dry_run:
            train()
            self._compile_model()
            self._copy_model_files()
            self._package()


Trainer.pretrained_path_or_model = Trainer._pretrained_path_or_model


@dataclass
class XfmTrainer(Trainer):
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
    def _populate_training_config(self, config: Dict[str, Any]):
        corpus_file: Path = self.corpus_file
        pt_path: Path = self.pretrained_path_or_model
        # pt_path: Path = self.pretrained_path_or_model
        # if pt_path is None:
        config['model_dir'] = str(self.temporary_dir.absolute())
        # else:
        #     conf['model_dir'] = str(self.pretrained_path_or_model.absolute())
        config['train'] = str(corpus_file.parent.absolute()) + '/*.txt'
        config['dev'] = config['train']

    def _get_train_method(self) -> Callable:
        config: Dict[str, Any] = self.training_config
        trainer: Callable = self.trainer_class.trainer_class(config)
        pt_path: Path = self.pretrained_path_or_model
        cp: str = None
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'setup spring config, pretrain path: {pt_path}')
        if pt_path is None:
            logger.info('training from scratch')
            # amrlib expects this
            config['smatch_dev'] = -1
            config['last_epoch'] = -1
        else:
            cp = str(self.pretrained_path_or_model.absolute() / 'model.pt')
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'using check point: {cp}')
        output_dir: Path = Path(config['model_dir'])
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'output dir: {output_dir}')
        # train() removed this
        assert not output_dir.is_dir()
        conf_file = output_dir / 'config.json'
        output_dir.mkdir(parents=True)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'copy {self.training_config_file} to {conf_file}')
            with open(conf_file, 'w') as f:
                json.dump(config, f, indent=4)
        train_fn: Callable = trainer.train
        train = (lambda: train_fn(checkpoint=cp))
        return train
