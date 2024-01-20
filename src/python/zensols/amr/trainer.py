"""Continues training on an AMR model.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Tuple, Set, Any, Type, ClassVar, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
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


class _ModelType(Enum):
    """The type of model to train, such as XFM T5, spring etc.

    """
    _ignore_ = ['_MOD_RE']
    _MOD_RE: ClassVar[re.Pattern]

    # supported models
    xfm = auto()
    spring = auto()

    @classmethod
    def from_meta(cls: Type[_ModelType], meta: Dict[str, Any]) -> _ModelType:
        mod: str = meta['inference_module']
        m: re.Pattern = cls._MOD_RE.match(mod)
        model_type_str: str = m.group(1)
        model_type: _ModelType = cls.__members__.get(model_type_str)
        if model_type is None:
            raise AmrError(f'No such (trainable) model: {model_type_str}')
        return model_type


_ModelType._MOD_RE = re.compile(r'^\.parse_([^.]+)\.inference$')


@dataclass
class Trainer(Dictable):
    """Interface in to the :mod:`amrlib` package's HuggingFace T5 model trainer.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {
        'corpus_file', 'pretrained_path', 'trainer_class', 'training_config',
        'token_model_name'}

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
    training_config_overrides: Dict[str, Any] = field(default=None)
    """More configuration that overrides/clobbers from the contents found in
    :obj:`training_config_file`.

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
    @persisted('_pretrained_path')
    def pretrained_path(self) -> Path:
        """The path to the pretrained ``pytorch_model.bin`` file."""
        return self.parser.installer.get_singleton_path()

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
            paths: Tuple[Path, ...] = tuple(self.pretrained_path.iterdir())
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

    @persisted('_training_config_content')
    def _get_training_config_content(self) -> Dict[str, Any]:
        train_file: Path = self._get_training_config_file()
        if train_file is not None:
            overrides: Dict[str, Any] = self.training_config_overrides
            with open(train_file) as f:
                conf = json.load(f)
            if self._get_model_type() == _ModelType.xfm:
                if overrides is not None:
                    for k in 'gen_args hf_args model_args'.split():
                        if k in self.training_config_overrides:
                            conf[k] = conf[k] | overrides[k]
                # by 4.35 HF defaults to safe tensors, but amrlib models were
                # trained before, this
                conf['hf_args']['save_safetensors'] = False
            else:
                conf.update(overrides)
            return conf

    @property
    @persisted('_token_model_name')
    def token_model_name(self) -> str:
        """The name of the tokenziation model as the pretrained AMR model files
        do not have these files.

        """
        train_conf: Dict[str, Any] = self._get_training_config_content()
        return
        if train_conf is not None:
            ga: Dict[str, str] = train_conf['gen_args']
            return ga['model_name_or_path']

    @persisted('_model_type')
    def _get_model_type(self) -> _ModelType:
        meta: Dict[str, str] = self._get_input_metadata()
        return _ModelType.from_meta(meta)

    @property
    @persisted('_training_config')
    def training_config(self) -> Dict[str, Any]:
        """The parameters given to the instance of the trainer, which is the
        class derived with :obj:`trainer_class`.

        """
        corpus_file: Path = self.corpus_file
        conf: Dict[str, Any] = self._get_training_config_content()
        if conf is not None:
            if self._get_model_type() == _ModelType.spring:
                conf['model_dir'] = str(self.pretrained_path.absolute())
                conf['train'] = str(corpus_file.parent.absolute()) + '/*.txt'
                conf['dev'] = conf['train']
            else:
                ga: Dict[str, str] = conf['gen_args']
                hf: Dict[str, str] = conf['hf_args']
                ga['corpus_dir'] = str(corpus_file.parent.absolute())
                ga['train_fn'] = corpus_file.name
                ga['tok_name_or_path'] = self.token_model_name
                ga['model_name_or_path'] = str(self.pretrained_path.absolute())
                ga['eval_fn'] = corpus_file.name
                hf['output_dir'] = str(self.temporary_dir)
            return conf

    def _write_config(self, config: Dict[str, any]):
        meta: Dict[str, str] = self._get_input_metadata()
        base_model: str = meta['base_model']
        cfile: Path = self.output_dir / f'model_parse_{self.model_name}.json'
        config = cp.deepcopy(config)
        config['gen_args']['corpus_dir'] = str(self.corpus_file.parent)
        config['gen_args']['model_name_or_path'] = base_model
        with open(cfile, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f'wrote: {cfile}')

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

    def _get_checkpoint_dir(self) -> Path:
        paths: Tuple[Path, ...] = tuple(self.temporary_dir.iterdir())
        cps = tuple(filter(lambda p: p.name.startswith('checkpoint'), paths))
        if len(cps) != 1:
            raise AmrError(
                f'Expecting 1 path at {self.temporary_dir} but got: {paths}')
        return cps[0]

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

    def _copy_model_files(self):
        fname: str = 'pytorch_model.bin'
        cp_dir: Path = self._get_checkpoint_dir()
        src: Path = cp_dir / fname
        dst: Path = self.output_dir / fname
        shutil.copy(src, dst)

    def _package(self):
        """Create a compressed file with the model and metadata used by the
        :class:`~zensols.install.installer.Installer` using resource library
        ``amr_parser:installer``.

        """
        out_tar_file: Path = self.package_dir / f'{self.output_dir.stem}.tar.gz'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'compressing model: {self.output_dir}')
        with tf.open(out_tar_file, "w:gz") as tar:
            tar.add(self.output_dir, arcname=self.output_dir.name)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {out_tar_file}')

    def __call__(self, dry_run: bool = False):
        """Train the model (see class docs).

        :param dry_run: when ``True``, don't do anything, just act like it.

        """
        self.write_to_log(logger)
        trainer_class: Type = self.trainer_class
        config: Dict[str, Any] = self.training_config
        train: Callable = trainer_class(config)
        if self._get_model_type() == _ModelType.spring:
            cp: str = str(self.pretrained_path.absolute() / 'model.pt')
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'using check point: {cp}')
            train_fn: Callable = train.train
            train = (lambda: train_fn(checkpoint=cp))
        elif not isinstance(train, Callable):
            train = train.train
        dir_path: Path
        for dir_path in (self.temporary_dir, self.output_dir):
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
        self.output_dir.mkdir(parents=True)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'training on {self.corpus_file} to {self.output_dir}')
        if not dry_run:
            train()
            self._write_config(config)
            self._write_metadata()
            self._rewrite_config()
            self._copy_model_files()
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'model written to {self.output_dir}')
            self._package()
