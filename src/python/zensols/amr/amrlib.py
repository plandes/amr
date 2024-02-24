"""AMR parser and generator model implementations using :mod:`amrlib`.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Iterable, Callable
from dataclasses import dataclass, field
import logging
import os
import warnings
import textwrap as tw
from pathlib import Path
from spacy.language import Language
from spacy.tokens import Span
import amrlib
from amrlib.models.inference_bases import GTOSInferenceBase, STOGInferenceBase
from zensols.util import loglevel
from zensols.persist import persisted
from zensols.install import Installer
from zensols.nlp import FeatureDocumentParser, Component
from . import (
    AmrFailure, AmrSentence, AmrDocument,
    AmrGeneratedSentence, AmrGeneratedDocument,
)
from .model import AmrParser, AmrGenerator

logger = logging.getLogger(__name__)


@dataclass
class _AmrlibModelContainer(object):
    """Contains an installer used to download and install a model that's then
    used by the API to parse AMR graphs or generate language from AMR graphs.

    """
    name: str = field(default=None)
    """The section name."""

    installer: Installer = field(default=None)
    """"Use to install the model files.  The installer must have one and only
    resource.

    """
    alternate_path: Path = field(default=None)
    """If set, use this alternate path to find the model files."""

    def __post_init__(self):
        # minimize warnings (T5)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        warnings.filterwarnings(
            'ignore',
            message=r'^This tokenizer was incorrectly instantiated with',
            category=FutureWarning)
        from transformers import logging
        logging.set_verbosity_error()
        # save the section name since AmrParser has its ``name`` attribute
        # replaced by the spaCy API
        self._init_name = self.name

    @property
    def model_path(self) -> Path:
        if self.alternate_path is None:
            pkg_path = self.installer.get_singleton_path().parent
        else:
            pkg_path = self.alternate_path
        return pkg_path

    def _load_model(self) -> Path:
        self.installer.install()
        model_path: Path = self.model_path
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'resolved model path: {model_path}')
        if amrlib.defaults.data_dir != model_path:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting AMR model directory to {model_path}')
            amrlib.defaults.data_dir = model_path
        return model_path


@dataclass
class AmrlibParser(_AmrlibModelContainer, AmrParser):
    def init_nlp_model(self, model: Language, component: Component):
        """Reset the installer to all reloads in a Python REPL with different
        installers.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'initializing ({id(model)}): {self.name}')
        doc_parser: FeatureDocumentParser = model.doc_parser
        new_parser: AmrParser = doc_parser.config_factory(self._init_name)
        self.installer = new_parser.installer

    # if the model doesn't change after its app configuration does for the life
    # of the interpreter, turn off caching in config amr_anon_feature_doc_stash
    @persisted('_parse_model', cache_global=True)
    def _create_parse_model(self) -> STOGInferenceBase:
        """The model that parses text in to AMR graphs.  This model is cached
        globally, as it is cached in the :mod:`amrlib` module as well.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('loading parse model')
        model_path = self._load_model()
        if model_path.name.find('gsii') > -1:
            with loglevel('transformers', logging.ERROR):
                model = amrlib.load_stog_model()
        else:
            model = amrlib.load_stog_model()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'using parse model: {model.__class__}')
        return model, model_path

    def _clear_model(self):
        self._parse_model.clear()
        amrlib.stog_model = None

    def _get_parse_model(self) -> STOGInferenceBase:
        model, prev_path = self._create_parse_model()
        cur_path = self.model_path
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'current path: {cur_path}, prev path: {prev_path}')
        if cur_path != prev_path:
            self._clear_model()
            model = self._create_parse_model()[0]
            amrlib.stog_model = model
        return model

    def _parse_sents(self, sents: Iterable[Span]) -> Iterable[AmrSentence]:
        # force load the model in to the global amrlib module space
        stog_model: STOGInferenceBase = self._get_parse_model()
        sent: Span
        for six, sent in enumerate(sents):
            graph: str = None
            err: AmrFailure = None
            try:
                graphs: List[str] = stog_model.parse_spans([sent])
                graph: str = graphs[0]
                err: AmrFailure = None
                if graph is None:
                    err = AmrFailure("Could not parse: empty graph " +
                                     f"(total={len(graphs)})", sent.text)
                if logger.isEnabledFor(logging.INFO):
                    graph_str = tw.shorten(str(graph), width=60)
                    logger.info(f'adding graph for sent {six}: <{graph_str}>')
            except Exception as e:
                err = AmrFailure(e, sent=sent.text)
            if err is None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'creating sentence with model: {self.model}')
                yield AmrSentence(graph, model=self.model)
            else:
                yield AmrSentence(err)


@dataclass
class AmrlibGenerator(_AmrlibModelContainer, AmrGenerator):
    use_tense: bool = field(default=True)
    """Try to add tense information by trying to tag the graph, which requires
    the sentence or annotations and then goes through an alignment.

    :see: :class:`amrlib.models.generate_t5wtense.inference.Inference`

    """
    def __post_init__(self):
        super().__post_init__()
        warnings.filterwarnings(
            'ignore',
            message=r'^`num_beams` is set to 1. However, `early_stopping` is ',
            category=UserWarning)

    @persisted('_generation_model', cache_global=True)
    def _get_generation_model(self) -> GTOSInferenceBase:
        """The model that generates sentences from an AMR graph."""
        logger.debug('loading generation model')
        self._load_model()
        return amrlib.load_gtos_model()

    def generate(self, doc: AmrDocument) -> AmrGeneratedDocument:
        """Generate a sentence from a spaCy document.

        :param doc: the spaCy document used to generate the sentence

        :return: a text sentence for each respective sentence in ``doc``

        """
        model: GTOSInferenceBase = self._get_generation_model()
        generate_fn: Callable = model.generate
        preds: Tuple[List[str], List[bool]] = generate_fn(list(map(
            lambda s: s.graph_string, doc)))
        sents: List[AmrGeneratedSentence] = []
        sent: AmrSentence
        for sent, (text, clipped) in zip(doc, zip(*preds)):
            sents.append(AmrGeneratedSentence(text, clipped, sent))
        return AmrGeneratedDocument(sents=sents, amr=doc)
