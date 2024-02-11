"""AMR parsing spaCy pipeline component and sentence generator.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Type, Union, Callable
from dataclasses import dataclass, field
import logging
import os
import warnings
import json
import textwrap as tw
from pathlib import Path
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
import amrlib
from amrlib.models.inference_bases import GTOSInferenceBase, STOGInferenceBase
from zensols.util import loglevel
from zensols.persist import persisted
from zensols.install import Installer
from zensols.nlp import (
    FeatureToken, TokenContainer, FeatureDocument, FeatureSentence,
    FeatureDocumentParser, Component, ComponentInitializer,
    SpacyFeatureDocumentParser
)
from . import (
    AmrError, AmrFailure, AmrSentence, AmrDocument,
    AmrFeatureSentence, AmrFeatureDocument,
    AmrGeneratedSentence, AmrGeneratedDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelContainer(object):
    """Contains an installer used to download and install a model that's then
    used by the API to parse AMR graphs or generate language from AMR graphs.

    """
    name: str = field()
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
class AmrParser(ModelContainer, ComponentInitializer):
    """Parses natural language into AMR graphs.  It has the ability to change
    out different installed models in the same Python session.

    """
    add_missing_metadata: bool = field(default=True)
    """Whether to add missing metadata to sentences when missing.

    :see: :meth:`add_metadata`

    """
    model: str = field(default='noop')
    """The :mod:`penman` AMR model to use when creating :class:`.AmrSentence`
    instances, which is one of ``noop`` or ``amr``.  The first does not modify
    the graph but the latter normalizes out inverse relationships such as
    ``ARG*-of``.

    """
    def init_nlp_model(self, model: Language, component: Component):
        """Reset the installer to all reloads in a Python REPL with different
        installers.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'initializing ({id(model)}): {self.name}')
        doc_parser: FeatureDocumentParser = model.doc_parser
        new_parser: AmrParser = doc_parser.config_factory(self.name)
        self.installer = new_parser.installer

    # if the model doesn't change after its app configuration does for the life
    # of the interpreter, turn off caching in config amr_anon_feature_doc_stash
    @persisted('_parse_model', cache_global=True)
    def _get_parse_model(self) -> STOGInferenceBase:
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

    @property
    def parse_model(self) -> STOGInferenceBase:
        model, prev_path = self._get_parse_model()
        cur_path = self.model_path
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'current path: {cur_path}, prev path: {prev_path}')
        if cur_path != prev_path:
            self._clear_model()
            model = self._get_parse_model()[0]
            amrlib.stog_model = model
        return model

    @staticmethod
    def is_missing_metadata(amr_sent: AmrSentence) -> bool:
        """Return whether ``amr_sent`` is missing annotated metadata.  T5 model
        sentences only have the ``snt`` metadata entry.

        :param amr_sent: the sentence to populate

        :see: :meth:`add_metadata`

        """
        return 'tokens' not in amr_sent.metadata

    @classmethod
    def add_metadata(cls: Type, amr_sent: AmrSentence,
                     sent: Union[TokenContainer, Span]):
        """Add missing annotation metadata parsed from spaCy if missing, which
        happens in the case of using the T5 AMR model.

        :param amr_sent: the sentence to populate

        :param sent: the spacCy sentence used as the source

        :see: :meth:`is_missing_metadata`

        """
        if isinstance(sent, TokenContainer):
            cls._add_metadata_cont(amr_sent, sent)
        else:
            cls._add_metadata_spacy(amr_sent, sent)

    @staticmethod
    def _add_metadata_cont(amr_sent: AmrSentence, sent: TokenContainer):
        def map_ent(t: Token) -> str:
            et = t.ent_
            return 'O' if et == FeatureToken.NONE else et

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'add metadata from token container: {amr_sent}')
        toks: Tuple[str] = tuple(sent.norm_token_iter())
        amr_sent.set_metadata('tokens', json.dumps(toks))
        if hasattr(sent[0], 'lemma_'):
            lms: Tuple[str] = tuple(map(lambda t: t.lemma_, sent.tokens))
            amr_sent.set_metadata('lemmas', json.dumps(lms))
        if hasattr(sent[0], 'ent_'):
            ents: Tuple[str] = tuple(map(map_ent, sent))
            amr_sent.set_metadata('ner_tags', json.dumps(ents))
        if hasattr(sent[0], 'tag_'):
            pt: Tuple[str] = tuple(map(lambda t: t.tag_, sent))
            amr_sent.set_metadata('pos_tags', json.dumps(pt))
        if hasattr(sent[0], 'ent_iob_'):
            iob: Tuple[str] = tuple(map(lambda t: t.ent_iob_, sent))
            amr_sent.set_metadata('ner_iob', json.dumps(iob))

    @staticmethod
    def _add_metadata_spacy(amr_sent: AmrSentence, sent: Span):
        def map_ent(t: Token) -> str:
            et = t.ent_type_
            return 'O' if len(et) == 0 else et

        def map_ent_iob(t: Token) -> str:
            et = t.ent_iob_
            return 'O' if len(et) == 0 else et

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'add metadata from spacy span: {amr_sent}')
        toks = tuple(map(lambda t: t.orth_, sent))
        lms = tuple(map(lambda t: t.lemma_, sent))
        amr_sent.set_metadata('tokens', json.dumps(toks))
        amr_sent.set_metadata('lemmas', json.dumps(lms))
        if hasattr(sent[0], 'ent_type_'):
            ents = tuple(map(map_ent, sent))
            amr_sent.set_metadata('ner_tags', json.dumps(ents))
        if hasattr(sent[0], 'tag_'):
            pt = tuple(map(lambda t: t.tag_, sent))
            amr_sent.set_metadata('pos_tags', json.dumps(pt))
        if hasattr(sent[0], 'ent_iob_'):
            iob: Tuple[str] = tuple(map(map_ent_iob, sent))
            amr_sent.set_metadata('ner_iob', json.dumps(iob))

    def __call__(self, doc: Doc) -> Doc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'parsing from {doc}')
        # force load the model in to the global amrlib module space
        stog_model: STOGInferenceBase = self.parse_model
        # add spacy underscore data holders for the amr data structures
        if not Doc.has_extension('amr'):
            Doc.set_extension('amr', default=[])
        if not Span.has_extension('amr'):
            Span.set_extension('amr', default=[])
        sent_graphs: List[AmrSentence] = []
        sent: Span
        for i, sent in enumerate(doc.sents):
            err: AmrFailure = None
            graphs: List[str] = None
            try:
                graphs = stog_model.parse_spans([sent])
                graph: str = graphs[0]
                if graph is None:
                    err = AmrFailure("Could not parse: empty graph " +
                                     f"(total={len(graphs)})", sent.text)
                if logger.isEnabledFor(logging.INFO):
                    graph_str = tw.shorten(str(graph), width=60)
                    logger.info(f'adding graph for sent {i}: <{graph_str}>')
            except Exception as e:
                err = AmrFailure(e, sent=sent.text)
            if err is not None:
                sent._.amr = AmrSentence(err)
                sent_graphs.append(sent._.amr)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'creating sentence with model: {self.model}')
                amr_sent = AmrSentence(graph, model=self.model)
                if self.add_missing_metadata and \
                   self.is_missing_metadata(amr_sent):
                    self.add_metadata(amr_sent, sent)
                sent._.amr = amr_sent
                sent_graphs.append(amr_sent)
        doc._.amr = AmrDocument(sent_graphs)
        return doc


@dataclass
class AmrGenerator(ModelContainer):
    """A callable that generates natural language text from an AMR graph.

    :see: :meth:`__call__`

    """
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

    @property
    @persisted('_generation_model', cache_global=True)
    def generation_model(self) -> GTOSInferenceBase:
        """The model that generates sentences from an AMR graph."""
        logger.debug('loading generation model')
        self._load_model()
        return amrlib.load_gtos_model()

    def __call__(self, doc: AmrDocument) -> Tuple[str]:
        """Generate a sentence from a spaCy document.

        :param doc: the spaCy document used to generate the sentence

        :return: a text sentence for each respective sentence in ``doc``

        """
        model: GTOSInferenceBase = self.generation_model
        generate_fn: Callable = model.generate
        # upgrade to amrlib 0.8.0
        if 0:
            from amrlib.models.generate_t5wtense.inference \
                import Inference as T5TenseInference
            if isinstance(model, T5TenseInference):
                org_fn: Callable = generate_fn
                generate_fn = (lambda s: org_fn(s, use_tense=self.use_tense))
        preds: Tuple[List[str], List[bool]] = generate_fn(list(map(
            lambda s: s.graph_string, doc)))
        sents: List[AmrGeneratedSentence] = []
        sent: AmrSentence
        for sent, (text, clipped) in zip(doc, zip(*preds)):
            sents.append(AmrGeneratedSentence(text, clipped, sent))
        return AmrGeneratedDocument(sents=sents, amr=doc)


@dataclass
class EntityCopySpacyFeatureDocumentParser(SpacyFeatureDocumentParser):
    """Copy spaCy ``ent_type_`` named entity (NER) tags to
    :class:`~zensols.nlp.container.FeatureToken` ``ent_`` tags.

    The AMR document's metadata ``ner_tags`` is populated in :class:`.AmrParser`
    from the spaCy document.  But this document parser instance is configured
    with embedded entities turned off so whitespace delimited tokens match with
    the alignments.

    """
    def _decorate_doc(self, spacy_doc: Span, feature_doc: FeatureDocument):
        ix2tok: Dict[int, Token] = {t.i: t for t in spacy_doc}
        ftok: FeatureToken
        for ftok in feature_doc.token_iter():
            stok: Token = ix2tok.get(ftok.i)
            if stok is not None and len(stok.ent_type_) > 0:
                ftok.ent_ = stok.ent_type_


@dataclass
class AmrFeatureDocumentFactory(object):
    """Creates :class:`.AmrFeatureDocument` from :class:`.AmrDocument`
    instances.

    """
    doc_parser: FeatureDocumentParser = field()
    """The document parser used to creates :class:`.AmrFeatureDocument`
    instances.

    """
    def to_feature_doc(self, amr_doc: AmrDocument, catch: bool = False,
                       add_metadata: bool = False) -> \
            Union[AmrFeatureDocument,
                  Tuple[AmrFeatureDocument, List[AmrFailure]]]:
        """Create a :class:`.AmrFeatureDocument` from a class:`.AmrDocument` by
        parsing the ``snt`` metadata with a
        :class:`~zensols.nlp.parser.FeatureDocumentParser`.

        :param add_metadata: add missing annotation metadata parsed from spaCy
                             if missing (see :meth:`.AmrParser.add_metadata`)

        :param catch: if ``True``, return caught exceptions creating a
                      :class:`.AmrFailure` from each and return them

        :return: an AMR feature document if ``catch`` is ``False``; otherwise, a
                 tuple of a document with sentences that were successfully
                 parsed and a list any exceptions raised during the parsing

        """
        sents: List[AmrFeatureSentence] = []
        fails: List[AmrFailure] = []
        amr_doc_text: str = None
        amr_sent: AmrSentence
        for amr_sent in amr_doc.sents:
            sent_text: str = None
            ex: Exception = None
            try:
                # force white space tokenization to match the already tokenized
                # metadata ('tokens' key); examples include numbers followed by
                # commas such as dates like "April 25 , 2008"
                sent_text = amr_sent.tokenized_text
                sent_doc: FeatureDocument = self.doc_parser(sent_text)
                sent: FeatureSentence = sent_doc.to_sentence(
                    contiguous_i_sent=True)
                sent = sent.clone(cls=AmrFeatureSentence, amr=None)
                if add_metadata:
                    AmrParser.add_metadata(amr_sent, sent_doc.spacy_doc)
                sents.append(sent)
            except Exception as e:
                fails.append(AmrFailure(e, sent=sent_text))
                ex = e
            if ex is not None and not catch:
                raise ex
        try:
            amr_doc_text = amr_doc.text
        except Exception as e:
            if not catch:
                raise e
            else:
                amr_doc_text = f'erorr: {e}'
                logger.error(f'could not parse AMR document text: {e}', e)
        doc = AmrFeatureDocument(
            sents=tuple(sents),
            text=amr_doc_text,
            amr=amr_doc)
        if catch:
            return doc, tuple(fails)
        else:
            return doc


@Language.factory('amr_parser')
def create_amr_parser(nlp: Language, name: str, parser_name: str) -> AmrParser:
    """Create an instance of :class:`.AmrParser`.

    """
    doc_parser: FeatureDocumentParser = nlp.doc_parser
    if logger.isEnabledFor(logging.INFO):
        dp_str: str = str(type(doc_parser))
        if hasattr(doc_parser, 'name'):
            dp_str += f' {doc_parser.name} ({dp_str})'
        logger.info(f"creating AMR component '{name}': doc parser: '{dp_str}'")
    parser: AmrParser = doc_parser.config_factory(parser_name)
    if not isinstance(parser, AmrParser):
        raise AmrError(
            f"Expecting type '{AmrParser}' but got:  '{type(parser)}'")
    return parser
