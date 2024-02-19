"""AMR parsing spaCy pipeline component and sentence generator.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Type
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
import logging
import json
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from zensols.nlp import FeatureDocumentParser, Component, ComponentInitializer
from . import AmrError, AmrSentence, AmrDocument

logger = logging.getLogger(__name__)


@dataclass
class AmrParser(ComponentInitializer, metaclass=ABCMeta):
    """Parses natural language into AMR graphs.  It has the ability to change
    out different installed models in the same Python session.

    """
    def init_nlp_model(self, model: Language, component: Component):
        """Initialize the parser with spaCy API components."""
        pass

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
                     sent: Span, clobber: bool = False):
        """Add missing annotation metadata parsed from spaCy if missing, which
        happens in the case of using the T5 AMR model.

        :param amr_sent: the sentence to populate

        :param sent: the spacCy sentence used as the source

        :param clobber: whether or not to overwrite any existing metadata fields

        :see: :meth:`is_missing_metadata`

        """
        def map_ent(t: Token) -> str:
            et = t.ent_type_
            return 'O' if len(et) == 0 else et

        def map_ent_iob(t: Token) -> str:
            et = t.ent_iob_
            return 'O' if len(et) == 0 else et

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'add metadata from spacy span: {amr_sent}')

        meta: Dict[str, str] = amr_sent.metadata
        tok: Token = sent[0]
        if clobber or 'tokens' not in meta:
            toks = tuple(map(lambda t: t.orth_, sent))
            amr_sent.set_metadata('tokens', json.dumps(toks))
        if clobber or 'lemmas' not in meta:
            lms = tuple(map(lambda t: t.lemma_, sent))
            amr_sent.set_metadata('lemmas', json.dumps(lms))
        if (clobber or 'ner_tags' not in meta) and hasattr(tok, 'ent_type_'):
            ents = tuple(map(map_ent, sent))
            amr_sent.set_metadata('ner_tags', json.dumps(ents))
        if (clobber or 'pos_tags' not in meta) and hasattr(tok, 'tag_'):
            pt = tuple(map(lambda t: t.tag_, sent))
            amr_sent.set_metadata('pos_tags', json.dumps(pt))
        if (clobber or 'ner_iob' not in meta) and hasattr(tok, 'ent_iob_'):
            iob: Tuple[str] = tuple(map(map_ent_iob, sent))
            amr_sent.set_metadata('ner_iob', json.dumps(iob))
        amr_sent.meta = meta

    @abstractmethod
    def annotate_amr(self, doc: Doc):
        """Add an ``amr`` attribute to the spaCy document.

        :param doc: the document to annotate

        """
        pass

    def __call__(self, doc: Doc) -> Doc:
        """See :meth:`annotate_amr`.

        :param doc: the document to annotate

        :return: ``doc``

        """
        self.annotate_amr(doc)
        return doc


@dataclass
class AmrGenerator(object):
    """A callable that generates natural language text from an AMR graph.

    :see: :meth:`__call__`

    """
    @abstractmethod
    def generate(self, doc: AmrDocument) -> Tuple[str]:
        """Generate a sentence from the AMR graph ``doc``.

        :param doc: the spaCy document used to generate the sentence

        :return: a text sentence for each respective sentence in ``doc``

        """
        pass

    def __call__(self, doc: AmrDocument) -> Tuple[str]:
        """See :meth:`generate`."""
        return self.generate(doc)


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
