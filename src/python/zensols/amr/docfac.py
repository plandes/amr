"""Feature sentence and document utilities.

"""
__author__ = 'Paul Landes'

from typing import Tuple, List, Dict, Union
from dataclasses import dataclass, field
import logging
from spacy.tokens import Span, Token
from zensols.nlp import (
    FeatureToken, FeatureSentence, FeatureDocument,
    FeatureDocumentParser, SpacyFeatureDocumentParser,
)
from . import (
    AmrFailure, AmrSentence, AmrDocument,
    AmrFeatureSentence, AmrFeatureDocument,
)
from .model import AmrParser

logger = logging.getLogger(__name__)


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
