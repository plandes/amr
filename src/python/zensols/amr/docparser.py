"""AMR document annotation.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Type, Iterable, Set, Any, Dict
from dataclasses import dataclass, field
import logging
import re
import json
import collections
from itertools import chain
import textwrap as tw
from spacy.tokens import Doc, Token, Span
from penman import constant, surface
from penman.graph import Graph, Triple, Attribute
from penman.surface import Alignment
from zensols.persist import persisted
from zensols.nlp import (
    SplitTokenMapper, TokenNormalizer, MapTokenNormalizer,
    FeatureSentence, FeatureDocument, FeatureToken,
    CachingFeatureDocumentParser, FeatureDocumentDecorator,
)
from . import (
    AmrError, Feature, AmrSentence, AmrDocument,
    AmrFeatureSentence, AmrFeatureDocument
)
from .coref import CoreferenceResolver
from .align import AmrAlignmentPopulator
from .model import AmrParser
from .spacyadapt import SpacyDocAdapter

logger = logging.getLogger(__name__)


class AmrParseError(AmrError):
    pass


@dataclass
class TokenAnnotationFeatureDocumentDecorator(FeatureDocumentDecorator):
    """Annotate features in AMR sentence graphs from indexes annotated from
    :class:`.AmrAlignmentPopulator`.

    """
    name: str = field()
    """The triple role (if :obj:`add_to_epi` is ``False``) used to label the
    edge between the token and the feature.  Otherwise, this string is used in
    the epidata of the graph.

    """
    feature_id: str = field()
    """The :class:`~zensols.nlp.FeatureToken` ID (attribute) to annotate in the
    AMR graph.

    """
    indexed: bool = field(default=False)
    """Whether or not to append an index to the role."""

    add_none: bool = field(default=False)
    """Whether add missing or empty values.  This includes string values of
    :obj:`zensols.nlp.FeatureToken.NONE`.

    """
    use_sent_index: bool = field(default=True)
    """Whether to map alignments to by (iterated) index position, or by using
    the per sentence index :obj:`~zensols.nlp.tok.FeatureToken` attribute
    ``i_sent``.  Set this to ``False`` if the the
    :class:`~zensols.nlp.parser.FeatureDocumentParser` was configured with a
    token normalizer configured with embedding named entities turned off.

    """
    method: str = field(default='attribute')
    """Where to add the data, which may be one of:

      * ``attribute``: add as a new attribute node using ``name`` as the role
        and the value as the attribute constant

      * ``epi``: as epigraph data; however, the current Penman implementation
        assume only alignments and the graph string will no longer be parsable

    Otherwise, it uses the string to format a replacement node text using
    ``target`` as the previous/original node text and ``value`` as the feature
    value text.

    """
    def __post_init__(self):
        if self.method == 'attribute':
            self._role = ':' + self.name

    def decorate(self, doc: FeatureDocument):
        if not isinstance(doc, AmrFeatureDocument):
            raise AmrParseError(
                f'Expecting AmrFeatureDocument but got: {type(doc)}')
        updates: List[AmrSentence] = []
        sent: AmrSentence
        for sent in doc.sents:
            if sent.is_failure:
                updates.append(sent.amr)
            else:
                updates.append(self._annotate_sentence(sent, doc))
        doc.amr.sents = updates
        doc.sync_amr_sents()

    def _format_feature_value(self, tok: FeatureToken) -> str:
        return getattr(tok, self.feature_id)

    def _is_none(self, feat_val: Any, tok: FeatureToken) -> bool:
        return feat_val is None or feat_val == FeatureToken.NONE

    def _annotate_token(self, tok: FeatureToken, source: Triple,
                        feature_triples: List[Attribute], graph: Graph):
        # create the triple from token data
        feat_val: Any = self._format_feature_value(tok)
        if self.add_none or not self._is_none(feat_val, tok):
            if self.method == 'attribute':
                val: str = constant.quote(feat_val)
                triple = Attribute(source[0], self._role, val)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'adding: {triple}')
                # add the feature as a triple to graph
                feature_triples.append(triple)
            elif self.method == 'epi':
                epis: Dict[Attribute, List] = graph.epidata
                epi_node: List[Any] = epis[source]
                epi_node.append(Feature(self.name, feat_val))
            else:
                epis: Dict[Attribute, List] = graph.epidata
                epi_node: List[Any] = epis[source]
                targ: str = source.target
                # keep literal (re)formatted text in the quotes
                quoted: bool = isinstance(targ, str) and targ.startswith('"')
                if quoted:
                    targ = constant.evaluate(targ)
                val: str = self.method.format(
                    target=targ,
                    value=feat_val)
                if quoted:
                    val = constant.quote(val)
                triple = Attribute(source.source, source.role, val)
                graph.triples.remove(source)
                graph.triples.append(triple)
                epis[triple] = epi_node

    def _align_idxs(self, graph: Graph, sent: AmrFeatureSentence,
                    tokens_by_i_sent: Dict[int, FeatureToken],
                    feat_trips: List[Attribute],
                    graph_tokens: List[str], source: Triple,
                    align: Alignment):
        use_sent_index: bool = self.use_sent_index
        tix: int
        for tix in align.indices:
            td: FeatureToken
            if use_sent_index:
                td = sent[tix]
            else:
                td = tokens_by_i_sent.get(tix)
            if td is not None:
                gtok: str = graph_tokens[tix]
                # alignment (AMR and feature normalization) sanity check
                if td.norm != gtok:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(
                            f'misalignment token index {tix}: <{td.norm}> != ' +
                            f'<{gtok}> in {sent} vs. {graph_tokens}')
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'token: {tix} -> {td}')
                self._annotate_token(td, source, feat_trips, graph)

    def _annotate_sentence(self, sent: AmrFeatureSentence,
                           doc: AmrFeatureDocument) -> AmrSentence:
        def map_attr(a: Attribute) -> Attribute:
            role: str = f'{a.role}{srcs[a.source]}'
            srcs[a.source] += 1
            return Attribute(a.source, role, a.target)

        if 'tokens' not in sent.amr.graph.metadata:
            raise AmrError(f'No tokens metadata in <{sent.amr.graph_string}>')
        tokens_by_i_sent: Dict[int, FeatureToken] = sent.tokens_by_i_sent
        graph: Graph = sent.amr.graph
        graph_tokens: List[str] = json.loads(graph.metadata['tokens'])
        aligns: Dict[Tuple, Alignment] = surface.alignments(graph)
        feat_trips: List[Attribute] = []
        trips: Triple = tuple(chain.from_iterable(
            (graph.instances(), graph.attributes())))
        # find triples that identify token index positions
        tix: int
        src_trip: Tuple[str, str, Any]
        for tix, src_trip in enumerate(trips):
            align: Alignment = aligns.get(src_trip)
            if align is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'triple: {src_trip}, align: {align}')
                self._align_idxs(graph, sent, tokens_by_i_sent, feat_trips,
                                 graph_tokens, src_trip, align)
        if self.indexed:
            srcs = collections.defaultdict(lambda: 1)
            feat_trips = tuple(map(map_attr, feat_trips))
        else:
            feat_trips = set(map(tuple, feat_trips))
        epis: Dict[Attribute, List] = graph.epidata
        for trip in feat_trips:
            if trip not in epis:
                epis[trip] = []
        graph.triples.extend(feat_trips)
        return AmrSentence(graph)


@dataclass
class AnnotationFeatureDocumentParser(CachingFeatureDocumentParser):
    """A document parser that adds and further annotates AMR graphs.  This has
    the advantage of avoiding a second AMR construction when annotating a graph
    with features (i.e. ent, POS tag, etc) because it uses (adapted) spaCy
    class's normalized features.  For this reason, use this class if your
    application needs such annotations.

    This parses and popluates AMR graphs as :class:`~zensols.amr.AmrDocument` at
    the document level and a :class:`~zensols.amr.AmrSentence` at the sentence
    level using a :class:`zensols.nlp.FeatureDocument`.

    This class will also recreate the AMR on normalized text of the document.
    This is necessary since AMR parsing and alignment happen at the spaCy level
    and token normalization happen at the :mod:`zensole.nlp` feature token
    level.  Since spaCy does not allow for filter tokens (i.e. stop words)
    there is no way to avoid a reparse.

    However, if your application makes no modification to the document, a
    second reparse is *not* needed and you should set :obj:`reparse` to False.

    A consideration is the adaptation spaCy module (:mod:`spacyadapt`) is not
    thoroughly tested and future updates might break.  If you do not feel
    comfortable using it, or can not, use the spacy pipline by setting
    ``amr_default:doc_parser = amr_anon_doc_parser`` in the application
    configuration and annotated the graph yourself.

    The AMR graphs are optionally cached using a
    :class:`~zensols.persist.domain.Stash` when :obj:`stash` is set.

    **Important:** when using stash caching only the :class:`.AmrDocument` is
    cached and not the entire feature document.  This could lead to the
    documents and AMR graphs getting out of sync if both are cached.  Use the
    :meth:`clear` method to clear the stash if ever in doubt.

    A new instance of :class:`~zensols.amr.AmrFeatureDocument` are returned.

    """
    amr_parser: AmrParser = field(default=None)
    """The AMR parser used to induce the graphs."""

    alignment_populator: AmrAlignmentPopulator = field(default=None)
    """Adds the alighment markings."""

    coref_resolver: CoreferenceResolver = field(default=None)
    """Adds coreferences between the sentences of the document."""

    reparse: bool = field(default=True)
    """Reparse the normalized :class:`~zensols.nlp.container.FeatureSentence`
    text for each AMR sentence, which is necessary when tokens are remove
    (i.e. stop words).  See the class docs.

    """
    amr_doc_class: Type[AmrFeatureDocument] = field(default=AmrFeatureDocument)
    """The :class:`~zensols.nlp.FeatureDocument` class created to store
    :class:`zensols.amr.AmrDocument` instances.

    """
    amr_sent_class: Type[AmrFeatureSentence] = field(default=AmrFeatureSentence)
    """The :class:`~zensols.nlp.FeatureSentence` class created to store
    :class:`zensols.amr.AmrSentence` instances.

    """
    def __post_init__(self):
        super().__post_init__()
        if self.amr_parser is None:
            raise AmrParseError(f"Missing 'amr_parser' field in {self}")

    @persisted('_token_normalize')
    def _get_token_normalizer(self) -> TokenNormalizer:
        mapper = SplitTokenMapper(regex=re.compile(r'\s+'))
        norm = MapTokenNormalizer(embed_entities=False)
        norm.mappers.append(mapper)
        return norm

    def _split_toks(self, fsent: FeatureSentence, ssent: Span):
        """Split tokens with multiple words (i.e. named entities) as the GSII
        parser doesn't handel them.  The sentence is recreated by iterating over
        the noramlized space split tokens.

        """
        norms: Iterable[Tuple[Token, str]] = \
            tuple(self._get_token_normalizer().normalize(ssent))
        if logger.isEnabledFor(logging.DEBUG):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'split on norms: \'{ssent.text}\' -> {norms}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sent len: {len(fsent)}, norms: len(norms)')
        used: Set[int] = set()
        toks: List[FeatureToken] = []
        for stok, norm in norms:
            ftok: FeatureToken = stok._ftok
            fid = id(ftok)
            if fid in used:
                ftok = ftok.clone()
            else:
                used.add(fid)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'setting norm <{norm}> -> {ftok}')
            ftok.norm = norm
            toks.append(ftok)
        fsent.tokens = tuple(toks)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"split: {' '.join(map(str, fsent.token_iter()))}")

    def _create_amr_doc(self, fdoc: FeatureDocument) -> AmrDocument:
        """Create an AMR document from the feature document by first annotating
        with token instances.  Since the token annotator takes a spaCy document,
        we adapt a feature document to a spaCy doc.

        We don't clone despite document modifications (split on
        whitespaces/hyphens and token reindexes) so the alignment matches
        correctly.

        """
        sdoc: Doc
        fdoc.update_indexes()
        if fdoc.spacy_doc is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('No spaCy doc in feature document, using adapted')
            sdoc = SpacyDocAdapter(fdoc)
        else:
            # add spacy_token back to tokens
            fdoc.set_spacy_doc(fdoc.spacy_doc)
            sdoc = SpacyDocAdapter(fdoc)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating AMR doc from {fdoc}')
        ssent: FeatureSentence
        for fsent, ssent in zip(fdoc, sdoc.sents):
            assert isinstance(fsent, FeatureSentence)
            self._split_toks(fsent, ssent)
        if logger.isEnabledFor(logging.DEBUG):
            sp = '|'.join(map(lambda t: t.orth_, sdoc))
            logger.debug(f'generated spacy doc: {sdoc} ({sp})')
        self.amr_parser(sdoc)
        if self.alignment_populator is not None:
            self.alignment_populator.align(sdoc)
        return sdoc._.amr

    def annotate(self, doc: FeatureDocument) -> AmrFeatureDocument:
        """Parse, annotate and annotate a new AMR feature document using
        features from ``doc``.  Since the AMR document itself is not cached,
        using a separate document cache is necessary for caching/storage.

        :param doc: the source feature document to parse in to AMRs

        :param key: the key used to cache the :class:`.AmrDocument`. in
                    :obj:`stash` if provided (see class docs)

        """
        amr_doc: AmrDocument = None
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'annotating {tw.shorten(str(doc), 60)}')
        if self.reparse:
            amr_doc = self._create_amr_doc(doc)
        else:
            amr_doc = doc.amr
        fsents: List[AmrFeatureSentence] = []
        for amr_sent, psent in zip(amr_doc.sents, doc.sents):
            amr_fsent = psent.clone(self.amr_sent_class, amr=amr_sent)
            fsents.append(amr_fsent)
        amr_fdoc: AmrFeatureDocument = doc.clone(
            self.amr_doc_class, sents=tuple(fsents), amr=amr_doc)
        super().decorate(amr_fdoc)
        if self.coref_resolver is not None:
            self.coref_resolver(amr_fdoc)
        return amr_fdoc

    def parse(self, text: str, *args, **kwargs) -> FeatureDocument:
        self._log_parse(text, logger)
        doc, key, loaded = self._load_or_parse(text, False, *args, **kwargs)
        if not loaded:
            doc = self.annotate(doc)
            if self.stash is not None:
                self.stash.dump(key, doc)
        return doc

    def clear(self):
        super().clear()
        if self.coref_resolver is not None:
            self.coref_resolver.clear()
