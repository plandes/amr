"""A utility class to reindex variables in an :class`.AmrDocument`.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple, List, Sequence, Any, Set, Union
from dataclasses import dataclass
import logging
import collections
from itertools import chain
from penman.graph import Graph, Instance, Attribute, Edge
from penman.layout import Push
from . import AmrError, AmrSentence

logger = logging.getLogger(__name__)


@dataclass
class VariableIndexer(object):
    """This reentrant class reindexes all variables for sentences of a
    :class:`.AmrDocument` so all node variables are unique.  This is done by:

      1. Index concepts by the first character of their name (i.e. ``s`` for
         ``see-01``) across sentences.

      2. Compile a list of variable replacements (i.e. ``s2`` -> ``s5``) on a
         per sentence basis.

      3. Replace variable names based on their document level index order
         (i.e. ``s``, ``s2``, etc).  This is done for all concepts, edges,
         roles, and the epigraph.  A new graph is created for those that have at
         least one modification, otherwise the original sentence is kept.

    """
    def _get_concept_index(self, sents: Sequence[AmrSentence]) -> \
            Dict[str, Tuple[int, Instance]]:
        """Create an index of single letter non-indexed variable (with out the
        index digits after the variable letter) to concept a list of concept
        triples having the non-indexed variable.

        :return: a mapping of variable to a list of ``(<sent index> , <concept
                 triple>)``

        """
        index: Dict[str, Tuple[int, Instance]] = collections.defaultdict(list)
        six: int
        sent: AmrSentence
        for six, sent in enumerate(sents):
            g: Graph = sent.graph
            inst: Instance
            for inst in g.instances():
                vname: str = inst.target[0]
                index[vname].append((six, inst))
        return index

    def _create_sent_repls(self, c_ix: Dict[str, Tuple[int, Instance]]) -> \
            Dict[int, Dict[str, Tuple[str, Instance]]]:
        """Create by sentence triple replacements.

        :param c_ix: index of single letter non-indexed variable

        :return: a mapping of sentence index to a sentence level mapping of
                 original (source) indexed varaible name

        """
        s_ix: Dict[int, Dict[str, Tuple[str, Instance]]] = \
            collections.defaultdict(dict)
        vname: str  # non-indexed variable name
        insts: Tuple[Instance, ...]
        for vname, insts in c_ix.items():
            vix: int  # variable index
            six: int  # sentence index
            for vix, (six, inst) in enumerate(insts):
                src: str = inst.source
                idx: str = str(vix + 1) if vix > 0 else ''
                targ: str = vname + idx
                if src != targ:
                    s_ix[six][src] = targ
        return s_ix

    def _replace_graphs(self, s_ix: Dict[int, Dict[str, Tuple[str, Instance]]],
                        sents: Sequence[AmrSentence]):
        """Rename the variables and replace graphs for those that have at least
        one replacement.

        :param s_ix: sentence triple replacements

        :param sents: the AMRs that are to be modified

        """
        def map_attr(vals: List[Any], repl: Dict[str, str]):
            repls: List[Any] = []
            for o in vals:
                if isinstance(o, Push):
                    targ: str = repl.get(o.variable)
                    if targ is not None:
                        o = Push(targ)
                repls.append(o)
            return repls

        six: int  # sentence index
        repls: Dict[int, Dict[str, str]]  # replacements
        for six, repls in s_ix.items():
            sent: AmrSentence = sents[six]
            g: Graph = sent.graph
            repl_trips: List[Tuple[Tuple[str, str, str],
                                   Tuple[str, str, str]]] = []
            n_repls: int = 0
            nodes: Set[Union[Instance, Attribute]] = \
                set(chain.from_iterable((g.instances(), g.attributes())))
            edges: Set[Edge] = set(g.edges())
            top: Tuple[str, str, str] = None
            otrip: Tuple[str, str, str]
            for otrip in g.triples:
                rtrip: Tuple[str, str, str] = otrip
                if otrip in nodes:
                    src: str = otrip[0]
                    targ: str = repls.get(src)
                    if g.top == src:
                        assert top is None
                        top = otrip[0] if targ is None else targ
                    if targ is not None:
                        rtrip = (targ, *otrip[1:])
                        n_repls += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'n-s{six}: {otrip} -> {rtrip}')
                elif otrip in edges:
                    ltarg: str = repls.get(otrip[0])
                    rtarg: str = repls.get(otrip[2])
                    if ltarg is not None or rtarg is not None:
                        lval = otrip[0] if ltarg is None else ltarg
                        rval = otrip[2] if rtarg is None else rtarg
                        rtrip = (lval, otrip[1], rval)
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'e-s{six}: {otrip} -> {rtrip}')
                        n_repls += 1
                else:
                    raise AmrError(f'Not a concept, attribute or edge: {otrip}')
                repl_trips.append((otrip, rtrip))
            if n_repls > 0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'sent {six} updates: {n_repls}')
                assert top is not None
                oepis: Dict[Attribute, List] = g.epidata
                repis: Dict[Attribute, List] = dict(map(
                    lambda t: (Attribute(*t[1]), map_attr(oepis[t[0]], repls)),
                    repl_trips))
                sent.graph = Graph(
                    triples=tuple(map(lambda t: t[1], repl_trips)),
                    top=top,
                    epidata=repis,
                    metadata=g.metadata)

    def reindex(self, sents: Sequence[AmrSentence]):
        """Reindex and repalce variables in ``sents``.  Any modified graphs are
        updated in the ``sents`` instances.

        :param sents: sentences whose variables will be reindexed

        """
        c_ix: Dict[str, Tuple[int, Instance]] = \
            self._get_concept_index(sents)
        s_ix: Dict[int, Dict[str, Tuple[str, Instance]]]
        s_ix = self._create_sent_repls(c_ix)
        self._replace_graphs(s_ix, sents)
