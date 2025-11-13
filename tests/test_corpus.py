from typing import Dict, Tuple, List
import sys
import unittest
import shutil
from pathlib import Path
from zensols.cli import CliHarness
from penman.surface import Alignment
from penman import constant, Graph
from zensols.amr import (
    ApplicationFactory, AmrFeatureSentence, AmrFeatureDocument
)
from zensols.amr.annotate import (
    AnnotatedAmrSentence, AnnotatedAmrDocument, AnnotatedAmrFeatureDocumentStash
)


class TestAnnotatedCorpus(unittest.TestCase):
    def setUp(self):
        self.maxDiff = sys.maxsize
        hrn = CliHarness(app_factory_class=ApplicationFactory)
        cmd = 'parse _ -c test-resources/lp.conf --level warn'
        self.inst = hrn.get_instance(cmd)
        self.assertFalse(self.inst is None)
        targ = Path('target')
        if targ.is_dir():
            shutil.rmtree(targ)

    def test_corp_anon(self):
        astash = self.inst.config_factory('amr_anon_doc_stash')
        k, doc = next(iter(astash))
        self.assertEqual('1943', k)
        self.assertEqual(AnnotatedAmrDocument, type(doc))
        self.assertEqual(4, len(doc))
        should = """\
Chapter 1 .
Once when I was six years old I saw a magnificent picture in a book , called True Stories from Nature , about the primeval forest .
It was a picture of a boa constrictor in the act of swallowing an animal .
Here is a copy of the drawing ."""
        self.assertEqual(should.split('\n'), [s.text for s in doc])

    def test_corp_feat_anon(self):
        def update(old, new, idx):
            self.assertEqual(old, tbs[idx])
            tbs[idx] = new

        astash = self.inst.config_factory('amr_anon_feature_doc_stash')
        self.assertEqual(AnnotatedAmrFeatureDocumentStash, type(astash))
        doc: AmrFeatureDocument = astash['1943']
        self.assertEqual(AmrFeatureDocument, type(doc))
        self.assertFalse(doc.amr is None)

        sent: AmrFeatureSentence = doc[1]
        self.assertEqual(AmrFeatureSentence, type(sent))

        amr_sent: AnnotatedAmrSentence = sent.amr
        self.assertEqual(AnnotatedAmrSentence, type(amr_sent))

        g: Graph = amr_sent.graph
        tbs = {x[0]: x[1].norm for x in sent.tokens_by_i_sent.items()}

        tbs[5] = 'year'
        update('six', 6, 4)
        update('saw', 'see-01', 8)
        update('I', 'i', 2)
        update('Once', 'once', 0)

        epis: Dict[Tuple[str, str, str], List] = g.epidata
        for (s, r, t), al in epis.items():
            als = tuple(map(lambda a: a.indices,
                            filter(lambda x: isinstance(x, Alignment), al)))
            if len(als) == 0:
                continue
            self.assertEqual(1, len(als))
            als = als[0]
            self.assertEqual(1, len(als))
            idx = als[0]
            doc_tok = tbs.get(idx)
            graph_tok = t if r == ':instance' else g.attributes(s, r)[0].target
            graph_tok = constant.evaluate(graph_tok)
            #print(doc_tok, ':', graph_tok, idx)
            self.assertEqual(doc_tok, graph_tok)
