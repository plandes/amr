from typing import List, Tuple, Set
import sys
import unittest
from zensols.cli import CliHarness
from zensols.amr import (
    suppress_warnings,
    Relation, RelationSet,
    AmrFeatureDocument, ApplicationFactory
)


class TestCoreference(unittest.TestCase):
    def setUp(self):
        self.maxDiff = sys.maxsize
        hrn = CliHarness(app_factory_class=ApplicationFactory)
        cmd = 'parse _ -c test-resources/lp.conf --level warn'
        self.inst = hrn.get_instance(cmd)
        self.assertFalse(self.inst is None)
        self.astash = self.inst.config_factory('amr_anon_feature_doc_stash')
        suppress_warnings()

    def _assert_magnificent(self, r: Relation):
        self.assertEqual(len(r), 3)
        should: List[str] = ['p / picture'] * 3
        self.assertEqual(should, list(map(lambda r: r.short, r.references)))
        should: Tuple[str] = (
            'magnificent picture book True Stories from Nature primeval forest',
            'It was a picture of a boa constrictor in the act of swallowing an animal .',
            'drawing')
        self.assertEqual(
            should, tuple(map(lambda r: r.subtree.text, r.references)))

    def _assert_temp_quant(self, r: Relation):
        self.assertEqual(len(r), 2)
        should: Tuple[str] = ('t / temporal-quantity', 'i / it')
        self.assertEqual(should, tuple(map(lambda r: r.short, r.references)))

    def _assert_both_rel(self, doc: AmrFeatureDocument):
        relset: RelationSet = doc.relation_set
        self.assertEqual(len(relset), 2)
        r1: Relation = relset[0]
        r2: Relation = relset[1]
        self.assertNotEqual(r1, r2)
        self._assert_magnificent(r1)
        self._assert_temp_quant(r2)

    def test_coref(self):
        k, doc = next(iter(self.astash))
        self.assertTrue(isinstance(doc, AmrFeatureDocument))
        self.assertEqual('1943', k)
        self.assertEqual(4, len(doc))

        self._assert_both_rel(doc)

        doc2: AmrFeatureDocument = doc.from_sentences(doc[1:3])
        if 0:
            doc.write(include_amr=False)
            doc2.write(include_amr=False)

        relset: RelationSet = doc2.relation_set
        self.assertEqual(len(relset), 1)
        r: Relation = relset[0]
        self._assert_temp_quant(r)
        self.assertEqual(doc.relation_set[1], r)

        doc_rels: Set[Relation] = set(doc.relation_set.relations)
        doc2_rels: Set[Relation] = set(doc2.relation_set.relations)
        diff_rels: Set[Relation] = doc_rels - doc2_rels
        self.assertEqual(len(diff_rels), 1)
        diff_rel = next(iter(diff_rels))
        self._assert_magnificent(diff_rel)

        doc3: AmrFeatureDocument = doc.from_sentences(doc[1:4])
        self._assert_both_rel(doc3)
