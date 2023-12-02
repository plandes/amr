import sys
import unittest
import shutil
from pathlib import Path
from zensols.cli import CliHarness
from zensols.amr import (
    AmrDocument, AmrFeatureDocument, AnnotatedAmrDocument,
    ApplicationFactory
)


class TestSlice(unittest.TestCase):
    def setUp(self):
        self.maxDiff = sys.maxsize
        hrn = CliHarness(app_factory_class=ApplicationFactory)
        cmd = 'parse _ -c test-resources/lp.conf --level warn'
        self.inst = hrn.get_instance(cmd)
        self.astash = self.inst.config_factory('amr_anon_feature_doc_stash')
        self.assertFalse(self.inst is None)
        targ = Path('target')
        if targ.is_dir():
            shutil.rmtree(targ)

    def _test_amr_docs_eq(self, doc, clone, deep=False):
        self.assertEqual(AnnotatedAmrDocument, type(clone))
        self.assertEqual(2, len(clone))
        self.assertNotEqual(id(doc), id(clone))
        if not deep:
            self.assertNotEqual(id(doc[1]), id(clone[0]))
        self.assertEqual(doc[1], clone[0])

    def _test_doc_amr_sents(self, fdoc):
        for sent in fdoc:
            self.assertEqual(sent.text, sent.amr.text)

    def _test_amr(self):
        k, doc = next(iter(self.astash))
        self.assertTrue(isinstance(doc, AmrFeatureDocument))
        self.assertEqual('1943', k)
        self.assertEqual(4, len(doc))
        doc = doc.amr
        self.assertEqual(AnnotatedAmrDocument, type(doc))
        self.assertEqual(4, len(doc))

        s1: AmrDocument = doc.from_sentences(doc.sents[1:3])
        self.assertEqual(AnnotatedAmrDocument, type(s1))
        self.assertEqual(2, len(s1))
        self.assertNotEqual(id(doc), id(s1))
        self.assertEqual(id(doc[1]), id(s1[0]))

        s2: AmrDocument = doc.from_sentences(doc.sents[0:4])
        self.assertEqual(4, len(s2))
        self.assertNotEqual(id(doc), id(s2))
        self.assertEqual(doc, s2)

        s3: AmrDocument = doc.from_sentences(doc.sents[1:3], deep=True)
        self._test_amr_docs_eq(doc, s3)

    def test_doc(self):
        k, doc = next(iter(self.astash))
        self.assertTrue(isinstance(doc, AmrFeatureDocument))
        self.assertEqual('1943', k)
        self.assertEqual(4, len(doc))

        s1: AmrFeatureDocument = doc.from_sentences(doc.sents[1:3])
        self.assertTrue(isinstance(doc, AmrFeatureDocument))
        self.assertEqual(2, len(s1))
        self.assertNotEqual(id(doc), id(s1))
        self.assertEqual(id(doc[1]), id(s1[0]))
        self._test_doc_amr_sents(s1)

        s2: AmrFeatureDocument = doc.from_sentences(doc.sents[0:4])
        self.assertEqual(4, len(s2))
        self.assertNotEqual(id(doc), id(s2))
        self.assertEqual(doc, s2)
        self._test_doc_amr_sents(s2)

        s3: AmrFeatureDocument = doc.from_sentences(doc.sents[1:3], deep=True)
        self.assertEqual(AmrFeatureDocument, type(s3))
        self.assertEqual(2, len(s3))
        self.assertNotEqual(id(doc), id(s3))
        self.assertNotEqual(id(doc[1]), id(s3[0]))
        self.assertEqual(doc[1], s3[0])
        self._test_amr_docs_eq(doc.amr, s3.amr, deep=True)
        self._test_doc_amr_sents(s3)
