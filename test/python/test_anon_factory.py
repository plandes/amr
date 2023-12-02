from typing import Tuple
import sys
import unittest
from io import StringIO
from pathlib import Path
from zensols.cli import CliHarness
from transformers import logging as translogging
from zensols.amr import (
    AmrFeatureDocument, AnnotatedAmrFeatureDocumentFactory,
    ApplicationFactory
)


translogging.set_verbosity_error()


class TestAnnotationFactory(unittest.TestCase):
    def setUp(self):
        self.maxDiff = sys.maxsize
        hrn = CliHarness(app_factory_class=ApplicationFactory)
        cmd = 'parse _ -c test-resources/lp.conf --level warn'
        self.inst = hrn.get_instance(cmd)
        self.assertFalse(self.inst is None)
        self.factory: AnnotatedAmrFeatureDocumentFactory = \
            self.inst.config_factory('amr_anon_doc_factory')

    def test_anon_factory(self):
        anon_file = Path('test-resources/anon-factory.json')
        should_file = Path('test-resources/anon-factory.txt')
        docs: Tuple[AmrFeatureDocument] = tuple(self.factory(anon_file))
        self.assertEqual(1, len(docs))

        doc: AmrFeatureDocument = docs[0]
        self.assertTrue(isinstance(doc, AmrFeatureDocument))

        self.assertEqual(3, len(doc.sents))
        self.assertEqual(('ex1.0', 'ex1.1', 'ex1.2'),
                         tuple(map(lambda s: s.metadata['id'], doc.amr.sents)))
        for sent in doc.amr.sents:
            self.assertTrue(len(sent.graph_only) > 10)

        vio = StringIO()
        doc.amr.write(writer=vio, limit_sent=0)
        val = vio.getvalue()
        if 0:
            with open(should_file, 'w') as f:
                f.write(val)
        with open(should_file) as f:
            should = f.read()
        self.assertEqual(should, val)
