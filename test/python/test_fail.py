from typing import Tuple
from io import StringIO
import unittest
from zensols.cli import CliHarness
from zensols.amr import (
    AmrError, AmrFailure, AmrSentence, AmrDocument, AmrFeatureDocument,
    ApplicationFactory, Application,
)


class TestApplication(unittest.TestCase):
    _DEFAULT_MODEL = 'gsii'
    _DEFAULT_TEST = f'{_DEFAULT_MODEL}-test'

    def _app(self, model_name: str, config: str = 'test'):
        self.model_name = model_name
        hrn = CliHarness(app_factory_class=ApplicationFactory)
        cmd = (f'parse _ -c test-resources/{config}.conf --level crit ' +
               f'--override amr_default.parse_model={self._DEFAULT_MODEL}')
        inst = hrn.get_instance(cmd)
        self.assertFalse(inst is None)
        return inst

    def test_parse(self):
        app: Application = self._app(self._DEFAULT_TEST)
        self.assertEqual(Application, type(app))
        astash = app.config_factory('amr_anon_feature_doc_stash')
        sent = AmrSentence('(h / have-org-role-91 :ARG0 (h2 / he :toki1 "34"))')
        sent.set_metadata('snt', 'Test sentence.')
        doc = AmrDocument([sent])
        sent.graph_string = '(h / have-org-role-91'
        with self.assertRaisesRegex(AmrError, "^Could not parse"):
            astash.to_feature_doc(doc, catch=False)
        fdoc: AmrFeatureDocument
        fails: Tuple[AmrFailure]
        fdoc, fails = astash.to_feature_doc(doc, catch=True)
        self.assertTrue(isinstance(fdoc, AmrFeatureDocument))
        self.assertEqual(tuple, type(fails))
        self.assertEqual(1, len(fails))
        fail: AmrFailure = fails[0]
        self.assertEqual(AmrFailure, type(fail))
        should = """\
Could not parse: 
  line 1
    (h / have-org-role-91
                         ^
DecodeError: Unexpected end of input: (h / have-org-role-91\
"""
        self.assertEqual(should, str(fail))
        sio = StringIO()
        fail.print_stack(writer=sio)
        stack: str = sio.getvalue()
        self.assertTrue(stack.startswith('Traceback (most recent call last):'))
        self.assertTrue(len(stack) > 1000)
