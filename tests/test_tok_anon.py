import sys
from io import StringIO
from util import BaseTestApplication
from zensols.nlp import FeatureDocumentParser
from zensols.amr import AmrFeatureDocument, Application
from zensols.amr.docparser import TokenAnnotationFeatureDocumentDecorator


class TestApplication(BaseTestApplication):
    def setUp(self):
        self.maxDiff = sys.maxsize
        self.sent = ("""\
Barack Hussein Obama II is an American politician who served as the 44th \
president of the United States from 2009 to 2017.\
""")
        self._clean_targ()

    def _get_parser(self, method: str) -> FeatureDocumentParser:
        app: Application = self._get_app('test-tok-anon')
        self.assertEqual(Application, type(app))
        dec: TokenAnnotationFeatureDocumentDecorator = \
            app.config_factory('amr_token_ent_doc_decorator')
        dec.method = method
        self.assertTrue(isinstance(dec, TokenAnnotationFeatureDocumentDecorator))
        return app.config_factory('amr_anon_doc_parser')

    def _test_ann(self, should_name: str, method: str, write: bool = False):
        DEBUG: bool = 0
        path: str = f'test-resources/tok-anon/{should_name}.txt'
        parser: FeatureDocumentParser = self._get_parser(method)
        doc: AmrFeatureDocument = parser(self.sent)
        if DEBUG:
            doc.amr.write()
            return
        if write:
            with open(path, 'w') as f:
                doc.amr.write(writer=f)
        with open(path) as f:
            should = f.read()
        sio = StringIO()
        doc.amr.write(writer=sio)
        self.assertEqual(should, sio.getvalue())

    def test_ann_node(self):
        self._test_ann('attribute', 'attribute')

    def test_ann_epi(self):
        self._test_ann('epi', 'epi')

    def test_ann_format(self):
        self._test_ann('format', '{target}~{value}')
