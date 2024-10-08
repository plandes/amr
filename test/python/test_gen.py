from util import BaseTestApplication
from zensols.amr import AmrFeatureDocument, AmrGeneratedDocument


class TestGeneration(BaseTestApplication):
    def setUp(self):
        super().setUp()
        fac = self._get_app().config_factory
        self.parser = fac('amr_anon_doc_parser')
        self.gen = fac('amr_generator_amrlib')

    def test_gen(self):
        s: str = 'Obama was the 44th president last year. He is no longer.'
        doc = self.parser(s)
        self.assertTrue(isinstance(doc, AmrFeatureDocument))
        gdoc: AmrGeneratedDocument = self.gen(doc.amr)
        self.assertEqual(AmrGeneratedDocument, type(gdoc))
        # starting
        should = ('Obama was 44th president last year.',
                  "He's no longer.")
        self.assertEqual(should, tuple(map(lambda s: s.text, gdoc)))
