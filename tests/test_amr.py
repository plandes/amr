from typing import List, Set
import sys
from io import BytesIO
import pickle
import json
from pathlib import Path
from util import BaseTestApplication
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from penman.graph import Graph
from zensols.nlp import FeatureDocumentParser
from zensols.amr import (
    AmrSentence, AmrDocument, AmrFeatureDocument, AmrFeatureSentence,
    Dumper, Application
)
from zensols.amr.annotate import AnnotationFeatureDocumentParser


class TestApplication(BaseTestApplication):
    _DEFAULT_MODEL = 'xfm_bart_base'
    _DEFAULT_TEST = f'{_DEFAULT_MODEL}-test'

    def setUp(self):
        super().setUp()
        self.maxDiff = sys.maxsize
        self.sent = ("""\
Barack Hussein Obama II is an American politician who served as the 44th \
president of the United States from 2009 to 2017. A member of the Democratic \
Party, he was the first African-American president of the United States.\
""")
        self._clean_targ()

    def _get_model_app(self, model_name: str, config: str = 'test'):
        self.model_name = model_name
        return self._get_app(config)

    def _get_should_sent_path(self, sent_num: int, desc: str = None) -> Path:
        desc = self.model_name if desc is None else desc
        return Path(f'test-resources/{desc}-sent-{sent_num}.txt')

    def _get_should_sent(self, sent_num: int, desc: str = None) -> str:
        path = self._get_should_sent_path(sent_num, desc)
        if path.is_file():
            with open(path) as f:
                return f.read().strip()

    def _test_doc(self, doc: AmrDocument, desc: str = None, show: bool = False,
                  write: bool = False):
        ssent1 = self._get_should_sent(1, desc)
        ssent2 = self._get_should_sent(2, desc)

        self.assertEqual(AmrDocument, type(doc))
        self.assertEqual(2, len(doc))
        self.assertEqual(AmrSentence, type(doc[0]))
        self.assertEqual(AmrSentence, type(doc[1]))
        self.assertEqual(Graph, type(doc[0].graph))
        self.assertEqual(str, type(doc[0].graph_string))
        self.assertEqual(Graph, type(doc[1].graph))
        self.assertEqual(str, type(doc[1].graph_string))
        doc.normalize()
        if show:
            print(f'should:\n{ssent1}\n')
            print(f'generated:\n{doc[0].graph_string}\n')
            print(f'should:\n{ssent2}\n')
            print(f'generated:\n{doc[1].graph_string}\n')
        if write:
            for i in range(2):
                path = self._get_should_sent_path(i + 1, desc)
                with open(path, 'w') as f:
                    f.write(doc[i].graph_string)
                print(f'wrote: {path}')
        else:
            self.assertEqual(ssent1, doc[0].graph_string)
            self.assertEqual(ssent2, doc[1].graph_string)

    def _test_create_data(self):
        model_test = self._DEFAULT_TEST
        app: Application = self._get_model_app(f'{model_test}')
        spacy_doc = app.doc_parser.parse_spacy_doc(self.sent)
        doc: AmrDocument = spacy_doc._.amr
        sent: AmrSentence
        for six, sent in enumerate(doc):
            tr_path: Path = f'test-resources/{model_test}-sent-{six + 1}.txt'
            with open(tr_path, 'w') as f:
                f.write(sent.graph_string + '\n')
            print(f'wrote: {tr_path}')

    def _fix_sent_norm(self, sent):
        meta = sent.metadata
        snt = meta['snt']
        meta['snt'] = snt.replace('African - American', 'African-American')
        sent.metadata = meta

    def _test_amr_clone(self):
        with open('test-resources/t5-test-sent-1.txt') as f:
            content = f.read()
        sent = AmrSentence(content)
        clone = sent.clone()
        self.assertFalse(id(sent) == id(clone))
        self.assertEqual(sent, clone)
        sent.set_metadata('v1', 'value1')
        self.assertEqual(len(sent.metadata) - 1, len(clone.metadata))
        self.assertNotEqual(sent, clone)

    def test_parse(self):
        app: Application = self._get_model_app(self._DEFAULT_TEST)
        self.assertEqual(Application, type(app))
        doc_parser = app.config_factory('amr_pipeline_doc_parser')
        spacy_doc = doc_parser.parse_spacy_doc(self.sent)
        self.assertEqual(Doc, type(spacy_doc))
        doc = spacy_doc._.amr
        self.assertEqual(2, len(tuple(spacy_doc.sents)))
        self.assertEqual(Span, type(next(iter(spacy_doc.sents))))
        self.assertEqual(AmrSentence, type(next(iter(spacy_doc.sents))._.amr))
        self._test_doc(doc)

    def test_pickle(self):
        app: Application = self._get_model_app(self._DEFAULT_TEST)
        doc_parser = app.config_factory('amr_pipeline_doc_parser')
        spacy_doc = doc_parser.parse_spacy_doc(self.sent)
        doc = spacy_doc._.amr
        self._test_doc(doc)
        bio = BytesIO()
        pickle.dump(doc, bio)
        bio.seek(0)
        doc2 = pickle.load(bio)
        self._test_doc(doc2)
        for sold, snew in zip(doc, doc2):
            self.assertEqual(sold.graph_string, snew.graph_string)

    def test_doc_clone(self):
        app: Application = self._get_model_app(self._DEFAULT_TEST)
        doc_parser = app.config_factory('amr_pipeline_doc_parser')
        spacy_doc = doc_parser.parse_spacy_doc(self.sent)
        doc: AmrDocument = spacy_doc._.amr
        sent = doc[0]
        clone = sent.clone()
        self.assertFalse(id(sent) == id(clone))
        self.assertEqual(sent, clone)
        sent.set_metadata('v1', 'value1')
        self.assertEqual(len(sent.metadata) - 1, len(clone.metadata))
        self.assertNotEqual(sent, clone)

    def _test_align(self, sent: AmrFeatureSentence):
        self.assertTrue(sent.amr.has_alignments)
        with open('test-resources/align.json') as f:
            should = json.load(f)
        sals = {str(x[0]): str(x[1]) for x in sent.indexed_alignments.items()}
        if 0:
            print(json.dumps(sals, indent=4))
        self.assertEqual(should, sals)

    def test_annotator(self):
        app: Application = self._get_model_app(self._DEFAULT_TEST)
        doc_parser = app.config_factory('amr_anon_doc_parser')
        self.assertTrue(isinstance(doc_parser, AnnotationFeatureDocumentParser))
        doc: AmrFeatureDocument = doc_parser(self.sent)
        self.assertEqual(AmrFeatureDocument, type(doc))
        self.assertEqual(2, len(tuple(doc.sents)))
        self.assertEqual(AmrFeatureSentence, type(next(iter(doc.sents))))
        self.assertEqual(AmrDocument, type(doc.amr))
        self._test_align(doc[0])
        sent: AmrSentence = doc.amr[1]
        self._fix_sent_norm(sent)
        self._test_doc(doc.amr)

    def test_filtering_annotator(self):
        app: Application = self._get_model_app('filter-test', 'test-filter')
        doc_parser = app.config_factory('amr_anon_doc_parser')
        doc: AmrFeatureDocument = doc_parser(self.sent)
        self.assertEqual(AmrFeatureDocument, type(doc))
        self.assertEqual(2, len(tuple(doc.sents)))
        self.assertEqual(AmrFeatureSentence, type(next(iter(doc.sents))))
        self.assertEqual(AmrDocument, type(doc.amr))
        self._test_doc(doc.amr, 'filter')

    def test_annotator_reload(self):
        app: Application = self._get_model_app(self._DEFAULT_TEST)
        doc_parser = app.config_factory('amr_anon_doc_parser')
        doc: AmrFeatureDocument = doc_parser(self.sent)
        doc2: AmrFeatureDocument = doc_parser(self.sent)
        self.assertNotEqual(id(doc), id(doc2))
        self.assertEqual(doc.text, doc2.text)
        self.assertEqual(doc.amr.graph_string, doc2.amr.graph_string)
        self.assertEqual(doc, doc2)
        sent2: AmrSentence = doc2.amr[1]
        self._fix_sent_norm(sent2)
        self._test_doc(doc2.amr)

    def test_dumper(self):
        bd: Path = self.target_dir
        dd: Path = bd / 'barack-hussein-obama-ii-is-an'
        app: Application = self._get_model_app(self._DEFAULT_TEST)
        doc_parser: FeatureDocumentParser = app.config_factory(
            'amr_pipeline_doc_parser')
        dumper: Dumper = app.config_factory('amr_dumper')
        dumper.target_dir = bd
        doc: AmrFeatureDocument = doc_parser(self.sent)
        paths: List[Path] = dumper(doc.amr)
        should: Set[Path] = {
            dd / 'barack-hussein-obama-ii-is-an.txt',
            dd / 'barack-hussein-obama-ii-is-an.pdf',
            dd / 'a-member-of-the-democratic-par.pdf',
            dd / 'a-member-of-the-democratic-par.txt',
            dd / 'doc.txt'}
        self.assertEqual(len(should), len(paths))
        self.assertEqual(should, set(paths))
        path: Path
        for path in paths:
            self.assertTrue(path.is_file(), f'dumper failed to create: {path}')
