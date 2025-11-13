from typing import Dict
import sys
import collections
from io import StringIO
from pathlib import Path
from zensols.amr import AmrDocument
from util import BaseTestApplication


class TestReindex(BaseTestApplication):
    def setUp(self):
        self.maxDiff = sys.maxsize
        self.should_file: Path = Path('test-resources/lp-reindexed.txt')

    def _get_doc(self) -> AmrDocument:
        return AmrDocument.from_source(Path('test-resources/lp.txt'))

    def _test_reindex_unique(self, doc, max_cnt: int, should: Dict[str, int]):
        debug: bool = 0
        cnts: Dict[str, int] = collections.defaultdict(lambda: 0)
        for sent in doc.sents:
            for var in sent.graph.variables():
                cnts[var] += 1
        if debug:
            from pprint import pprint
            pprint(dict(cnts))
        self.assertEqual(max_cnt, max(cnts.values()))
        if should is None:
            should = [1] * len(cnts)
            self.assertEqual(should, list(cnts.values()))
        else:
            for k, v in should.items():
                self.assertEqual(v, cnts[k])

    def test_reindex_unique(self):
        doc: AmrDocument = self._get_doc()
        self._test_reindex_unique(doc, 3, dict(p=3))
        doc.reindex_variables()
        self._test_reindex_unique(doc, 1, None)

    def test_reindex_all_lp(self):
        app = self._get_app()
        inst = app.config_factory('amr_anon_corpus_installer')
        lp: Path = inst.get_singleton_path()
        inst()
        self.assertTrue(lp.is_file())
        doc = AmrDocument.from_source(lp)
        self._test_reindex_unique(doc, 721, dict(p=531))
        doc.reindex_variables()
        self._test_reindex_unique(doc, 1, None)

    def test_reindex_by_write(self):
        write: bool = 0
        write_formatted: bool = 0

        doc: AmrDocument = self._get_doc()
        if write_formatted:
            for s in doc.sents:
                s.graph
                s.invalidate_graph_string()
            with open(Path('test-resources/lp-formatted.txt'), 'w') as f:
                doc.write(writer=f)
            return

        doc.reindex_variables()
        sio = StringIO()
        doc.write(writer=sio)
        actual: str = sio.getvalue()
        if write:
            with open(self.should_file, 'w') as f:
                f.write(actual)
        with open(self.should_file) as f:
            should = f.read()
        self.assertEqual(should, actual)
