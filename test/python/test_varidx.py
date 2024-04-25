import sys
from io import StringIO
from pathlib import Path
from zensols.amr import AmrDocument
from util import BaseTestApplication


class TestReindex(BaseTestApplication):
    def setUp(self):
        self.maxDiff = sys.maxsize

    def test_reindex(self):
        write: bool = 0
        write_formatted: bool = 0

        should_file: Path = Path('test-resources/lp-reindexed.txt')
        doc = AmrDocument.from_source(Path('test-resources/lp.txt'))
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
            with open(should_file, 'w') as f:
                f.write(actual)
        with open(should_file) as f:
            should = f.read()
        self.assertEqual(should, actual)
