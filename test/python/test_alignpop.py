from typing import List
from pathlib import Path
from penman.graph import Graph
from penman.model import Model
import penman.models.noop
from amrlib.graph_processing.amr_loading import load_amr_entries
from amrlib.alignments.faa_aligner import FAA_Aligner
from zensols.amr.alignpop import AlignmentPopulator, PathAlignment
from util import BaseTestApplication


class TestAlignmentPopulation(BaseTestApplication):
    def setUp(self):
        self.fac = self._get_app('test-alignpop').config_factory

    def test_align(self):
        inst = self.fac('amr_anon_corpus_installer')
        inst()
        path: Path = inst.get_singleton_path()
        model: Model = penman.models.noop.model
        graph_strs: List[str] = load_amr_entries(str(path))
        ugraphs: List[Graph] = [
            penman.decode(gs, model=model) for gs in graph_strs]
        sents: List[str] = [g.metadata['snt'] for g in ugraphs]
        inference = FAA_Aligner()
        agraphs, aligns = inference.align_sents(sents, graph_strs)
        tups = zip(sents, ugraphs, agraphs, aligns)
        for i, (sent, ugraph, agraph_str, align) in enumerate(tups):
            agraph: Graph = penman.decode(agraph_str, model=model)
            agraph.metadata['snt'] = sent
            # fast align leaves a space at the end of the alignment string
            agraph.metadata['alignments'] = align.strip()
            ap = AlignmentPopulator(graph=agraph)
            aligns: List[PathAlignment]
            misses = ()
            try:
                misses = ap.get_missing_alignments()
            except Exception as e:
                print(f'failed parse ({i}): {e}')
                print(penman.encode(agraph, model=model))
                self.assertTrue(False)
            if len(misses) > 0:
                print(f'found {len(misses)} missing alignments (i={i}):')
                for miss in misses:
                    print(f'missing: {miss}')
                print(penman.encode(agraph, model=model))
                self.assertTrue(False)
