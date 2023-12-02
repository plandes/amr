#!/usr/bin/env python

"""This example uses the parser as it is configured for the command line
application.  See the ``appctx.py`` example to customize/override the
default configuration and use different parsers.

"""

from penman.graph import Graph
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.amr import AmrDocument, AmrSentence, ApplicationFactory


if __name__ == '__main__':
    sent: str = """

He was George Washington and first president of the United States.
He was born On February 22, 1732.

""".replace('\n', ' ').strip()
    # get the AMR document parser
    doc_parser: FeatureDocumentParser = ApplicationFactory.get_doc_parser()
    # the parser creates a NLP centric feature document as provided in the
    # zensols.nlp package
    doc: FeatureDocument = doc_parser(sent)
    # the AMR object graph data structure is provided in the feature document
    amr_doc: AmrDocument = doc.amr
    # dump a human readable output of the AMR document
    amr_doc.write()
    # get the first AMR sentence instance
    amr_sent: AmrSentence = amr_doc.sents[0]
    print('sentence:')
    print(' ', amr_sent.text)
    print('tuples:')
    # show the Penman graph representation
    pgraph: Graph = amr_sent.graph
    print(f'variables: {", ".join(pgraph.variables())}')
    for t in pgraph.triples:
        print(' ', t)
    print('edges:')
    for e in pgraph.edges():
        print(' ', e)
