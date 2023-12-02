#!/usr/bin/env python

"""This example shows how to use the AMR API using a custom application context.
This is useful when you want to customize the parser, such as using limited
token features as given in the ``amr_pipline_doc_parser`` overridden section,
which is used in place of the default ``amr_anon_doc_parser``.  The latter
caches AMR parses and reuses them when reparsing the same text.

Use ``./appctx.py -h`` for command line help.

"""

from dataclasses import dataclass
import logging
from io import StringIO
from penman.graph import Graph
from zensols.cli import CliHarness, ProgramNameConfigurator
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.amr import AmrDocument, AmrSentence

logger = logging.getLogger(__name__)

CONFIG = """
[default]
root_dir = ${appenv:root_dir}

[cli]
apps = list: log_cli, app

[log_cli]
class_name = zensols.cli.LogConfigurator
format = ${program:name}: %%(message)s
log_name = ${program:name}
level = debug

[import]
sections = list: imp_conf

# import the 'zensols.amr' library with the gsii model
[imp_conf]
type = importini
config_files = list:
  resource(zensols.util): resources/default.conf,
  resource(zensols.amr): resources/default.conf,
  resource(zensols.nlp): resources/obj.conf,
  resource(zensols.nlp): resources/component.conf,
  resource(zensols.amr): resources/obj.conf,
  resource(zensols.amr): resources/annotate.conf

# override the parse to keep only the norm, ent
[amr_pipline_doc_parser]
token_feature_ids = eval: set('ent_ tag_'.split())

[app]
class_name = ${program:name}.Application
# uncomment the following line and comment the one after to use a caching parser
#doc_parser = instance: amr_anon_doc_parser
doc_parser = instance: amr_pipline_doc_parser
"""


@dataclass
class Application(object):
    """Demostrate how to use the AMR framework with an application context

    """
    doc_parser: FeatureDocumentParser

    def parse(self, sent: str = None):
        """Parse a sentence in to an AMR and show it.

        """
        if sent is None:
            sent = """

He was George Washington and first president of the United States.
He was born On February 22, 1732.

""".replace('\n', ' ').strip()
        # the parser creates a NLP centric feature document as provided in the
        # zensols.nlp package
        doc: FeatureDocument = self.doc_parser(sent)
        # the AMR object graph data structure is provided in the feature
        # document
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


if (__name__ == '__main__'):
    CliHarness(
        app_config_resource=StringIO(CONFIG),
        app_config_context=ProgramNameConfigurator(
            None, default='appctx').create_section(),
        proto_args='',
        proto_factory_kwargs={'reload_pattern': '^appctx'},
    ).run()
