"""Command line entry point to the application.

"""
__author__ = 'Paul Landes'

from typing import List, Any, Dict
from dataclasses import dataclass
import sys
from zensols.cli import ApplicationFactory, ActionResult, CliHarness
from zensols.nlp import FeatureDocumentParser
from .dumper import Dumper


@dataclass
class ApplicationFactory(ApplicationFactory):
    def __init__(self, *args, **kwargs):
        kwargs['package_resource'] = 'zensols.amr'
        super().__init__(*args, **kwargs)

    @classmethod
    def get_doc_parser(cls) -> FeatureDocumentParser:
        """Return the natural language parser that also creates AMR graphs as
        the ``amr`` attribute in the document.

        """
        harness: CliHarness = cls.create_harness()
        return harness['app'].doc_parser

    @classmethod
    def get_dumper(cls) -> Dumper:
        """Return a object that plots and writes AMR content in human readable
        formats.

        """
        harness: CliHarness = cls.create_harness()
        return harness['amr_dumper']


def main(args: List[str] = sys.argv, **kwargs: Dict[str, Any]) -> ActionResult:
    harness: CliHarness = ApplicationFactory.create_harness(relocate=False)
    harness.invoke(args, **kwargs)
