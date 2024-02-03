from typing import Union
import unittest
from zensols.util import Failure
from zensols.cli import CliHarness
from zensols.amr import Application, ApplicationFactory


class BaseTestApplication(unittest.TestCase):
    def _get_app(self, config: str = 'test') -> Application:
        model: str = 'gsii'
        if hasattr(self, '_DEFAULT_MODEL'):
            model = self._DEFAULT_MODEL
        hrn = CliHarness(app_factory_class=ApplicationFactory)
        cmd = (f'parse _ -c test-resources/{config}.conf --level warn ' +
               f'--override amr_default.parse_model={model}')
        inst: Union[Failure, Application] = hrn.get_instance(cmd)
        self.assertFalse(inst is None)
        if isinstance(inst, Failure):
            inst.rethrow()
        return inst
