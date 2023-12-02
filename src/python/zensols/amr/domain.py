"""Error and exception classes.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
from zensols.util import APIError, Failure

logger = logging.getLogger(__name__)


@dataclass
class AmrFailure(Failure):
    """A container class that describes AMR graph creation or handling error.

    """
    sent: str = field(default=None)
    """The natural language sentence that cased the error (usually parsing)."""


class AmrError(APIError):
    """Raised for package API errors.

    """
    def __init__(self, msg: str, sent: str = None):
        if sent is not None:
            msg = f'{msg}: {sent}'
        super().__init__(msg)
        self.sent = sent
