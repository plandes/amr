"""Error and exception classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from dataclasses import dataclass, field
import logging
from typing import Type
from penman.epigraph import Epidatum
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
            msg = f'{msg}: <{sent}>'
        super().__init__(msg)
        self.sent = sent
        self.message = msg

    def to_failure(self) -> AmrFailure:
        """Create an :class:`.AmrFailure` from this error."""
        return AmrFailure(
            exception=self,
            message=self.message,
            sent=self.sent)


class FeatureMarker(Epidatum):
    __slots__ = ('feat_id', 'value')

    def __init__(self, feat_id: str, value: str):
        super().__init__()
        self.feat_id = feat_id
        self.value = value

    @classmethod
    def from_string(cls: Type[Feature], s: str) -> Feature:
        s = s.lstrip('~')
        return cls(*s.split('.'))

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        return self.feat_id == other.feat_id and self.value == other.value

    def __str__(self):
        return f'~{self.feat_id}.{self.value}'

    def __repr__(self) -> str:
        return f"{self.feat_id}='{self.value}'"


class Feature(FeatureMarker):
    __slots__ = ()
    mode = 2
