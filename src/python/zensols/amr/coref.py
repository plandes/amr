"""Wrap the :mod:`amr_coref` module for AMR Co-refernce resolution.

"""
__author__ = 'Paul Landes'

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import textwrap as tw
import platform
import torch
from zensols.util import time, Hasher
from zensols.persist import persisted, Stash, DictionaryStash
from zensols.install import Installer
from amr_coref.coref.inference import Inference
from . import AmrFailure, AmrFeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class CoreferenceResolver(object):
    """Resolve coreferences in AMR graphs.

    """
    installer: Installer = field()
    """The :mod:`amr_coref` module's coreference module installer."""

    stash: Stash = field(default_factory=DictionaryStash)
    """The stash used to cache results.  It takes a while to inference but the
    results in memory size are small.

    """
    use_multithreading: bool = field(default=True)
    """By default, multithreading is enabled for Linux systems.  However, an
    error is raised when invoked from a child thread.  Set to ``False`` to off
    multithreading for coreference resolution.

    """
    robust: bool = field(default=True)
    """Whether to robustly deal with exceptions in the coreference model.  If
    ``True``, instances of :class:`.AmrFailure` are stored in the stash and
    empty coreferences used for caught errors.

    """
    def _use_multithreading(self) -> bool:
        return self.use_multithreading and \
            not platform.system() != 'Linux'

    @property
    @persisted('_model')
    def model(self) -> Inference:
        """The :mod:`amr_coref` coreference model."""
        use_multithreading: bool = True
        if not self._use_multithreading():
            if logger.isEnabledFor(logging.INFO):
                logger.info('turning off AMR coref multithreading for ' +
                            f'platform: {platform.system()}')
            use_multithreading = False
        self.installer()
        model_path: Path = self.installer.get_singleton_path()
        device = None if torch.cuda.is_available() else 'cpu'
        return Inference(
            str(model_path),
            device=device,
            use_multithreading=use_multithreading)

    def _resolve(self, doc: AmrFeatureDocument) -> \
            Dict[str, List[Tuple[int, str]]]:
        """Use the coreference model and return the output."""
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'resolving coreferences for {doc}')
        model: Inference = self.model
        graph_strs = tuple(map(lambda s: s.amr.graph_string, doc.sents))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('resolving coreferences graph:')
            for gs in graph_strs:
                logger.debug(f'  {gs}')
        with time(f'resolved {len(doc)} sentence coreferences'):
            return model.coreference(graph_strs)

    def _create_key(self, doc: AmrFeatureDocument) -> str:
        """Create a unique key based on the text of the sentences of the
        document.

        """
        hasher = Hasher()
        for sent in doc:
            hasher.update(sent.text)
        return hasher()

    def clear(self):
        """Clear the stash cashe."""
        self.stash.clear()

    def __call__(self, doc: AmrFeatureDocument):
        """Return the coreferences of the AMR sentences of the document.  If
        the document is cashed in :obj:`stash` use that.  Otherwise use the
        model to compute it and return it.

        :param doc: the document used in the model to perform coreference
                    resolution

        :return: the coreferences tuples as ``(<document index>, <variable>)``

        """
        ref: Dict[str, List[Tuple[int, str]]]
        key: str = self._create_key(doc)
        ref: Union[AmrFailure, Dict[str, List[Tuple[int, str]]]] = \
            self.stash.load(key)
        if ref is None:
            try:
                ref = self._resolve(doc)
            except Exception as e:
                text: str = tw.shorten(doc.text, width=60)
                ref = AmrFailure(
                    exception=e,
                    message=f'could not co-reference <{text}>',
                    sent=doc.text)
                logger.warning(str(ref))
            self.stash.dump(key, ref)
        if not isinstance(ref, AmrFailure):
            doc.coreference_relations = tuple(map(tuple, ref.values()))
