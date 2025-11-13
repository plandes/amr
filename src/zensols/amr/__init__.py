def suppress_warnings():
    """New warnings starting with Python 3.12 in :mod:`amrlib`."""
    import warnings
    warnings.filterwarnings(
        'ignore',
        message=r"invalid escape sequence '\\d'",
        category=SyntaxWarning)
    warnings.filterwarnings(
        'ignore',
        message=("'pin_memory' argument is set as true but not supported "
                 "on MPS now, device pinned memory won't be used."),
        category=UserWarning)


suppress_warnings()


from .domain import *
from .sent import *
from .doc import *
from .container import *
from .app import *
from .cli import *
