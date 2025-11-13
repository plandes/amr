def suppress_warnings():
    """New warnings starting with Python 3.12 in :mod:`amrlib`."""
    import warnings
    warnings.filterwarnings(
        'ignore',
        message=r"invalid escape sequence '\\d'",
        category=SyntaxWarning)


suppress_warnings()


from .domain import *
from .sent import *
from .doc import *
from .container import *
from .app import *
from .cli import *
