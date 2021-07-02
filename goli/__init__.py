try:
    import dgl
except ImportError:
    raise ImportError(
        "DGL not installed. Please install it following the official documentation at https://github.com/dmlc/dgl/#installation."
    )


from ._version import __version__

from .config import load_config

from . import utils
from . import features
from . import data
from . import nn
from . import trainer
