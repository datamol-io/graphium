from importlib.metadata import version
from importlib.metadata import PackageNotFoundError

try:
    __version__ = version("graphium")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"
