try:
    from ._version import __version__
except ImportError:
    # For development, when the package is not installed
    __version__ = "unknown, pkg not installed"
