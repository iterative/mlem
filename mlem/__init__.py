from . import api  # noqa
from .config import CONFIG
from .ext import ExtensionLoader
from .version import __version__

if CONFIG.AUTOLOAD_EXTS:
    ExtensionLoader.load_all()

__all__ = ["api", "__version__"]
