"""
MLEM is a tool to help you version and deploy your Machine Learning models:
* Serialize any model trained in Python into ready-to-deploy format
* Model lifecycle management using Git and GitOps principles
* Provider-agnostic deployment
"""
import mlem.log  # noqa

from . import api  # noqa
from .config import LOCAL_CONFIG
from .ext import ExtensionLoader
from .version import __version__

if LOCAL_CONFIG.AUTOLOAD_EXTS:
    ExtensionLoader.load_all()

__all__ = ["api", "__version__", "LOCAL_CONFIG"]
