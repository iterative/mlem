"""
This package tests import chain pkg -> pkg.__init__ -> pgk.impl -> pkg.subpkg -> pkg.subpkg.__init__ -> pkg.subpkg.impl
"""
from .impl import pkg_func  # noqa
