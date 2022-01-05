"""
This package tests import chains:
test_pack_1.test_model -> test_pack_2
test_pack_1.test_model -> test_pack_1.__init__ -> test_pack_1.test_model_type
"""
import numpy  # noqa

from .model import TestM  # noqa
