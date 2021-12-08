"""
This package tests import chains:
test_pack_1.test_model -> test_pack_2
test_pack_1.test_model -> test_pack_1.__init__ -> test_pack_1.test_model_type
"""
from .test_model import TestM  # noqa
from .test_model_type import TestModelType  # noqa
