"""
We import model from this module to check that intermediate reqs (pandas and this module) are not collected
"""
import pandas  # pylint: disable=unused-import # noqa
from model_trainer import model  # pylint: disable=unused-import # noqa
