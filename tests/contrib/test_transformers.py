from functools import partial

import numpy as np
import pytest
import tensorflow as tf
import torch
from pydantic import parse_obj_as
from transformers import (
    AlbertModel,
    AlbertTokenizer,
    BatchEncoding,
    DistilBertModel,
    DistilBertTokenizer,
    TensorType,
)

from mlem.contrib.transformers import ADDITIONAL_DEPS, BatchEncodingType
from mlem.core.data_type import DataAnalyzer, DataType
from tests.conftest import data_write_read_check

FULL_TESTS = True

TOKENIZERS = {
    AlbertTokenizer: "albert-base-v2",
    DistilBertTokenizer: "distilbert-base-uncased",
}

MODELS = {
    AlbertModel: "albert-base-v2",
    DistilBertModel: "distilbert-base-uncased",
}

ONE_MODEL = AlbertModel
ONE_TOKENIZER = AlbertTokenizer

for_model = pytest.mark.parametrize(
    "model",
    [ONE_MODEL.from_pretrained(MODELS[ONE_MODEL])]
    if not FULL_TESTS
    else [m.from_pretrained(v) for m, v in MODELS.items()],
)

for_tokenizer = pytest.mark.parametrize(
    "tokenizer",
    [ONE_TOKENIZER.from_pretrained(TOKENIZERS[ONE_TOKENIZER])]
    if not FULL_TESTS
    else [m.from_pretrained(v) for m, v in TOKENIZERS.items()],
)


def test_analyzing_model():
    pass


def test_analyzing_tokenizer():
    pass


def test_serving_model():
    pass


def test_serving_tokenizer():
    pass


def test_model_reqs():
    pass


def test_tokenizer_reqs():
    pass


# pylint: disable=protected-access
@for_tokenizer
@pytest.mark.parametrize(
    "return_tensors,typename,eq",
    [
        ("pt", "TorchTensor", lambda a, b: torch.all(a.eq(b))),
        ("tf", "TFTensor", lambda a, b: tf.equal(a, b)._numpy().all()),
        ("np", "NumpyNdarray", lambda a, b: np.equal(a, b).all()),
        (None, "Array", None),
    ],
)
def test_batch_encoding(tokenizer, return_tensors, typename, eq):
    data = tokenizer("aaa bbb", return_tensors=return_tensors)

    data_type = DataAnalyzer.analyze(data)
    assert isinstance(data_type, BatchEncodingType)
    expected_reqs = ["transformers"]
    if return_tensors is not None:
        expected_reqs += [ADDITIONAL_DEPS[TensorType(return_tensors)]]
    assert data_type.get_requirements().modules == expected_reqs

    item_type = DataAnalyzer.analyze(data["input_ids"], is_dynamic=True).dict()
    expected_payload = {
        "item_types": {
            "attention_mask": item_type,
            "input_ids": item_type,
            "token_type_ids": item_type,
        },
        "type": "batch_encoding",
    }
    if return_tensors is not None:
        expected_payload["return_tensors"] = return_tensors
    if "token_type_ids" not in data:
        del expected_payload["item_types"]["token_type_ids"]
    assert data_type.dict() == expected_payload
    data_type2 = parse_obj_as(DataType, data_type.dict())
    assert data_type2 == data_type

    assert data_type.get_model().__name__ == data_type2.get_model().__name__
    schema_item_type = {"items": {"type": "integer"}, "type": "array"}
    if return_tensors is None:
        schema_item_type = {"type": "integer"}
    expected_schema = {
        "definitions": {
            f"attention_mask_{typename}": {
                "items": schema_item_type,
                "title": f"attention_mask_{typename}",
                "type": "array",
            },
            f"input_ids_{typename}": {
                "items": schema_item_type,
                "title": f"input_ids_{typename}",
                "type": "array",
            },
            f"token_type_ids_{typename}": {
                "items": schema_item_type,
                "title": f"token_type_ids_{typename}",
                "type": "array",
            },
        },
        "properties": {
            "attention_mask": {
                "$ref": f"#/definitions/attention_mask_{typename}"
            },
            "input_ids": {"$ref": f"#/definitions/input_ids_{typename}"},
            "token_type_ids": {
                "$ref": f"#/definitions/token_type_ids_{typename}"
            },
        },
        "required": ["input_ids", "token_type_ids", "attention_mask"],
        "title": "DictType",
        "type": "object",
    }
    if "token_type_ids" not in data:
        del expected_schema["definitions"][f"token_type_ids_{typename}"]
        del expected_schema["properties"]["token_type_ids"]
        expected_schema["required"].remove("token_type_ids")
    assert data_type.get_model().schema() == expected_schema
    n_payload = data_type.get_serializer().serialize(data)
    deser = data_type.get_serializer().deserialize(n_payload)
    assert _batch_encoding_equals(data, deser, eq)
    parse_obj_as(data_type.get_model(), n_payload)

    data_type = data_type.bind(data)
    data_write_read_check(
        data_type, custom_eq=partial(_batch_encoding_equals, equals=eq)
    )


def _batch_encoding_equals(first, second, equals):
    assert isinstance(first, BatchEncoding)
    assert isinstance(second, BatchEncoding)

    assert first.keys() == second.keys()

    for key in first:
        if equals is not None:
            assert equals(first[key], second[key])
        else:
            assert first[key] == second[key]
    return True
