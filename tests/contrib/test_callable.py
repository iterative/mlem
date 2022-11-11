import posixpath
from typing import Any, ClassVar, Dict, Optional

import pytest
from sklearn.linear_model import LinearRegression

from mlem.contrib.callable import CallableModelType
from mlem.core.artifacts import LOCAL_STORAGE, Artifacts, Storage
from mlem.core.data_type import PrimitiveType
from mlem.core.model import (
    Argument,
    ModelAnalyzer,
    ModelHook,
    ModelIO,
    ModelType,
    Signature,
)


def clbl_model(some_argname):
    return some_argname


class ModelClass:
    def predict_method(self, some_argname):
        return some_argname


@pytest.mark.parametrize("model", [clbl_model, ModelClass().predict_method])
def test_callable_analyze(model, tmpdir):
    mt = ModelAnalyzer.analyze(model, 1)
    assert isinstance(mt, CallableModelType)
    returns = PrimitiveType(ptype="int")

    signature = mt.methods["__call__"]
    assert signature.name == "__call__"
    assert signature.args[0] == Argument(name="some_argname", type_=returns)
    assert signature.returns == returns

    assert mt.model(1) == 1

    artifacts = mt.dump(LOCAL_STORAGE, str(tmpdir / "model"))

    mt.unbind()
    with pytest.raises(ValueError):
        mt.call_method("__call__", 1)

    mt.load(artifacts)
    assert mt.call_method("__call__", 1) == 1


class SklearnWrappedModel:
    def __init__(self):
        self.model = LinearRegression()
        self.model.fit([[1], [2]], [1, 2])

    def run_predict(self, data):
        return self.model.predict(data)


def test_complex_pickle_loading_simple(tmpdir):
    model = SklearnWrappedModel()
    mt = ModelAnalyzer.analyze(model.run_predict, [[1]])
    assert isinstance(mt, CallableModelType)
    mt.model([[1]])
    artifacts = mt.dump(LOCAL_STORAGE, str(tmpdir / "model"))
    assert len(artifacts) == 1
    mt.unbind()
    with pytest.raises(ValueError):
        mt.call_method("__call__", [[1]])
    mt.load(artifacts)
    assert mt.model is not model.run_predict
    mt.call_method("__call__", [[1]])


class ComplexModelIO(ModelIO):
    type: ClassVar = "complex_test_model_io"
    fname: ClassVar = "myfname.ext"

    def dump(self, storage: Storage, path, model: "ComplexModel") -> Artifacts:
        with storage.open(posixpath.join(path, self.fname)) as (f, art):
            f.write(model.prefix.encode("utf8"))
            return {self.fname: art}

    def load(self, artifacts: Artifacts):
        with artifacts[self.fname].open() as f:
            return ComplexModel(prefix=f.read().decode("utf8"))


class ComplexModel:
    def __init__(self, prefix: str):
        self.prefix = prefix


class ComplexModelType(ModelType, ModelHook):
    type: ClassVar = "complex_test_model"
    io: ModelIO = ComplexModelIO()
    methods: Dict[str, Signature] = {}
    valid_types: ClassVar = (ComplexModel,)

    @classmethod
    def process(
        cls, obj: ComplexModel, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        return ComplexModelType().bind(obj)

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, ComplexModel)


class ComplexWrappedModel:
    def __init__(self, prefix):
        self.model = ComplexModel(prefix)

    def run_predict(self, data):
        return self.model.prefix + data


def test_complex_pickle_loading(tmpdir):
    model = ComplexWrappedModel("a")
    mt = ModelAnalyzer.analyze(model.run_predict, "b")
    assert isinstance(mt, CallableModelType)
    assert mt.model("b") == "ab"
    artifacts = mt.dump(LOCAL_STORAGE, tmpdir)
    assert len(artifacts) == 3
    mt.unbind()
    with pytest.raises(ValueError):
        mt.call_method("__call__", "b")
    mt.load(artifacts)
    assert mt.model is not model.run_predict
    assert mt.call_method("__call__", "b") == "ab"
