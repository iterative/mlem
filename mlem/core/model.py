"""
Base classes to work with ML models in MLEM
"""
import inspect
import pickle
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from pydantic import BaseModel

from mlem.core.artifacts import Artifacts, Storage
from mlem.core.base import MlemObject
from mlem.core.dataset_type import (
    DatasetAnalyzer,
    DatasetType,
    UnspecifiedDatasetType,
)
from mlem.core.hooks import Analyzer, Hook
from mlem.core.requirements import Requirements, WithRequirements
from mlem.utils.module import get_object_requirements


class ModelIO(MlemObject):
    """
    IO base class for models
    """

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "model_io"

    @abstractmethod
    def dump(self, storage: Storage, path, model) -> Artifacts:
        """ """
        raise NotImplementedError()

    @abstractmethod
    def load(self, artifacts: Artifacts):
        """
        Must load and return model
        :return: model object
        """
        raise NotImplementedError()


class SimplePickleIO(ModelIO):
    type: ClassVar[str] = "simple_pickle"

    def dump(self, storage: Storage, path: str, model) -> Artifacts:
        with storage.open(path) as (f, art):
            pickle.dump(model, f)
        return [art]

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError("Invalid artifacts: should be one .pkl file")
        with artifacts[0].open() as f:
            return pickle.load(f)


class Argument(BaseModel):
    name: str
    type_: DatasetType
    required: bool = True
    default: Any = None
    kw_only: bool = False

    @classmethod
    def from_argspec(
        cls,
        name: str,
        argspec: inspect.FullArgSpec,
        defaults: Dict[str, Any],
        auto_infer: bool = False,
        **call_kwargs,
    ):
        if name in argspec.annotations and isinstance(
            argspec.annotations[name], DatasetType
        ):
            type_ = argspec.annotations[name]
        elif auto_infer:
            if name not in call_kwargs and name not in defaults:
                raise TypeError(
                    f"auto_infer=True, but no value for {name} argument"
                )
            type_ = DatasetAnalyzer.analyze(
                defaults.get(name, call_kwargs.get(name))
            )
        else:
            type_ = UnspecifiedDatasetType()
        return Argument(
            name=name,
            type_=type_,
            required=name not in defaults,
            default=defaults.get(name),
        )


def compose_args(
    argspec: inspect.FullArgSpec,
    skip_first: bool = False,
    auto_infer: bool = False,
    **call_kwargs,
):
    args_defaults = dict(
        zip(
            reversed(argspec.args or ()),
            reversed(argspec.defaults or ()),
        )
    )
    args = argspec.args
    if skip_first:
        args = args[1:]
    return [
        Argument.from_argspec(
            name, argspec, args_defaults, auto_infer, **call_kwargs
        )
        for name in args
    ] + [
        Argument.from_argspec(
            name,
            argspec,
            argspec.kwonlydefaults or {},
            auto_infer,
            **call_kwargs,
        )
        for name in argspec.kwonlyargs
    ]


class Signature(BaseModel, WithRequirements):
    name: str
    args: List[Argument]
    returns: DatasetType
    varargs: Optional[str] = None
    varkw: Optional[str] = None

    @classmethod
    def from_method(
        cls, method: Callable, auto_infer: bool = False, **call_kwargs
    ):
        # no support for positional-only args, but who uses them anyway
        argspec = inspect.getfullargspec(method)
        if "return" in argspec.annotations and isinstance(
            argspec.annotations["return"], DatasetType
        ):
            returns = argspec.annotations["return"]
        elif auto_infer:
            result = method(**call_kwargs)
            returns = DatasetAnalyzer.analyze(result)
        else:
            returns = UnspecifiedDatasetType()
        return Signature(
            name=method.__name__,
            args=compose_args(
                argspec,
                skip_first=argspec.args[0] == "self",
                auto_infer=auto_infer,
                **call_kwargs,
            ),
            returns=returns,
            varkw=argspec.varkw,
            varargs=argspec.varargs,
        )

    def has_unspecified_args(self):
        return isinstance(self.returns, UnspecifiedDatasetType) or any(
            isinstance(a.type_, UnspecifiedDatasetType) for a in self.args
        )

    def get_requirements(self):
        return self.returns.get_requirements() + [
            r for a in self.args for r in a.type_.get_requirements().__root__
        ]


T = TypeVar("T", bound="ModelType")


class ModelType(ABC, MlemObject, WithRequirements):
    """
    Base class for dataset type metadata.
    """

    class Config:
        type_root = True
        exclude = {"model"}

    abs_name: ClassVar[str] = "model_type"

    model: Any = None

    io: ModelIO
    methods: Dict[str, Signature]

    def load(self, artifacts: Artifacts):
        self.model = self.io.load(artifacts)

    def dump(self, storage: Storage, path: str):
        return self.io.dump(storage, path, self.model)

    def bind(self: T, model: Any) -> T:
        self.model = model
        return self

    def call_method(self, name, *input_data):
        """
        Calls model method with given name on given input data

        :param name: name of the method to call
        :param input_data: argument for the method
        :return: call result
        """
        self._check_method(name)
        signature = self.methods[name]
        output_data = self._call_method(signature.name, *input_data)
        return output_data

    def _check_method(self, name):
        if self.model is None:
            raise ValueError(f"Model {self} is not loaded")
        if name not in self.methods:
            raise ValueError(f"Model '{self}' doesn't expose method '{name}'")

    def _call_method(self, wrapped: str, *input_data):
        # with switch_curdir(self.curdir):
        if hasattr(self, wrapped):
            return getattr(self, wrapped)(*input_data)
        return getattr(self.model, wrapped)(*input_data)

    def resolve_method(self, method_name: str = None):
        """Checks if method with this name exists

        :param method_name: name of the method.
        If not provided, this model must have only one method and it will be used"""
        if method_name is None:
            if len(self.methods) > 1:
                raise ValueError(
                    f"Please provide one of {list(self.methods.keys())} as method name"
                )
            method_name = next(iter(self.methods))
        self._check_method(method_name)
        return method_name

    def unbind(self):
        self.model = None
        return self

    def get_requirements(self) -> Requirements:
        return Requirements.new(
            [
                r
                for m in self.methods.values()
                for r in m.get_requirements().__root__
            ]
        ) + get_object_requirements(self.model)


class ModelHook(Hook[ModelType], ABC):
    valid_types: ClassVar[Optional[Tuple[Type, ...]]] = None

    @classmethod
    @abstractmethod
    def process(  # pylint: disable=arguments-differ # so what
        cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        raise NotImplementedError


class ModelAnalyzer(Analyzer[ModelType]):
    base_hook_class = ModelHook
    hooks: List[Type[ModelHook]]  # type: ignore

    @classmethod
    def analyze(  # pylint: disable=arguments-differ # so what
        cls, obj, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:
        return super().analyze(obj, sample_data=sample_data, **kwargs)
