from functools import wraps
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Set, Type, Union

from pydantic import BaseModel
from pydantic.main import ModelMetaclass

from mlem.polydantic.lazy import LazyModel

if TYPE_CHECKING:
    from pydantic.typing import (
        AbstractSetIntStr,
        MappingIntStrAny,
        TupleGenerator,
    )


class PolyModelMetaclass(ModelMetaclass):
    class Config:
        exclude: Set[str] = set()

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: B902
        config = namespace.get("Config", None)
        if config is not None:
            fields = getattr(config, "fields", {})
            exclude = getattr(config, "exclude", set())
            for exclude_field in exclude:
                field_conf = fields.get(exclude_field, {})
                field_conf["exclude"] = True
                fields[exclude_field] = field_conf
            config.fields = fields
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class PolyModel(LazyModel, metaclass=PolyModelMetaclass):
    """Base class that enable polymorphism in pydantic models

    Attributes:
        __type_root__: set to True for root of your hierarchy (parent model)
        __type_map__: mapping subclass alias -> subclass (fills up automatically when you subclass parent model)
        __type_field__: name of the field containing subclass alias in serialized model
    """

    __type_root__: ClassVar[bool] = True
    __type_map__: ClassVar[Dict[str, Type]] = {}
    __type_field__: ClassVar[str] = "type"
    __default_type__: ClassVar[Optional[str]] = None
    parent: ClassVar[Optional[Type["PolyModel"]]] = None

    @classmethod
    def __is_root__(cls):
        return cls.__dict__.get("__type_root__", False)

    @classmethod
    def validate(cls, value):
        """Polymorphic magic goes here"""
        if isinstance(value, cls):
            return value
        value = value.copy()
        type_name = value.pop(cls.__type_field__, cls.__default_type__)
        if type_name is None:
            raise ValueError(
                f"Type field was not provided and no default type specified in {cls.parent.__name__}"
            )
        child_cls = cls.__type_map__[type_name]
        return child_cls(**value)

    @wraps(BaseModel.dict)
    def dict(self, **kwargs):
        """Add alias field"""
        result = super().dict(**kwargs)
        alias = self.__get_alias__()
        if (
            not kwargs.get("exclude_defaults", False)
            or alias != self.__default_type__
        ):
            result[self.__type_field__] = alias
        return result

    def _iter(
        self,
        to_dict: bool = False,
        by_alias: bool = False,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> "TupleGenerator":
        yield from super()._iter(
            to_dict=to_dict,
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        alias = self.__get_alias__()
        if not exclude_defaults or alias != self.__default_type__:
            yield self.__type_field__, alias

    def __iter__(self):
        """Add alias field"""
        yield from super().__iter__()
        yield self.__type_field__, self.__get_alias__()

    @classmethod
    def __get_alias__(cls):
        return cls.__dict__.get(
            cls.__type_field__, f"{cls.__module__}.{cls.__name__}"
        )

    def __init_subclass__(cls: Type["PolyModel"]):
        """Register subtypes to __type_map__"""
        if cls.__is_root__():  # Parent model initialization
            cls.__type_map__ = {}
            cls.__annotations__[  # pylint: disable=no-member
                cls.__type_field__
            ] = ClassVar[str]
            cls.__class_vars__.add(cls.__type_field__)

        for parent in cls.mro():  # Looking for parent model
            if not issubclass(parent, PolyModel):
                continue
            if parent.__is_root__():
                if parent == PolyModel:
                    break
                alias = cls.__get_alias__()
                if alias is not None and alias is not ...:
                    parent.__type_map__[alias] = cls
                    setattr(cls, cls.__type_field__, alias)
                cls.parent = parent
                break
        else:
            raise ValueError("No parent with __type__root__ == True found")

        super(PolyModel, cls).__init_subclass__()
