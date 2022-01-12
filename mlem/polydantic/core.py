from functools import wraps
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Set, Type, Union

from pydantic import BaseConfig, BaseModel
from pydantic.main import ModelMetaclass

from mlem.polydantic.lazy import LazyModel

if TYPE_CHECKING:
    from pydantic.typing import (
        AbstractSetIntStr,
        MappingIntStrAny,
        TupleGenerator,
    )


class PolyModelMetaclass(ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: B902
        config = namespace.get("Config", None)
        if config is not None:
            namespace["__is_root__"] = config.__dict__.get("type_root", False)

            fields = getattr(config, "fields", {})
            exclude = getattr(config, "exclude", set())
            for exclude_field in exclude:
                field_conf = fields.get(exclude_field, {})
                field_conf["exclude"] = True
                fields[exclude_field] = field_conf
            config.fields = fields
        else:
            namespace["__is_root__"] = False

        return super().__new__(mcs, name, bases, namespace, **kwargs)


class PolyModel(LazyModel, metaclass=PolyModelMetaclass):
    """Base class that enable polymorphism in pydantic models

    Attributes:
        __type_root__: set to True for root of your hierarchy (parent model)
        __type_map__: mapping subclass alias -> subclass (fills up automatically when you subclass parent model)
        __type_field__: name of the field containing subclass alias in serialized model
    """

    __type_map__: ClassVar[Dict[str, Type]] = {}
    __parent__: ClassVar[Optional[Type["PolyModel"]]] = None
    __is_root__: ClassVar[bool]

    class Config:
        """
        Attributes:
            type_root: set to True for root of your hierarchy (parent model)
            type_field: name of the field containing subclass alias in serialized model
            default_type: default type to assume if not type_field is provided
        """

        exclude: Set[str] = set()
        type_root: bool = True
        type_field: str = "type"
        default_type: Optional[str] = None

    if TYPE_CHECKING:
        __config__: ClassVar[Config, BaseConfig]

    @classmethod
    def validate(cls, value):
        """Polymorphic magic goes here"""
        if isinstance(value, cls):
            return value
        value = value.copy()
        type_name = value.pop(
            cls.__config__.type_field, cls.__config__.default_type
        )
        if type_name is None:
            raise ValueError(
                f"Type field was not provided and no default type specified in {cls.__parent__.__name__}"
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
            or alias != self.__config__.default_type
        ):
            result[self.__config__.type_field] = alias
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
        if not exclude_defaults or alias != self.__config__.default_type:
            yield self.__config__.type_field, alias

    def __iter__(self):
        """Add alias field"""
        yield from super().__iter__()
        yield self.__config__.type_field, self.__get_alias__()

    @classmethod
    def __get_alias__(cls):
        return cls.__dict__.get(
            cls.__config__.type_field, f"{cls.__module__}.{cls.__name__}"
        )

    def __init_subclass__(cls: Type["PolyModel"]):
        """Register subtypes to __type_map__"""
        if cls.__is_root__:  # Parent model initialization
            cls.__type_map__ = {}
            cls.__annotations__[  # pylint: disable=no-member
                cls.__config__.type_field
            ] = ClassVar[str]
            cls.__class_vars__.add(cls.__config__.type_field)

        for parent in cls.mro():  # Looking for parent model
            if not issubclass(parent, PolyModel):
                continue
            if parent.__is_root__:  # pylint: disable=no-member
                if parent == PolyModel:
                    break
                alias = cls.__get_alias__()
                if alias is not None and alias is not ...:
                    parent.__type_map__[alias] = cls
                    setattr(cls, cls.__config__.type_field, alias)
                cls.__parent__ = parent
                break
        else:
            raise ValueError("No parent with type__root == True found")

        super(PolyModel, cls).__init_subclass__()
