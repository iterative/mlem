from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Set, Type, Union

from pydantic import BaseConfig, parse_obj_as
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
        """Manipulate config to support `exclude` and set `__is_root__` attribute"""
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
        type_root: bool = False  # actual default is kinda False via metaclass TODO: fix this strange logic
        type_field: str = "type"
        default_type: Optional[str] = None

    if TYPE_CHECKING:
        __config__: ClassVar[Config, BaseConfig]

    @classmethod
    def validate(cls, value):
        """Polymorphic magic goes here"""
        if isinstance(value, cls):
            return value
        if not cls.__is_root__:
            return super().validate(value)
        value = value.copy()
        type_name = value.pop(
            cls.__config__.type_field, cls.__config__.default_type
        )

        if type_name is None:
            raise ValueError(
                f"Type field was not provided and no default type specified in {cls.__parent__.__name__}"
            )
        child_cls = cls.__resolve_subtype__(type_name)
        if child_cls is cls:
            return super().validate(value)
        return parse_obj_as(child_cls, value)

    @classmethod
    def __resolve_subtype__(cls, type_name: str) -> Type["PolyModel"]:
        return cls.__type_map__[type_name]

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
        if self.__is_root__:
            alias = self.__get_alias__(self.__config__.type_field)
            if not exclude_defaults or alias != self.__config__.default_type:
                yield self.__config__.type_field, alias

        for parent in self.__iter_parents__(include_top=False):
            alias = parent.__get_alias__()
            if not exclude_defaults or alias != parent.__config__.default_type:
                yield parent.__type_field__(), alias

    def __iter__(self):
        """Add alias field"""
        yield from super().__iter__()
        for parent in self.__iter_parents__(include_top=False):
            yield parent.__type_field__(), parent.__get_alias__()

    @classmethod
    def __iter_parents__(cls, include_top=True):
        if cls.__parent__ is None:
            return
        if include_top or not cls.__is_top_root__():
            yield cls
        if not cls.__is_top_root__():
            yield from cls.__parent__.__iter_parents__(include_top=include_top)

    @classmethod
    def __type_field__(cls):
        return cls.__parent__.__config__.type_field

    @classmethod
    def __is_top_root__(cls):
        return cls.__parent__ is cls

    @classmethod
    def __get_alias__(cls, type_field: str = None):
        return cls.__dict__.get(
            type_field or cls.__type_field__(),
            f"{cls.__module__}.{cls.__name__}",
        )

    def __init_subclass__(cls: Type["PolyModel"]):
        """Register subtypes to __type_map__"""
        if cls is PolyModel:
            super(PolyModel, cls).__init_subclass__()
            return
        if cls.__is_root__:  # Parent model initialization
            type_field = cls.__config__.type_field
            cls.__type_map__ = {}
            alias = cls.__get_alias__(type_field)
            if alias is not None and alias is not ...:
                cls.__type_map__[alias] = cls
            cls.__annotations__[  # pylint: disable=no-member
                type_field
            ] = ClassVar[str]
            cls.__class_vars__.add(type_field)
            cls.__parent__ = cls

        parents = [
            p
            for p in cls.mro()[1:]
            if issubclass(p, PolyModel)
            and p.__is_root__  # pylint: disable=no-member
        ]
        if len(parents) > 0:
            parent: Type[PolyModel] = parents[0]
            cls.__parent__ = parent
            alias = cls.__get_alias__()
            if alias is not None and alias is not ...:
                parent.__type_map__[alias] = cls
                setattr(cls, parent.__config__.type_field, alias)

        super(PolyModel, cls).__init_subclass__()
