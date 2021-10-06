from functools import wraps
from typing import AbstractSet, ClassVar, Dict, Optional, Set, Type

from pydantic import BaseModel


class PolyModel(BaseModel):
    """Base class that enable polymorphism in pydantic models

    Attributes:
        __type_root__: set to True for root of your hierarchy (parent model)
        __type_map__: mapping subclass alias -> subclass (fills up automatically when you subclass parent model)
        __type_field__: name of the field containing subclass alias in serialized model
        __transient_fields__: set of fields that will not be serialized (you must annotate them as ClassVar)
        __inherit_transient_fields__: whether to inherit __transient_fields__ from parents
    """

    __type_root__: ClassVar[bool] = True
    __type_map__: ClassVar[Dict[str, Type]] = {}
    __type_field__: ClassVar[str] = "type"
    __transient_fields__: ClassVar[Set[str]] = set()
    __inherit_transient_fields__: ClassVar[bool] = True
    parent: ClassVar[Optional[Type["PolyModel"]]] = None

    @classmethod
    def __is_root__(cls):
        return cls.__dict__.get("__type_root__", False)

    @classmethod
    def validate(cls, value):
        """Polymorphic magic goes here"""
        if isinstance(value, cls):
            return value
        type_name = value.pop(cls.__type_field__)
        child_cls = cls.__type_map__[type_name]
        return child_cls(**value)

    @wraps(BaseModel.dict)
    def dict(self, **kwargs):
        """Add alias field"""
        result = super().dict(**kwargs)
        result[self.__type_field__] = self.__get_alias__()
        return result

    @wraps(BaseModel._calculate_keys)
    def _calculate_keys(self, *args, **kwargs) -> Optional[AbstractSet[str]]:
        """Exclude transient stuff"""
        kwargs["exclude"] = (
            kwargs.get("exclude", None) or set()
        ) | self.__transient_fields__
        return super()._calculate_keys(*args, **kwargs)

    def __iter__(self):
        """Add alias field"""
        yield from super().__iter__()
        yield self.__type_field__, self.__get_alias__()

    @classmethod
    def __get_alias__(cls):
        return cls.__dict__.get(
            cls.__type_field__, f"{cls.__module__}.{cls.__name__}"
        )

    def __setattr__(self, key, value):
        """Allow setting for transient fields"""
        if key in self.__transient_fields__:
            self.__dict__[key] = value
            return
        super().__setattr__(key, value)

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

        if cls.__inherit_transient_fields__:
            cls.__transient_fields__ = {
                f
                for c in cls.mro()
                for f in getattr(c, "__transient_fields__", [])
            }
        super(PolyModel, cls).__init_subclass__()
