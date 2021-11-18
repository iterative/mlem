from typing import Any, Type

from pydantic import BaseModel, Field, parse_obj_as
from pydantic.fields import FieldInfo


class LazyModel(BaseModel):
    def __setattr__(self, name, value):
        """
        To be able to use properties with setters
        """
        prop = getattr(self.__class__, name, None)
        if (
            prop is not None
            and isinstance(prop, property)
            and prop.fset is not None
        ):
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)


def lazy_field(
    type_: Type,
    alias: str,
    alias_cache: str,
    *args,
    parse_as_type: Any = None,
    **field_kwargs
):
    field_info: FieldInfo = Field(*args, alias=alias, **field_kwargs)

    def getter(self):
        value = getattr(self, alias_cache)
        if not isinstance(value, type_):
            value = parse_obj_as(parse_as_type or type_, value)
            setattr(self, alias_cache, value)
        return value

    def setter(self, value):
        setattr(self, alias_cache, value)

    def getter_raw(self):
        value = getattr(self, alias_cache)
        if isinstance(value, type_):
            value = value.dict()
        return value

    return (
        property(fget=getter, fset=setter),
        property(fget=getter_raw, fset=setter),
        field_info,
    )
