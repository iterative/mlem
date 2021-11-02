from typing import Type

from pydantic import Field, parse_obj_as
from pydantic.fields import FieldInfo


def lazy_field(
    type_: Type, alias: str, alias_cache: str, *args, **field_kwargs
):

    field_info: FieldInfo = Field(*args, alias=alias, **field_kwargs)

    @property
    def getter(self):
        value = getattr(self, alias_cache)
        if not isinstance(value, type_):
            value = parse_obj_as(type_, value)
            setattr(self, alias_cache, value)
        return value

    @property
    def getter_raw(self):
        value = getattr(self, alias_cache)
        if isinstance(value, type_):
            value = value.dict()
        return value

    return getter, getter_raw, field_info
