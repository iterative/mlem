from typing import TYPE_CHECKING, Any, Callable, Optional, Type, Union

from pydantic import BaseModel, Field, parse_obj_as
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from pydantic.typing import AbstractSetIntStr, DictStrAny, MappingIntStrAny


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

    def dict(  # pylint: disable=useless-super-delegation
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = True,
        skip_defaults: bool = None,
        exclude_unset: bool = True,
        exclude_defaults: bool = True,
        exclude_none: bool = False,
    ) -> "DictStrAny":
        # changing defaults
        return super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    def json(  # pylint: disable=useless-super-delegation
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = True,
        skip_defaults: bool = None,
        exclude_unset: bool = True,
        exclude_defaults: bool = True,
        exclude_none: bool = False,
        encoder: Optional[Callable[[Any], Any]] = None,
        models_as_dict: bool = True,
        **dumps_kwargs: Any,
    ) -> str:
        # changing defaults
        return super().json(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            encoder=encoder,
            models_as_dict=models_as_dict,
            **dumps_kwargs,
        )


def lazy_field(
    type_: Type,
    alias: str,
    alias_cache: str,
    *args,
    parse_as_type: Any = None,
    **field_kwargs,
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
