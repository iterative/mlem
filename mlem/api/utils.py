import contextlib
import re
from functools import wraps
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel
from typing_extensions import Literal

from mlem.core.base import MlemABC, build_mlem_object, load_impl_ext
from mlem.core.errors import InvalidArgumentError, MlemObjectNotFound
from mlem.core.metadata import load, load_meta
from mlem.core.objects import MlemData, MlemModel, MlemObject


def get_data_value(data: Any, batch_size: Optional[int] = None) -> Any:
    if isinstance(data, str):
        return load(data, batch_size=batch_size)
    if isinstance(data, MlemData):
        # TODO: https://github.com/iterative/mlem/issues/29
        #  fix discrepancies between model and data meta objects
        if data.data_type is None or data.data_type.data is None:
            if batch_size:
                return data.read_batch(batch_size)
            data.load_value()
        return data.data

    # TODO: https://github.com/iterative/mlem/issues/29
    #  should we check whether this data is parseable by MLEM?
    #  I guess not cause one may have a model with data input of unknown format/type
    return data


def get_model_meta(
    model: Union[str, MlemModel], load_value: bool = True
) -> MlemModel:
    if isinstance(model, MlemModel):
        if load_value and model.get_value() is None:
            model.load_value()
        return model
    if isinstance(model, str):
        model = load_meta(model, force_type=MlemModel)
        if load_value:
            model.load_value()
        return model
    raise InvalidArgumentError(
        f"The object {model} is neither MlemModel nor path to it"
    )


MM = TypeVar("MM", bound=MlemObject)


@overload
def ensure_meta(
    as_class: Type[MM],
    obj_or_path: Union[str, MM],
    allow_typename: bool = False,
) -> Union[MM, Type[MM]]:
    pass


@overload
def ensure_meta(
    as_class: Type[MM],
    obj_or_path: Union[str, MM],
    allow_typename: Literal[False] = False,
) -> MM:
    pass


def ensure_meta(
    as_class: Type[MM],
    obj_or_path: Union[str, MM],
    allow_typename: bool = False,
) -> Union[MM, Type[MM]]:
    if isinstance(obj_or_path, str):
        try:
            return load_meta(obj_or_path, force_type=as_class)
        except MlemObjectNotFound:
            if allow_typename:
                impl = load_impl_ext(
                    as_class.abs_name, obj_or_path, raise_on_missing=False
                )
                if impl is None or not issubclass(impl, as_class):
                    raise
                return impl
            raise
    if isinstance(obj_or_path, as_class):
        return obj_or_path
    raise ValueError(f"Cannot get {as_class} from '{obj_or_path}'")


MO = TypeVar("MO", bound=MlemABC)


def ensure_mlem_object(
    as_class: Type[MO], obj: Union[str, MO], **kwargs
) -> MO:
    if isinstance(obj, str):
        return build_mlem_object(as_class, obj, conf=kwargs)
    if isinstance(obj, as_class):
        return obj
    raise ValueError(f"Cannot create {as_class} from '{obj}'")


def parse_import_type_modifier(type_: str) -> Tuple[str, Optional[str]]:
    """If the same object can be imported from different types of files,
    modifier helps to specify which format do you want to use
    like this: pandas[csv] or pandas[json]
    """
    match = re.match(r"(\w*)\[(\w*)]", type_)
    if not match:
        return type_, None
    return match.group(1), match.group(2)


class ApiEvent(BaseModel):
    name: str
    error: Optional[str] = None
    kwargs: Dict[str, Any] = {}


_current_event: Optional[ApiEvent] = None


def set_api_event_param(key: str, value):
    if _current_event:
        _current_event.kwargs[key] = value


@contextlib.contextmanager
def api_event_scope(name: str) -> Iterator[ApiEvent]:
    global _current_event  # pylint: disable=global-statement
    event = ApiEvent(name=name)
    tmp = _current_event
    _current_event = event
    try:
        yield event
    finally:
        _current_event = tmp


def api_telemetry(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with api_event_scope(f.__name__) as event:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                event.error = e.__class__.__name__
                raise
            finally:
                send_api_event(event.name, event.error, **event.kwargs)

    return inner


def send_api_event(name: str, error: str = None, **kwargs):
    from mlem.cli.utils import is_cli
    from mlem.telemetry import telemetry

    if not is_cli():
        telemetry.send_event("api", name, error, **kwargs)
