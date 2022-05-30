import re
from typing import Any, Optional, Tuple, Type, TypeVar, Union

from mlem.core.base import MlemABC, build_mlem_object
from mlem.core.errors import InvalidArgumentError
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


def get_model_meta(model: Union[str, MlemModel]) -> MlemModel:
    if isinstance(model, MlemModel):
        if model.get_value() is None:
            model.load_value()
        return model
    if isinstance(model, str):
        model = load_meta(model, force_type=MlemModel)
        model.load_value()
        return model
    raise InvalidArgumentError(
        f"The object {model} is neither MlemModel nor path to it"
    )


MM = TypeVar("MM", bound=MlemObject)


def ensure_meta(as_class: Type[MM], obj_or_path: Union[str, MM]) -> MM:
    if isinstance(obj_or_path, str):
        return load_meta(obj_or_path, force_type=as_class)
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
