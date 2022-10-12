"""Exceptions raised by the MLEM."""
from typing import List, Optional

from mlem.constants import MLEM_CONFIG_FILE_NAME


class MlemError(Exception):
    """Base class for all MLEM exceptions."""

    def __init__(self, msg, *args):
        assert msg
        self.msg = msg
        super().__init__(msg, *args)


class DeserializationError(MlemError):
    pass


class SerializationError(MlemError):
    pass


class MlemProjectNotFound(MlemError):
    _message = "{MLEM_CONFIG_FILE_NAME} folder wasn't found when searching through the path. Search has started from here: path={path}, fs={fs}, rev={rev}"

    def __init__(self, path, fs=None, rev=None) -> None:

        self.path = path
        self.fs = fs
        self.rev = rev
        self.message = self._message.format(
            MLEM_CONFIG_FILE_NAME=MLEM_CONFIG_FILE_NAME,
            path=path,
            fs=fs,
            rev=rev,
        )
        super().__init__(self.message)


class LocationNotFound(MlemError):
    """Thrown if MLEM could not resolve location"""


class EndpointNotFound(MlemError):
    """Thrown if MLEM could not resolve endpoint"""


class RevisionNotFound(LocationNotFound):
    _message = "Revision '{rev}' wasn't found in path={path}, fs={fs}"

    def __init__(
        self,
        rev,
        path,
        fs=None,
    ) -> None:

        self.path = path
        self.fs = fs
        self.rev = rev
        self.message = self._message.format(path=path, fs=fs, rev=rev)
        super().__init__(self.message)


class FileNotFoundOnImportError(FileNotFoundError, MlemError):
    """Thrown if import failed because nothing was found at provided location"""


class InvalidArgumentError(ValueError, MlemError):
    """Thrown if arguments are invalid."""


class MlemObjectNotSavedError(ValueError, MlemError):
    """Thrown if we can't do something before we save MLEM object"""


class MlemObjectNotLoadedError(ValueError, MlemError):
    """Thrown if model or data value is not loaded"""


class UnsupportedDataBatchLoadingType(ValueError, MlemError):
    """Thrown if batch loading of data with unsupported file type is called"""

    _message = "Batch-loading data of type '{data_type}' is currently not supported. Please remove batch parameter."

    def __init__(
        self,
        data_type,
    ) -> None:

        self.data_type = data_type
        self.message = self._message.format(data_type=data_type)
        super().__init__(self.message)


class UnsupportedDataBatchLoading(MlemError):
    """Thrown if batch loading of data is called for import workflow"""


class WrongMethodError(ValueError, MlemError):
    """Thrown if wrong method name for model is provided"""


class MlemObjectNotFound(FileNotFoundError, MlemError):
    """Thrown if we can't find MLEM object"""


class ObjectExistsError(ValueError, MlemError):
    """Thrown if we attempt to write object, but something already exists in the given path"""


class HookNotFound(MlemError):
    """Thrown if object does not have suitable hook"""


class MultipleHooksFound(MlemError):
    """Thrown if more than one hook found for object"""


class WrongMetaType(TypeError, MlemError):
    def __init__(self, meta, force_type):
        loc = f"from {meta.loc.uri} " if meta.is_saved else ""
        super().__init__(
            f"Wrong type of meta loaded, got {meta.object_type} {loc}instead of {force_type.object_type}"
        )


class WrongMetaSubType(TypeError, MlemError):
    def __init__(self, meta, force_type):
        loc = f"from {meta.loc.uri} " if meta.is_saved else ""
        super().__init__(
            f"Wrong type of meta loaded, got {meta.object_type} {meta.type} {loc}instead of {force_type.object_type} {force_type.type}"
        )


class WrongABCType(TypeError, MlemError):
    def __init__(self, instance, expected_abc_type):
        super().__init__(
            f"Wrong implementation type, got {instance.type} instead of {expected_abc_type.type}"
        )


class DeploymentError(MlemError):
    """Thrown if something goes wrong during deployment process"""


class WrongRequirementsError(MlemError):
    def __init__(self, wrong, missing, fix):
        self.wrong = wrong
        self.missing = f"\nMissing packages: {missing}." if missing else ""
        self.fix = fix

        super().__init__(
            f"Wrong requirements: {self.wrong} {self.missing}\nTo fix it, run `{fix}`"
        )


class UnknownImplementation(MlemError):
    def __init__(self, type_name: str, abs_name: str):
        self.abs_name = abs_name
        self.type_name = type_name
        super().__init__(f"Unknown {abs_name} implementation: {type_name}")


class UnknownConfigSection(MlemError):
    def __init__(self, section: str):
        self.section = section
        super().__init__(f'Unknown config section "{section}"')


class ExtensionRequirementError(MlemError, ImportError):
    def __init__(self, ext: str, reqs: List[str], extra: Optional[str]):
        self.ext = ext
        self.reqs = reqs
        self.extra = extra
        extra_install = (
            "" if extra is None else f"`pip install mlem[{extra}]` or "
        )
        reqs_str = " ".join(reqs)
        super().__init__(
            f"Extension '{ext}' requires additional dependencies: {extra_install}`pip install {reqs_str}`"
        )
