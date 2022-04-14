"""Exceptions raised by the MLEM."""
from mlem.constants import MLEM_DIR


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


class MlemRootNotFound(MlemError):
    _message = "{MLEM_DIR} folder wasn't found when searching through the path. Search has started from here: path={path}, fs={fs}, rev={rev}"

    def __init__(self, path, fs=None, rev=None) -> None:

        self.path = path
        self.fs = fs
        self.rev = rev
        self.message = self._message.format(
            MLEM_DIR=MLEM_DIR, path=path, fs=fs, rev=rev
        )
        super().__init__(self.message)


class LocationNotFound(MlemError):
    """Thrown if MLEM could not resolve location"""


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
    """Thrown if model or dataset value is not loaded"""


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


class DeploymentError(MlemError):
    """Thrown if something goes wrong during deployment process"""
