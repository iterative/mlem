"""Exceptions raised by the MLEM."""


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
    _message = "{MLEM_DIR} folder wasn't found when searching through the path. Search has started from here: path={path}, fs={fs}"

    def __init__(
        self,
        path,
        fs,
    ) -> None:
        from mlem.utils.root import MLEM_DIR  # circular imports

        self.path = path
        self.fs = fs
        self.message = self._message.format(
            MLEM_DIR=MLEM_DIR, path=path, fs=fs
        )
        super().__init__(self.message)


class InvalidArgumentError(ValueError, MlemError):
    """Thrown if arguments are invalid."""


class MlemObjectNotSavedError(ValueError, MlemError):
    """Thrown if we can't do something before we save MLEM object"""


class ObjectExistsError(ValueError, MlemError):
    """Thrown if we attempt to write object, but something already exists in the given path"""
