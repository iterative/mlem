import posixpath
import random
import re
import time
from typing import List, Tuple

from fsspec import AbstractFileSystem

from mlem.utils.path import make_posix

LOCK_EXT = "lock"


class LockTimeoutError(Exception):
    pass


class FSLock:
    def __init__(
        self,
        fs: AbstractFileSystem,
        dirpath: str,
        name: str,
        timeout: float = None,
        retry_timeout: float = 0.1,
        *,
        salt=None,
    ):
        self.fs = fs
        self.dirpath = make_posix(str(dirpath))
        self.name = name
        self.timeout = timeout
        self.retry_timeout = retry_timeout
        self._salt = salt
        self._timestamp = None

    @property
    def salt(self):
        if self._salt is None:
            self._salt = random.randint(10**3, 10**4)
        return self._salt

    @property
    def timestamp(self):
        if self._timestamp is None:
            self._timestamp = time.time_ns()
        return self._timestamp

    @property
    def lock_filename(self):
        return f"{self.name}.{self.timestamp}.{self.salt}.{LOCK_EXT}"

    @property
    def lock_path(self):
        return posixpath.join(self.dirpath, self.lock_filename)

    def _list_locks(self) -> List[Tuple[int, int]]:
        locks = [
            posixpath.basename(make_posix(f))
            for f in self.fs.listdir(self.dirpath, detail=False)
        ]
        locks = [
            f[len(self.name) :]
            for f in locks
            if f.startswith(self.name) and f.endswith(LOCK_EXT)
        ]
        pat = re.compile(rf"\.(\d+)\.(\d+)\.{LOCK_EXT}")
        locks_re = [pat.match(lock) for lock in locks]
        return [
            (int(m.group(1)), int(m.group(2)))
            for m in locks_re
            if m is not None
        ]

    def _double_check(self):
        locks = self._list_locks()
        if not locks:
            return False
        minlock = min(locks)
        c = minlock == (self._timestamp, self._salt)
        return c

    def _write_lockfile(self):
        self.fs.touch(self.lock_path)

    def _clear(self):
        self._timestamp = None
        self._salt = None

    def _delete_lockfile(self):
        try:
            self.fs.delete(self.lock_path)
        except FileNotFoundError:
            pass

    def __enter__(self):
        start = time.time()

        self._write_lockfile()
        time.sleep(self.retry_timeout)

        while not self._double_check():
            if self.timeout is not None and time.time() - start > self.timeout:
                self._delete_lockfile()
                self._clear()
                raise LockTimeoutError(
                    f"Lock aquiring timeouted after {self.timeout}"
                )
            time.sleep(self.retry_timeout)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._delete_lockfile()
        self._clear()
