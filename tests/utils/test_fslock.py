import os
import time
from threading import Thread

from fsspec.implementations.local import LocalFileSystem

from mlem.utils.fslock import LOCK_EXT, FSLock
from mlem.utils.path import make_posix

NAME = "testlock"


# pylint: disable=protected-access
def test_fslock(tmpdir):
    fs = LocalFileSystem()
    lock = FSLock(fs, tmpdir, NAME)

    with lock:
        assert lock._timestamp is not None
        assert lock._salt is not None
        lock_path = make_posix(
            os.path.join(
                tmpdir, f"{NAME}.{lock._timestamp}.{lock._salt}.{LOCK_EXT}"
            )
        )
        assert lock.lock_path == lock_path
        assert fs.exists(lock_path)

    assert lock._timestamp is None
    assert lock._salt is None
    assert not fs.exists(lock_path)


def _work(dirname, num):
    time.sleep(0.3 + num / 5)
    with FSLock(LocalFileSystem(), dirname, NAME, salt=num):
        path = os.path.join(dirname, NAME)
        if os.path.exists(path):
            with open(path, "r+", encoding="utf8") as f:
                data = f.read()
        else:
            data = ""
        time.sleep(0.05)
        with open(path, "w", encoding="utf8") as f:
            f.write(data + f"{num}\n")


def test_fslock_concurrent(tmpdir):
    start = 0
    end = 10
    threads = [
        Thread(target=_work, args=(tmpdir, n)) for n in range(start, end)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(os.path.join(tmpdir, NAME), encoding="utf8") as f:
        data = f.read()

    assert data.splitlines() == [str(i) for i in range(start, end)]
    assert os.listdir(tmpdir) == [NAME]
