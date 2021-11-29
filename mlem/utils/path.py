from typing import Optional


def make_posix(path: Optional[str]):
    if not path:
        return path
    return path.replace("\\", "/")
