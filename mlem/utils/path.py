from typing import Optional


def make_posix(path: Optional[str]):
    """Turn windows path into posix"""
    if not path:
        return path
    return path.replace("\\", "/")
