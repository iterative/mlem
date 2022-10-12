import re


def is_long_sha(sha: str):
    return re.match(r"^[a-f\d]{40}$", sha)
