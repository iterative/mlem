import os
import pathlib
from typing import Dict, Optional
from urllib.parse import quote_plus, urlparse

import git
import requests

from mlem.config import CONFIG


def get_github_kwargs(uri: str):
    """Parse URI to git repo to get dict with all URI parts"""
    # TODO: do we lose URL to the site, like https://github.com?
    # should be resolved as part of https://github.com/iterative/mlem/issues/4
    sha: Optional[str]
    parsed = urlparse(uri)
    parts = pathlib.Path(parsed.path).parts
    org, repo, *path = parts[1:]
    if not path:
        return {"org": org, "repo": repo, "path": ""}
    if path[0] == "tree":
        sha = path[1]
        branches = {quote_plus(k) for k in ls_github_branches(org, repo)}
        # match beginning of path with one of existing branches
        for i, part in enumerate(path[2:], start=2):
            if sha in branches:
                path = path[i:]
                break
            sha = f"{sha}%2F{part}"
        else:
            raise ValueError(f'Could not resolve branch from uri "{uri}"')
    else:
        sha = None
    return {
        "org": org,
        "repo": repo,
        "sha": sha,
        "path": os.path.join(*path),
    }


def get_github_envs() -> Dict:
    """Get authentification envs"""
    kwargs = {}
    if CONFIG.GITHUB_TOKEN is not None:
        kwargs["username"] = CONFIG.GITHUB_USERNAME
        kwargs["token"] = CONFIG.GITHUB_TOKEN
    return kwargs


def ls_branches(repo_url: str) -> Dict[str, str]:
    git.cmd.Git().ls_remote(repo_url)
    g = git.cmd.Git()
    remote_refs: Dict[str, str] = dict(
        tuple(reversed(ref.split("\t")[:2]))
        for ref in g.ls_remote(repo_url).split("\n")
    )

    return {"/".join(k.split("/")[2:]): v for k, v in remote_refs.items()}


def ls_github_branches(org: str, repo: str):
    result = requests.get(
        f"https://api.github.com/repos/{org}/{repo}/branches",
        auth=(CONFIG.GITHUB_USERNAME, CONFIG.GITHUB_TOKEN),
    )
    if result.status_code == 200:
        return {b["name"]: b["commit"]["sha"] for b in result.json()}
    result.raise_for_status()
    return None
