"""Github URI support
Extension type: uri

Implementation of `GithubResolver`
"""
import pathlib
import posixpath
from typing import ClassVar, Dict, Optional
from urllib.parse import quote_plus, urlparse

import requests
from fsspec.implementations.github import GithubFileSystem

from mlem.config import LOCAL_CONFIG
from mlem.core.meta_io import CloudGitResolver
from mlem.utils.git import is_long_sha


def ls_branches(repo_url: str) -> Dict[str, str]:
    """List branches in remote git repo"""
    import git

    g = git.cmd.Git()
    remote_refs: Dict[str, str] = dict(
        tuple(reversed(ref.split("\t")[:2]))
        for ref in g.ls_remote(repo_url).split("\n")
    )

    return {"/".join(k.split("/")[2:]): v for k, v in remote_refs.items()}


def ls_github_branches(org: str, repo: str):
    """List branches in github repo"""
    return _ls_github_refs(org, repo, "branches")


def ls_github_tags(org: str, repo: str):
    """List tags in github repo"""
    return _ls_github_refs(org, repo, "tags")


def github_check_rev(org: str, repo: str, rev: str):
    """Check that rev exists in a github repo"""
    res = requests.head(
        f"https://api.github.com/repos/{org}/{repo}/commits/{rev}",
        auth=(LOCAL_CONFIG.GITHUB_USERNAME, LOCAL_CONFIG.GITHUB_TOKEN),  # type: ignore
    )
    return res.status_code == 200


def _ls_github_refs(org: str, repo: str, endpoint: str):
    result = requests.get(
        f"https://api.github.com/repos/{org}/{repo}/{endpoint}",
        auth=(LOCAL_CONFIG.GITHUB_USERNAME, LOCAL_CONFIG.GITHUB_TOKEN),  # type: ignore
    )
    if result.status_code == 200:
        return {b["name"]: b["commit"]["sha"] for b in result.json()}
    result.raise_for_status()
    return None


class GithubResolver(CloudGitResolver):
    """Resolve https://github.com URLs"""

    type: ClassVar = "github"
    FS: ClassVar = GithubFileSystem
    PROTOCOL: ClassVar = "github"
    GITHUB_COM: ClassVar = "https://github.com"

    # TODO: https://github.com//issues/388
    PREFIXES: ClassVar = [GITHUB_COM, PROTOCOL + "://"]
    versioning_support: ClassVar = True

    @classmethod
    def get_envs(cls):
        kwargs = {}
        if LOCAL_CONFIG.GITHUB_TOKEN is not None:
            kwargs["username"] = LOCAL_CONFIG.GITHUB_USERNAME
            kwargs["token"] = LOCAL_CONFIG.GITHUB_TOKEN
        return kwargs

    @classmethod
    def get_kwargs(cls, uri):
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
            if is_long_sha(sha):
                path = path[2:]
            else:
                refs = ls_github_branches(org, repo)
                refs.update(ls_github_tags(org, repo))
                branches = {quote_plus(k) for k in refs}
                # match beginning of path with one of existing branches
                # "" is hack for cases with empty path (like 'github.com/org/rep/tree/branch/')
                for i, part in enumerate(path[2:] + [""], start=2):
                    if sha in branches:
                        path = path[i:]
                        break
                    sha = f"{sha}%2F{part}"
                else:
                    raise ValueError(
                        f'Could not resolve branch from uri "{uri}"'
                    )
        else:
            sha = None
        return {
            "org": org,
            "repo": repo,
            "sha": sha,
            "path": posixpath.join(*path) if path else "",
        }

    @classmethod
    def check_rev(cls, options):
        return github_check_rev(
            options["org"], options["repo"], options["sha"]
        )

    @classmethod
    def get_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GithubFileSystem,
    ):
        fullpath = posixpath.join(project or "", path)
        return (
            f"https://github.com/{fs.org}/{fs.repo}/tree/{fs.root}/{fullpath}"
        )

    @classmethod
    def get_project_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GithubFileSystem,
        uri: str,
    ):
        return f"https://github.com/{fs.org}/{fs.repo}/{project or ''}"
