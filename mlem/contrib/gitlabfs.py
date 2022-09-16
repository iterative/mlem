"""Gitlab URI support
Extension type: uri

Implementation of `GitlabFileSystem` and `GitlabResolver`
"""
import posixpath
from typing import ClassVar, Optional
from urllib.parse import quote_plus, urlparse, urlsplit

import gitlab
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.registry import known_implementations
from gitlab import Gitlab, GitlabGetError

from mlem.core.meta_io import CloudGitResolver

GL_TYPES = {"blob": "file", "tree": "directory"}


class GitlabFileSystem(AbstractFileSystem):  # pylint: disable=abstract-method
    """Interface to files in githlab"""

    url = "https://gitlab.com/api/projects/{project_id}/repository/tree"
    protocol = "gitlab"

    def __init__(
        self, project_id, sha=None, username=None, token=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.project_id = project_id
        if (username is None) ^ (token is None):
            raise ValueError("Auth required both username and token")
        self.username = username
        self.token = token
        self.gl = Gitlab()
        self.project = self.gl.projects.get(self.project_id)
        if sha is None:
            default_branch = [
                b.name for b in self.project.branches.list() if b.default
            ]
            if len(default_branch) == 0:
                raise ValueError("No sha provided and no default branch")
            sha = default_branch[0]

        self.root = sha
        self.ls("")

    def ls(self, path, detail=False, sha=None, **kwargs):
        """List files at given path

        Parameters
        ----------
        path: str
            Location to list, relative to repo root
        detail: bool
            If True, returns list of dicts, one per file; if False, returns
            list of full filenames only
        sha: str (optional)
            List at the given point in the repo history, branch or tag name or commit
            SHA
        """
        path = self._strip_protocol(path)
        if path not in self.dircache or sha not in [self.root, None]:
            try:
                r = self.project.repository_tree(
                    path=path, ref=sha or self.root
                )
            except GitlabGetError as e:
                if e.response_code == 404:
                    raise FileNotFoundError() from e
                raise
            out = [
                {
                    "name": f["path"],
                    "mode": f["mode"],
                    "type": GL_TYPES[f["type"]],
                    "size": f.get("size", 0),
                    "sha": sha,
                }
                for f in r
            ]
            if sha in [self.root, None]:
                self.dircache[path] = out
        else:
            out = self.dircache[path]

        if detail:
            return out
        return sorted([f["name"] for f in out])

    def invalidate_cache(self, path=None):
        self.dircache.clear()

    @classmethod
    def _strip_protocol(cls, path):
        if "@" in path:
            return cls._get_kwargs_from_urls(path)["path"]
        return super()._strip_protocol(path)

    @classmethod
    def _get_kwargs_from_urls(cls, path):
        parsed_path = urlsplit(path)
        protocol = parsed_path.scheme
        if protocol != "gitlab":
            return {"path": path}
        project_id, path = super()._strip_protocol(path).split("@", maxsplit=2)
        sha, path = _mathch_path_with_ref(project_id, path)
        return {
            "project_id": project_id,
            "path": path,
            "sha": sha,
            "protocol": protocol,
        }

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        sha=None,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError
        return MemoryFile(
            None,
            None,
            self.project.files.get(path, ref=sha or self.root).decode(),
        )


def ls_gitlab_refs(project_id):
    gl = Gitlab()
    project = gl.projects.get(project_id)
    return [b.name for b in project.branches.list()]


def _mathch_path_with_ref(project_id, path):
    path = path.split("/")
    sha = path[0]
    refs = ls_gitlab_refs(project_id)
    # refs.update(ls_github_tags(org, repo))
    branches = {quote_plus(k) for k in refs}
    # match beginning of path with one of existing branches
    # "" is hack for cases with empty path (like 'github.com/org/rep/tree/branch/')
    for i, part in enumerate(path[1:] + [""], start=1):
        if sha in branches:
            path = path[i:]
            break
        sha = f"{sha}%2F{part}"
    else:
        raise ValueError(f'Could not resolve branch from path "{path}"')
    return sha, posixpath.join(*path)


known_implementations["gitlab"] = {
    "class": f"{GitlabFileSystem.__module__}.{GitlabFileSystem.__name__}"
}


class GitlabResolver(CloudGitResolver):
    """Resolve https://gitlab.com URIs"""

    type: ClassVar = "gitlab"
    FS: ClassVar = GitlabFileSystem
    PROTOCOL: ClassVar = "gitlab"
    GITLAB_COM: ClassVar = "https://gitlab.com"

    # TODO: https://github.com//issues/388
    PREFIXES: ClassVar = [GITLAB_COM, PROTOCOL + "://"]
    versioning_support: ClassVar = True

    @classmethod
    def get_kwargs(cls, uri):
        """Parse URI to git repo to get dict with all URI parts"""
        # TODO: do we lose URL to the site, like https://github.com?
        # should be resolved as part of https://github.com/iterative/mlem/issues/4
        sha: Optional[str]
        parsed = urlparse(uri)
        project_id, *path = parsed.path.strip("/").split("/-/blob/")
        if not path:
            return {"project_id": project_id, "path": ""}
        sha, path = _mathch_path_with_ref(project_id, path[0])
        return {"project_id": project_id, "sha": sha, "path": path}

    @classmethod
    def check_rev(cls, options):
        gl = gitlab.Gitlab()
        try:
            gl.projects.get(options["project_id"]).branches.get(options["sha"])
            return True
        except GitlabGetError:
            return False

    @classmethod
    def get_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GitlabFileSystem,
    ):
        fullpath = posixpath.join(project or "", path)
        return (
            f"https://gitlab.com/{fs.project_id}/-/blob/{fs.root}/{fullpath}"
        )

    @classmethod
    def get_project_uri(  # pylint: disable=unused-argument
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GitlabFileSystem,
        uri: str,
    ):
        return f"https://gitlab.com/{fs.project_id}/-/tree/{fs.root}/{project or ''}"
