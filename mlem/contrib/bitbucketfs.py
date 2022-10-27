"""BitBucket URI support
Extension type: uri

Implementation of `BitbucketFileSystem` and `BitbucketResolver`
"""
import posixpath
from typing import ClassVar, Dict, Optional
from urllib.parse import quote_plus, urljoin, urlparse, urlsplit

import requests
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.registry import known_implementations
from pydantic import Field
from requests import HTTPError

from mlem.config import MlemConfigBase
from mlem.core.meta_io import CloudGitResolver
from mlem.utils.git import is_long_sha

BITBUCKET_ORG = "https://bitbucket.org"


class BitbucketWrapper:

    tree_endpoint = "/api/internal/repositories/{repo}/tree/{rev}/{path}"
    repo_endpoint = "/api/2.0/repositories/{repo}"
    refs_endpoint = "/api/2.0/repositories/{repo}/refs"
    file_endpoint = "/api/2.0/repositories/{repo}/src/{rev}/{path}"

    def __init__(
        self, url: str, username: Optional[str], password: Optional[str]
    ):
        self.username = username
        self.password = password
        self.url = url
        self.refs_cache: Dict[str, Dict[str, str]] = {}

    @property
    def auth(self):
        if self.username is not None and self.password is not None:
            return self.username, self.password
        return None

    def tree(self, path: str, repo: str, rev: str):
        rev = self.get_rev_sha(repo, rev)
        r = requests.get(
            urljoin(
                self.url,
                self.tree_endpoint.format(path=path or "", repo=repo, rev=rev),
            ),
            auth=self.auth,
        )
        r.raise_for_status()
        return r.json()[0]["contents"]

    def get_default_branch(self, repo: str):
        r = requests.get(
            urljoin(self.url, self.repo_endpoint.format(repo=repo)),
            auth=self.auth,
        )
        r.raise_for_status()
        return r.json()["mainbranch"]["name"]

    def open(self, path: str, repo: str, rev: str):
        rev = self.get_rev_sha(repo, rev)
        r = requests.get(
            urljoin(
                self.url,
                self.file_endpoint.format(path=path, repo=repo, rev=rev),
            ),
            auth=self.auth,
        )
        r.raise_for_status()
        return r.content

    def _get_refs(self, repo: str) -> Dict[str, str]:
        r = requests.get(
            urljoin(self.url, self.refs_endpoint.format(repo=repo)),
            auth=self.auth,
        )
        r.raise_for_status()
        return {v["name"]: v["target"]["hash"] for v in r.json()["values"]}

    def get_refs(self, repo: str) -> Dict[str, str]:
        if repo not in self.refs_cache:
            self.refs_cache[repo] = self._get_refs(repo)
        return self.refs_cache[repo]

    def invalidate_cache(self):
        self.refs_cache = {}

    def get_rev_sha(self, repo: str, rev: str):
        if is_long_sha(rev):
            return rev
        return self.get_refs(repo).get(rev, rev)

    def check_rev(self, repo: str, rev: str) -> bool:
        r = requests.head(
            urljoin(
                self.url,
                self.file_endpoint.format(path="", repo=repo, rev=rev),
            )
        )
        return r.status_code == 200


class BitbucketConfig(MlemConfigBase):
    class Config:
        section = "bitbucket"

    USERNAME: Optional[str] = Field(default=None, env="BITBUCKET_USERNAME")
    PASSWORD: Optional[str] = Field(default=None, env="BITBUCKET_PASSWORD")


class BitBucketFileSystem(
    AbstractFileSystem
):  # pylint: disable=abstract-method
    def __init__(
        self,
        repo: str,
        sha: str = None,
        host: str = BITBUCKET_ORG,
        username: str = None,
        password: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        conf = BitbucketConfig.local()
        self.password = password or conf.PASSWORD
        self.username = username or conf.USERNAME
        self.repo = repo
        self.host = host

        self.bb = BitbucketWrapper(host, self.username, self.password)
        if sha is None:
            sha = self.bb.get_default_branch(repo)
        self.root = sha
        self.ls("")

    def invalidate_cache(self, path=None):
        super().invalidate_cache(path)
        self.dircache.clear()
        self.bb.invalidate_cache()

    def ls(self, path, detail=False, sha=None, **kwargs):
        path = self._strip_protocol(path)
        if path not in self.dircache or sha not in [self.root, None]:
            try:
                r = self.bb.tree(
                    path=path, repo=self.repo, rev=sha or self.root
                )
            except HTTPError as e:
                if e.response.status_code == 404:
                    raise FileNotFoundError() from e
                raise
            out = [
                {
                    "name": posixpath.join(path, f["name"]),
                    "mode": None,
                    "type": f["type"],
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

    @classmethod
    def _strip_protocol(cls, path):
        if "@" in path:
            return cls._get_kwargs_from_urls(path)["path"]
        return super()._strip_protocol(path)

    @classmethod
    def _get_kwargs_from_urls(cls, path):
        parsed_path = urlsplit(path)
        protocol = parsed_path.scheme
        if protocol != "bitbucket":
            return {"path": path}
        repo, path = super()._strip_protocol(path).split("@", maxsplit=2)
        sha, path = _mathch_path_with_ref(repo, path)
        return {
            "path": path,
            "sha": sha,
            "protocol": protocol,
            "repo": repo,
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
            self.bb.open(path, self.repo, rev=sha or self.root),
        )


known_implementations["bitbucket"] = {
    "class": f"{BitBucketFileSystem.__module__}.{BitBucketFileSystem.__name__}"
}


def ls_bb_refs(repo) -> Dict[str, str]:
    conf = BitbucketConfig.local()
    password = conf.PASSWORD
    username = conf.USERNAME
    return BitbucketWrapper(
        BITBUCKET_ORG, username=username, password=password
    ).get_refs(repo)


def _mathch_path_with_ref(repo, path):
    path = path.split("/")
    sha = path[0]
    refs = ls_bb_refs(repo)
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


class BitBucketResolver(CloudGitResolver):
    """Resolve bitbucket URIs"""

    type: ClassVar = "bitbucket"
    FS = BitBucketFileSystem
    PROTOCOL = "bitbucket"

    # TODO: https://github.com/iterative/mlem/issues/388
    PREFIXES = [BITBUCKET_ORG, PROTOCOL + "://"]
    versioning_support = True

    @classmethod
    def get_kwargs(cls, uri):
        sha: Optional[str]
        parsed = urlparse(uri)
        repo, *path = parsed.path.strip("/").split("/src/")
        if not path:
            return {"repo": repo, "path": ""}
        sha, path = _mathch_path_with_ref(repo, path[0])
        return {"repo": repo, "sha": sha, "path": path}

    @classmethod
    def check_rev(cls, options):
        conf = BitbucketConfig.local()
        password = conf.PASSWORD
        username = conf.USERNAME
        return BitbucketWrapper(
            BITBUCKET_ORG, username=username, password=password
        ).check_rev(options["repo"], options["sha"])

    @classmethod
    def get_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: BitBucketFileSystem,
    ):
        fullpath = posixpath.join(project or "", path)
        return f"{BITBUCKET_ORG}/{fs.repo}/src/{fs.root}/{fullpath}"

    @classmethod
    def get_project_uri(  # pylint: disable=unused-argument
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: BitBucketFileSystem,
        uri: str,
    ):
        return f"{BITBUCKET_ORG}/{fs.repo}/src/{fs.root}/{project or ''}"
