import posixpath
from abc import abstractmethod
from typing import Optional
from urllib.parse import urljoin

import requests
from atlassian import Bitbucket
from atlassian.bitbucket import Cloud
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFile
from pydantic import Field

from mlem.config import MlemConfig


class BitbucketWrapper:
    def __init__(self, url: str):
        self.url = url

    @abstractmethod
    def tree(self, path: str, repo: str, rev: str):
        raise NotImplementedError

    @abstractmethod
    def get_default_branch(self, repo: str):
        raise NotImplementedError

    @abstractmethod
    def open(self, path: str, repo: str, rev: str):
        raise NotImplementedError


class AuthBitbucket(BitbucketWrapper):
    def __init__(self, url: str, username: str, password: str):
        super().__init__(url)
        self.bitbucket = Bitbucket(
            url=url, username=username, password=password, cloud=True
        )
        self.bitbucket2 = Cloud(
            username=username, password=password, cloud=True
        )

    def tree(self, path: str, repo: str, rev: str):
        self.bitbucket.repositories.get

    def get_default_branch(self, repo: str):
        self.bitbucket2.repositories.get(repo)["mainbranch"]["name"]
        return self.bitbucket.get_repo("", repo)["mainbranch"]["name"]

    def open(self, path: str, repo: str, rev: str):
        pass


class NoAuthBitbucket(BitbucketWrapper):
    tree_endpoint = "/api/internal/repositories/{repo}/tree/{rev}/{path}"
    repo_endpoint = "/api/2.0/repositories/{repo}"
    file_endpoint = "/api/2.0/repositories/{repo}/src/{rev}/{path}"

    def tree(self, path: str, repo: str, rev: str):
        r = requests.get(
            urljoin(
                self.url,
                self.tree_endpoint.format(path=path or "", repo=repo, rev=rev),
            )
        )
        r.raise_for_status()
        return r.json()[0]["contents"]

    def get_default_branch(self, repo: str):
        r = requests.get(
            urljoin(self.url, self.repo_endpoint.format(repo=repo))
        )
        r.raise_for_status()
        return r.json()["mainbranch"]["name"]

    def open(self, path: str, repo: str, rev: str):
        r = requests.get(
            urljoin(
                self.url,
                self.file_endpoint.format(path=path, repo=repo, rev=rev),
            )
        )
        r.raise_for_status()
        return r.content


class BitbucketConfig(MlemConfig):
    class Config:
        section = "bitbucket"

    USERNAME: Optional[str] = Field(default=None, env="BITBUCKET_USERNAME")
    PASSWORD: Optional[str] = Field(default=None, env="BITBUCKET_PASSWORD")


CONFIG = BitbucketConfig()


class BitBucketFileSystem(AbstractFileSystem):
    def __init__(
        self,
        repo: str,
        sha: str = None,
        host: str = "https://bitbucket.org",
        username: str = None,
        password: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bb: BitbucketWrapper
        self.password = password or CONFIG.PASSWORD
        self.username = username or CONFIG.USERNAME
        self.repo = repo
        self.host = host

        if self.username and self.password:
            self.bb = AuthBitbucket(host, self.username, self.password)
        else:
            self.bb = NoAuthBitbucket(host)
        if sha is None:
            sha = self.bb.get_default_branch(repo)
        self.root = sha

    def invalidate_cache(self, path=None):
        super().invalidate_cache(path)
        self.dircache.clear()

    def ls(self, path, detail=False, sha=None, **kwargs):
        path = self._strip_protocol(path)
        if path not in self.dircache or sha not in [self.root, None]:
            try:
                r = self.bb.tree(
                    path=path, repo=self.repo, rev=sha or self.root
                )
            except FileNotFoundError as e:
                if e.response_code == 404:
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

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        sha=None,
        **kwargs
    ):
        if mode != "rb":
            raise NotImplementedError
        return MemoryFile(
            None,
            None,
            self.bb.open(path, self.repo, rev=sha or self.root),
        )
