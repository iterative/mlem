import requests
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import infer_storage_options
from gitlab import Gitlab


class GitlabFileSystem(AbstractFileSystem):
    """Interface to files in github

    An instance of this class provides the files residing within a remote github
    repository. You may specify a point in the repos history, by SHA, branch
    or tag (default is current master).

    Given that code files tend to be small, and that github does not support
    retrieving partial content, we always fetch whole files.

    When using fsspec.open, allows URIs of the form:

    - "github://path/file", in which case you must specify org, repo and
      may specify sha in the extra args
    - 'github://org:repo@/precip/catalog.yml', where the org and repo are
      part of the URI
    - 'github://org:repo@sha/precip/catalog.yml', where the sha is also included

    ``sha`` can be the full or abbreviated hex of the commit you want to fetch
    from, or a branch or tag name (so long as it doesn't contain special characters
    like "/", "?", which would have to be HTTP-encoded).

    For authorised access, you must provide username and token, which can be made
    at https://github.com/settings/tokens
    """

    url = "https://gitlab.com/api/projects/{project_id}/repository/tree"
    # rurl = "https://raw.githubusercontent.com/{org}/{repo}/{sha}/{path}"
    protocol = "gitlab"

    def __init__(self, project_id, repo, sha=None, username=None, token=None, **kwargs):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.repo = repo
        if (username is None) ^ (token is None):
            raise ValueError("Auth required both username and token")
        self.username = username
        self.token = token
        self.gl = Gitlab()
        self.project = self.gl.projects.get(self.project_id)
        # if sha is None:
        #     # look up default branch (not necessarily "master")
        #     u = "https://api.github.com/repos/{org}/{repo}"
        #     r = requests.get(u.format(org=org, repo=repo), **self.kw)
        #     r.raise_for_status()
        #     sha = r.json()["default_branch"]

        self.root = "main" #sha
        self.ls("")

    @property
    def kw(self):
        if self.username:
            return {"auth": (self.username, self.token)}
        return {}

    @classmethod
    def repos(cls, org_or_user, is_org=True):
        """List repo names for given org or user

        This may become the top level of the FS

        Parameters
        ----------
        org_or_user: str
            Name of the github org or user to query
        is_org: bool (default True)
            Whether the name is an organisation (True) or user (False)

        Returns
        -------
        List of string
        """
        r = requests.get(
            "https://api.github.com/{part}/{org}/repos".format(
                part=["users", "orgs"][is_org], org=org_or_user
            )
        )
        r.raise_for_status()
        return [repo["name"] for repo in r.json()]

    @property
    def tags(self):
        """Names of tags in the repo"""
        r = requests.get(
            "https://api.github.com/repos/{org}/{repo}/tags"
            "".format(org=self.org, repo=self.repo),
            **self.kw,
        )
        r.raise_for_status()
        return [t["name"] for t in r.json()]

    @property
    def branches(self):
        """Names of branches in the repo"""
        r = requests.get(
            "https://api.github.com/repos/{org}/{repo}/branches"
            "".format(org=self.org, repo=self.repo),
            **self.kw,
        )
        r.raise_for_status()
        return [t["name"] for t in r.json()]

    @property
    def refs(self):
        """Named references, tags and branches"""
        return {"tags": self.tags, "branches": self.branches}

    def ls(self, path, detail=False, sha=None, _sha=None, **kwargs):
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
        _sha: str (optional)
            List this specific tree object (used internally to descend into trees)
        """
        path = self._strip_protocol(path)
        if path == "":
            _sha = sha or self.root
        if _sha is None:
            parts = path.rstrip("/").split("/")
            so_far = ""
            _sha = sha or self.root
            for part in parts:
                out = self.ls(so_far, True, sha=sha, _sha=_sha)
                so_far += "/" + part if so_far else part
                out = [o for o in out if o["name"] == so_far]
                if not out:
                    raise FileNotFoundError(path)
                out = out[0]
                if out["type"] == "file":
                    if detail:
                        return [out]
                    else:
                        return path
                _sha = out["sha"]
        if path not in self.dircache or sha not in [self.root, None]:
            r = self.project.repository_tree(path=path)
            out = [
                {
                    "name": path + "/" + f["path"] if path else f["path"],
                    "mode": f["mode"],
                    "type": {"blob": "file", "tree": "directory"}[f["type"]],
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
        else:
            return sorted([f["name"] for f in out])

    def invalidate_cache(self, path=None):
        self.dircache.clear()

    @classmethod
    def _strip_protocol(cls, path):
        opts = infer_storage_options(path)
        if "username" not in opts:
            return super()._strip_protocol(path)
        return opts["path"].lstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(path):
        opts = infer_storage_options(path)
        if "username" not in opts:
            return {}
        out = {"org": opts["username"], "repo": opts["password"]}
        if opts["host"]:
            out["sha"] = opts["host"]
        return out

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
        return MemoryFile(None, None, self.project.files.get(path, ref=sha or self.root).decode())


def main():
    fs = GitlabFileSystem("mike0sv/fsspec-test", "")
    print(fs.ls(""))
    print(fs.open("README.md").read())


if __name__ == '__main__':
    main()