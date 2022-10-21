import posixpath
from typing import ClassVar, Optional
from urllib.parse import urlparse

from morefs.gitlab import GitlabFileSystem, check_rev, match_path_with_ref

from mlem.core.meta_io import CloudGitResolver


class GitlabResolver(CloudGitResolver):
    type: ClassVar = "gitlab"
    FS = GitlabFileSystem
    PROTOCOL = "gitlab"
    GITLAB_COM = "https://gitlab.com"

    # TODO: support on-prem gitlab (other hosts)
    PREFIXES = [GITLAB_COM, PROTOCOL + "://"]
    versioning_support = True

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
        sha, path = match_path_with_ref(project_id, path[0])
        return {"project_id": project_id, "sha": sha, "path": path}

    @classmethod
    def check_rev(cls, options):
        return check_rev(options["project_id"], options["sha"])

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
