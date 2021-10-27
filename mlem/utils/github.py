from typing import Dict

import git


def ls_remotes(repo_url: str) -> Dict[str, str]:
    git.cmd.Git().ls_remote(repo_url)
    g = git.cmd.Git()
    remote_refs: Dict[str, str] = dict(
        tuple(reversed(ref.split("\t")[:2]))
        for ref in g.ls_remote(repo_url).split("\n")
    )
    return remote_refs
