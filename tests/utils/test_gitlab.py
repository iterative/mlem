from mlem.utils.gitlab import GitlabFileSystem


def test_ls():
    fs = GitlabFileSystem("mike0sv/fsspec-test", "")
    print(fs.ls(""))