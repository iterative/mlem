try:
    from ._mlem_version import version as __version__
    from ._mlem_version import version_tuple
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except (LookupError, ImportError):
        __version__ = "UNKNOWN"
        version_tuple = ()  # type: ignore

__all__ = ["__version__", "version_tuple"]
