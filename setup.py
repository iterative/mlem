import importlib.util
import os
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

# Prevents pkg_resources import in entry point script,
# see https://github.com/ninjaaron/fast-entry_points.
# This saves about 200 ms on startup time for non-wheel installs.
# try:
#     import fastentrypoints  # noqa: F401, pylint: disable=unused-import
# except ImportError:
#     pass  # not able to import when installing through pre-commit


# Read package meta-data from version.py
# see https://packaging.python.org/guides/single-sourcing-package-version/
pkg_dir = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(pkg_dir, "mlem", "version.py")
spec = importlib.util.spec_from_file_location("mlem.version", version_path)
mlem_version = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(mlem_version)  # type: ignore
version = mlem_version.__version__  # type: ignore


# To achieve consistency between the build version and the one provided
# by your package during runtime, you need to **pin** the build version.
#
# This custom class will replace the version.py module with a **static**
# `__version__` that your package can read at runtime, assuring consistency.
#
# References:
#   - https://docs.python.org/3.7/distutils/extending.html
#   - https://github.com/python/mypy
class build_py(_build_py):
    def pin_version(self):
        path = os.path.join(self.build_lib, "mlem")
        self.mkpath(path)
        with open(
            os.path.join(path, "version.py"), "w", encoding="utf8"
        ) as fobj:
            fobj.write("# AUTOGENERATED at build time by setup.py\n")
            fobj.write(f'__version__ = "{version}"\n')

    def run(self):
        self.execute(self.pin_version, ())
        _build_py.run(self)


install_requires = [
    "dill",
    "requests",
    "isort==4.3.21",  # TODO: https://github.com/iterative/mlem/issues/45 version
    "docker",
    "pydantic<2",
    "click<9",
    "aiohttp<4",
    "aiohttp_swagger<2",
    "Jinja2<3",
    "fsspec>=2021.7.0",
    "fusepy",  # TMP
    "cached-property",
    "entrypoints",
]

# storage
dvc = ["dvc~=2.0"]

# data
pandas = ["pandas", "lxml", "openpyxl", "xlrd", "tables", "pyarrow"]
numpy = ["numpy"]
sql = ["sqlalchemy", "psycopg2-binary"]

# models
sklearn = ["scipy", "scikit-learn"]
tensorflow = ["tensorflow"]
catboost = ["catboost"]
xgboost = ["xgboost"]
lightgbm = ["lightgbm"]

# serve & deploy
fastapi = ["uvicorn", "fastapi"]
sagemaker = ["boto3", "sagemaker"]

all_libs = (
    dvc
    + pandas
    + numpy
    + sql
    + sklearn
    + tensorflow
    + catboost
    + xgboost
    + lightgbm
    + fastapi
    + sagemaker
)

tests = [
    "pytest",
    # we use this to suppress pytest-related false positives in our tests.
    "pylint-pytest",
    # we use this to suppress some messages in tests, eg: foo/bar naming,
    # and, protected method calls in our tests
    "pylint-plugin-utils",
    "pytest-lazy-fixture==0.6.3",
    "pytest-mock",
] + all_libs


setup_args = dict(  # noqa: C408
    name="mlem",
    version=version,
    description="Version and deploy your models following GitOps",
    long_description=(Path(__file__).parent / "README.md").read_text(
        encoding="utf8"
    ),
    author="Mikhail Sveshnikov",
    author_email="mike0sv@iterative.ai",
    download_url="https://github.com/iterative/mlem",
    license="Apache License 2.0",
    install_requires=install_requires,
    extras_require={
        "tests": tests,
        "all": all_libs,
        "dvc": dvc,
        "pandas": pandas,
        "numpy": numpy,
        "sql": sql,
        "sklearn": sklearn,
        "tensoflow": tensorflow,
        "catboost": catboost,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
        "fastapi": fastapi,
        "sagemaker": sagemaker,
    },
    keywords="data-science data-version-control machine-learning git mlops"
    " developer-tools reproducibility collaboration ai",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    url="http://mlem.ai",
    entry_points={
        "console_scripts": ["mlem = mlem.cli:cli"],
        # Additional mechanism for plugins.
        # This is the way for mlem to find implementations in installed modules.
        # Since mlem has some "optional" implementations,
        # we should populate them like this as well
        "mlem.contrib": [
            "dataset_reader.numpy = mlem.contrib.numpy:NumpyArrayReader",
            "dataset_reader.pandas = mlem.contrib.pandas:PandasReader",
            "dataset_type.dataframe = mlem.contrib.pandas:DataFrameType",
            "dataset_type.lightgbm = mlem.contrib.lightgbm:LightGBMDatasetType",
            "dataset_type.ndarray = mlem.contrib.numpy:NumpyNdarrayType",
            "dataset_type.number = mlem.contrib.numpy:NumpyNumberType",
            "dataset_type.xgboost_dmatrix = mlem.contrib.xgboost:DMatrixDatasetType",
            "dataset_writer.numpy = mlem.contrib.numpy:NumpyArrayWriter",
            "dataset_writer.pandas = mlem.contrib.pandas:PandasWriter",
            "model_io.catboost_io = mlem.contrib.catboost:CatBoostModelIO",
            "model_io.lightgbm_io = mlem.contrib.lightgbm:LightGBMModelIO",
            "model_io.xgboost_io = mlem.contrib.xgboost:XGBoostModelIO",
            "model_type.catboost = mlem.contrib.catboost:CatBoostModel",
            "model_type.lightgbm = mlem.contrib.lightgbm:LightGBMModel",
            "model_type.sklearn = mlem.contrib.sklearn:SklearnModel",
            "model_type.xgboost = mlem.contrib.xgboost:XGBoostModel",
            "server.fastapi = mlem.contrib.fastapi:FastAPIServer",
        ],
    },
    cmdclass={"build_py": build_py},
    zip_safe=False,
)

if __name__ == "__main__":
    setup(**setup_args)
