from pathlib import Path

from setuptools import find_packages, setup

install_requires = [
    "dill",
    "requests",
    "isort>4",
    "docker",
    "pydantic>=1.9.0,<2",
    "typer",
    "click<8.1",
    "rich",
    "aiohttp<4",
    "aiohttp_swagger<2",
    "Jinja2>=3",
    "fsspec>=2021.7.0",
    "pyparsing<3",  # legacy resolver problem
    "cached-property",
    "entrypoints",
    "filelock",
    "appdirs",
    "python-daemon",
    "distro",
    "gitpython",
]

tests = [
    "pytest",
    "pytest-cov",
    "pytest-lazy-fixture==0.6.3",
    "pytest-mock",
    "pylint",
    # we use this to suppress pytest-related false positives in our tests.
    "pylint-pytest",
    # we use this to suppress some messages in tests, eg: foo/bar naming,
    # and, protected method calls in our tests
    "pylint-plugin-utils",
    "s3fs==2021.10.1",
    "boto3==1.19.12",
    "botocore==1.22.12",
    "adlfs",
    "gcsfs",
    "testcontainers",
]

extras = {
    "tests": tests,
    "dvc": ["dvc~=2.0"],
    "pandas": ["pandas", "lxml", "openpyxl", "xlrd", "tables", "pyarrow"],
    "numpy": ["numpy"],
    "sklearn": ["scipy", "scikit-learn"],
    "catboost": ["catboost"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "fastapi": ["uvicorn", "fastapi"],
    "sagemaker": ["boto3==1.19.12", "sagemaker"],
    "torch": ["torch"],
}

extras["all"] = [_ for e in extras.values() for _ in e]
extras["tests"] += extras["all"]

setup_args = dict(  # noqa: C408
    name="mlem",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "fastentrypoints>=0.12"],
    description="Version and deploy your models following GitOps principles",
    long_description=(Path(__file__).parent / "README.md").read_text(
        encoding="utf8"
    ),
    long_description_content_type="text/markdown",
    maintainer="Iterative",
    maintainer_email="support@mlem.ai",
    author="Mikhail Sveshnikov",
    author_email="mike0sv@iterative.ai",
    download_url="https://github.com/iterative/mlem",
    license="Apache License 2.0",
    install_requires=install_requires,
    extras_require=extras,
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
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    url="https://mlem.ai",
    entry_points={
        "console_scripts": ["mlem = mlem.cli:app"],
        # Additional mechanism for plugins.
        # This is the way for mlem to find implementations in installed modules.
        # Since mlem has some "optional" implementations,
        # we should populate them like this as well
        "mlem.contrib": [
            "artifact.dvc = mlem.contrib.dvc:DVCArtifact",
            "dataset_reader.numpy = mlem.contrib.numpy:NumpyArrayReader",
            "dataset_reader.pandas = mlem.contrib.pandas:PandasReader",
            "dataset_type.dataframe = mlem.contrib.pandas:DataFrameType",
            "dataset_type.series = mlem.contrib.pandas:SeriesType",
            "dataset_type.lightgbm = mlem.contrib.lightgbm:LightGBMDatasetType",
            "dataset_type.ndarray = mlem.contrib.numpy:NumpyNdarrayType",
            "dataset_type.number = mlem.contrib.numpy:NumpyNumberType",
            "dataset_type.xgboost_dmatrix = mlem.contrib.xgboost:DMatrixDatasetType",
            "dataset_writer.numpy = mlem.contrib.numpy:NumpyArrayWriter",
            "dataset_writer.pandas = mlem.contrib.pandas:PandasWriter",
            "dataset_type.torch = mlem.contrib.torch:TorchTensorDatasetType",
            "deploy.heroku = mlem.contrib.heroku.meta:HerokuDeploy",
            "deploy_state.heroku = mlem.contrib.heroku.meta:HerokuState",
            "env.heroku = mlem.contrib.heroku.meta:HerokuEnvMeta",
            "model_io.catboost_io = mlem.contrib.catboost:CatBoostModelIO",
            "model_io.lightgbm_io = mlem.contrib.lightgbm:LightGBMModelIO",
            "model_io.pickle = mlem.contrib.callable:PickleModelIO",
            "model_io.xgboost_io = mlem.contrib.xgboost:XGBoostModelIO",
            "model_io.torch_io = mlem.contrib.torch:TorchModelIO",
            "model_type.callable = mlem.contrib.callable:CallableModelType",
            "model_type.catboost = mlem.contrib.catboost:CatBoostModel",
            "model_type.lightgbm = mlem.contrib.lightgbm:LightGBMModel",
            "model_type.sklearn = mlem.contrib.sklearn:SklearnModel",
            "model_type.xgboost = mlem.contrib.xgboost:XGBoostModel",
            "model_type.torch = mlem.contrib.torch:TorchModel",
            "packager.docker = mlem.contrib.docker.base:DockerImagePackager",
            "packager.docker_dir = mlem.contrib.docker.base:DockerDirPackager",
            "packager.pip = mlem.contrib.pip.base:PipPackager",
            "packager.whl = mlem.contrib.pip.base:WhlPackager",
            "server.fastapi = mlem.contrib.fastapi:FastAPIServer",
            "server.heroku = mlem.contrib.heroku.build:HerokuServer",
            "storage.dvc = mlem.contrib.dvc:DVCStorage",
        ],
    },
    zip_safe=False,
)

if __name__ == "__main__":
    setup(**setup_args)
