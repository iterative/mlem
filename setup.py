from pathlib import Path

from setuptools import find_packages, setup

install_requires = [
    "dill",
    "requests",
    "isort>=5.10",
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
    "gitpython",
    "flatdict",
    "iterative-telemetry",
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
    "s3fs",
    "boto3",
    "botocore",
    "adlfs",
    "gcsfs",
    "testcontainers",
    "emoji",
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
    # "sagemaker": ["boto3==1.19.12", "sagemaker"],
    "torch": ["torch"],
    "azure": ["adlfs>=2021.10.0", "azure-identity>=1.4.0", "knack"],
    "gs": ["gcsfs>=2021.11.1"],
    "hdfs": [
        "pyarrow>=1",
        "fsspec[arrow]",
    ],
    "s3": ["s3fs[boto3]>=2021.11.1", "aiobotocore[boto3]>2"],
    "ssh": ["bcrypt", "sshfs[bcrypt]>=2021.11.2"],
    "rmq": ["pika"],
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
            "client.rmq = mlem.contrib.rabbitmq:RabbitMQClient",
            "data_reader.lightgbm = mlem.contrib.lightgbm:LightGBMDataReader",
            "data_reader.numpy = mlem.contrib.numpy:NumpyArrayReader",
            "data_reader.numpy_number = mlem.contrib.numpy:NumpyNumberReader",
            "data_reader.pandas = mlem.contrib.pandas:PandasReader",
            "data_reader.pandas_series = mlem.contrib.pandas:PandasSeriesReader",
            "data_reader.torch = mlem.contrib.torch:TorchTensorReader",
            "data_type.dataframe = mlem.contrib.pandas:DataFrameType",
            "data_type.lightgbm = mlem.contrib.lightgbm:LightGBMDataType",
            "data_type.ndarray = mlem.contrib.numpy:NumpyNdarrayType",
            "data_type.number = mlem.contrib.numpy:NumpyNumberType",
            "data_type.series = mlem.contrib.pandas:SeriesType",
            "data_type.torch = mlem.contrib.torch:TorchTensorDataType",
            "data_type.xgboost_dmatrix = mlem.contrib.xgboost:DMatrixDataType",
            "data_writer.lightgbm = mlem.contrib.lightgbm:LightGBMDataWriter",
            "data_writer.numpy = mlem.contrib.numpy:NumpyArrayWriter",
            "data_writer.numpy_number = mlem.contrib.numpy:NumpyNumberWriter",
            "data_writer.pandas = mlem.contrib.pandas:PandasWriter",
            "data_writer.pandas_series = mlem.contrib.pandas:PandasSeriesWriter",
            "data_writer.torch = mlem.contrib.torch:TorchTensorWriter",
            "deployment.heroku = mlem.contrib.heroku.meta:HerokuDeployment",
            "deploy_state.heroku = mlem.contrib.heroku.meta:HerokuState",
            "docker_registry = mlem.contrib.docker.base:DockerRegistry",
            "docker_registry.docker_io = mlem.contrib.docker.base:DockerIORegistry",
            "docker_registry.heroku = mlem.contrib.heroku.build:HerokuRemoteRegistry",
            "docker_registry.remote = mlem.contrib.docker.base:RemoteRegistry",
            "env.heroku = mlem.contrib.heroku.meta:HerokuEnv",
            "import.pandas = mlem.contrib.pandas:PandasImport",
            "model_io.catboost_io = mlem.contrib.catboost:CatBoostModelIO",
            "model_io.lightgbm_io = mlem.contrib.lightgbm:LightGBMModelIO",
            "model_io.pickle = mlem.contrib.callable:PickleModelIO",
            "model_io.xgboost_io = mlem.contrib.xgboost:XGBoostModelIO",
            "model_io.torch_io = mlem.contrib.torch:TorchModelIO",
            "model_type.callable = mlem.contrib.callable:CallableModelType",
            "model_type.catboost = mlem.contrib.catboost:CatBoostModel",
            "model_type.lightgbm = mlem.contrib.lightgbm:LightGBMModel",
            "model_type.sklearn = mlem.contrib.sklearn:SklearnModel",
            "model_type.sklearn_pipeline = mlem.contrib.sklearn:SklearnPipelineType",
            "model_type.xgboost = mlem.contrib.xgboost:XGBoostModel",
            "model_type.torch = mlem.contrib.torch:TorchModel",
            "builder.docker = mlem.contrib.docker.base:DockerImageBuilder",
            "builder.docker_dir = mlem.contrib.docker.base:DockerDirBuilder",
            "builder.pip = mlem.contrib.pip.base:PipBuilder",
            "builder.whl = mlem.contrib.pip.base:WhlBuilder",
            "server.fastapi = mlem.contrib.fastapi:FastAPIServer",
            "server.heroku = mlem.contrib.heroku.build:HerokuServer",
            "server.rmq = mlem.contrib.rabbitmq:RabbitMQServer",
            "storage.dvc = mlem.contrib.dvc:DVCStorage",
        ],
        "mlem.config": [
            "core = mlem.config:MlemConfig",
            "pandas = mlem.contrib.pandas:PandasConfig",
        ],
    },
    zip_safe=False,
)

if __name__ == "__main__":
    setup(**setup_args)
