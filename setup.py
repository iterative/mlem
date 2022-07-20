from pathlib import Path

from setuptools import find_packages, setup

install_requires = [
    "dill",
    "requests",
    "isort>=5.10",
    "pydantic>=1.9.0,<2",
    "typer",
    "click<8.2",
    "rich",
    "aiohttp<4",
    "aiohttp_swagger<2",
    "Jinja2>=3",
    "fsspec>=2021.7.0",
    "pyparsing<4",  # legacy resolver problem
    "cached-property",
    "entrypoints",
    "gitpython",
    "python-gitlab",
    "flatdict",
    "iterative-telemetry==0.0.0",
]

tests = [
    "pytest",
    "pytest-cov",
    "pytest-lazy-fixture==0.6.3",
    "pytest-mock",
    "pylint<2.14",
    # we use this to suppress pytest-related false positives in our tests.
    "pylint-pytest",
    # we use this to suppress some messages in tests, eg: foo/bar naming,
    # and, protected method calls in our tests
    "pylint-plugin-utils",
    # we use this to mark tests that needs to be retried
    "flaky",
    "s3fs",
    "boto3",
    "botocore",
    "adlfs",
    "gcsfs",
    "testcontainers",
    "emoji",
    "lxml",
    "openpyxl",
    "xlrd",
    "tables",
    "pyarrow",
    "skl2onnx",
]

extras = {
    "tests": tests,
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "sklearn": ["scikit-learn"],
    "onnx": ["onnx"],
    "onnxruntime": [
        "protobuf==3.20.1",
        "onnxruntime",
    ],  # TODO - see if it can be merged with onnx
    "catboost": ["catboost"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "fastapi": ["uvicorn", "fastapi"],
    # "sagemaker": ["boto3==1.19.12", "sagemaker"],
    "torch": ["torch"],
    "tensorflow": ["tensorflow"],
    "azure": ["adlfs>=2021.10.0", "azure-identity>=1.4.0", "knack"],
    "gs": ["gcsfs>=2021.11.1"],
    "hdfs": [
        "pyarrow>=1",
        "fsspec[arrow]",
    ],
    "s3": ["s3fs[boto3]>=2021.11.1", "aiobotocore[boto3]>2"],
    "ssh": ["bcrypt", "sshfs[bcrypt]>=2021.11.2"],
    "rmq": ["pika"],
    "docker": ["docker"],
    "heroku": ["docker"],
    "dvc": ["dvc~=2.0"],
}

# add DVC extras
for e in [
    "azure",
    "gdrive",
    "gs",
    "hdfs",
    "oss",
    "s3",
    "ssh",
    "ssh_gssapi",
    "webdav",
    "webhdfs",
    "webdhfs_kerberos",
]:
    extras[f"dvc-{e}"] = [f"dvc[{e}]~=2.0"]

extras["all"] = [_ for e in extras.values() for _ in e]
extras["tests"] += [e for e in extras["all"] if e[: len("dvc[")] != "dvc["]

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
            "resolver.bitbucket = mlem.contrib.bitbucketfs:BitBucketResolver",
            "model_type.callable = mlem.contrib.callable:CallableModelType",
            "model_io.pickle = mlem.contrib.callable:PickleModelIO",
            "model_type.catboost = mlem.contrib.catboost:CatBoostModel",
            "model_io.catboost_io = mlem.contrib.catboost:CatBoostModelIO",
            "deployment.docker_container = mlem.contrib.docker.base:DockerContainer",
            "deploy_state.docker_container = mlem.contrib.docker.base:DockerContainerState",
            "builder.docker_dir = mlem.contrib.docker.base:DockerDirBuilder",
            "env.docker = mlem.contrib.docker.base:DockerEnv",
            "docker_registry.docker_io = mlem.contrib.docker.base:DockerIORegistry",
            "builder.docker = mlem.contrib.docker.base:DockerImageBuilder",
            "docker_registry = mlem.contrib.docker.base:DockerRegistry",
            "docker_registry.remote = mlem.contrib.docker.base:RemoteRegistry",
            "artifact.dvc = mlem.contrib.dvc:DVCArtifact",
            "storage.dvc = mlem.contrib.dvc:DVCStorage",
            "server.fastapi = mlem.contrib.fastapi:FastAPIServer",
            "resolver.github = mlem.contrib.github:GithubResolver",
            "resolver.gitlab = mlem.contrib.gitlabfs:GitlabResolver",
            "docker_registry.heroku = mlem.contrib.heroku.build:HerokuRemoteRegistry",
            "deployment.heroku = mlem.contrib.heroku.meta:HerokuDeployment",
            "env.heroku = mlem.contrib.heroku.meta:HerokuEnv",
            "deploy_state.heroku = mlem.contrib.heroku.meta:HerokuState",
            "server.heroku = mlem.contrib.heroku.server:HerokuServer",
            "data_reader.lightgbm = mlem.contrib.lightgbm:LightGBMDataReader",
            "data_type.lightgbm = mlem.contrib.lightgbm:LightGBMDataType",
            "data_writer.lightgbm = mlem.contrib.lightgbm:LightGBMDataWriter",
            "model_type.lightgbm = mlem.contrib.lightgbm:LightGBMModel",
            "model_io.lightgbm_io = mlem.contrib.lightgbm:LightGBMModelIO",
            "data_reader.numpy = mlem.contrib.numpy:NumpyArrayReader",
            "data_writer.numpy = mlem.contrib.numpy:NumpyArrayWriter",
            "data_type.ndarray = mlem.contrib.numpy:NumpyNdarrayType",
            "data_reader.numpy_number = mlem.contrib.numpy:NumpyNumberReader",
            "data_type.number = mlem.contrib.numpy:NumpyNumberType",
            "data_writer.numpy_number = mlem.contrib.numpy:NumpyNumberWriter",
            "model_io.model_proto = mlem.contrib.onnx:ModelProtoIO",
            "model_type.onnx = mlem.contrib.onnx:ONNXModel",
            "data_type.dataframe = mlem.contrib.pandas:DataFrameType",
            "import.pandas = mlem.contrib.pandas:PandasImport",
            "data_reader.pandas = mlem.contrib.pandas:PandasReader",
            "data_reader.pandas_series = mlem.contrib.pandas:PandasSeriesReader",
            "data_writer.pandas_series = mlem.contrib.pandas:PandasSeriesWriter",
            "data_writer.pandas = mlem.contrib.pandas:PandasWriter",
            "data_type.series = mlem.contrib.pandas:SeriesType",
            "builder.pip = mlem.contrib.pip.base:PipBuilder",
            "builder.whl = mlem.contrib.pip.base:WhlBuilder",
            "client.rmq = mlem.contrib.rabbitmq:RabbitMQClient",
            "server.rmq = mlem.contrib.rabbitmq:RabbitMQServer",
            "model_type.sklearn = mlem.contrib.sklearn:SklearnModel",
            "model_type.sklearn_pipeline = mlem.contrib.sklearn:SklearnPipelineType",
            "model_type.tf_keras = mlem.contrib.tensorflow:TFKerasModel",
            "model_io.tf_keras = mlem.contrib.tensorflow:TFKerasModelIO",
            "data_type.tf_tensor = mlem.contrib.tensorflow:TFTensorDataType",
            "data_reader.tf_tensor = mlem.contrib.tensorflow:TFTensorReader",
            "data_writer.tf_tensor = mlem.contrib.tensorflow:TFTensorWriter",
            "model_type.torch = mlem.contrib.torch:TorchModel",
            "model_io.torch_io = mlem.contrib.torch:TorchModelIO",
            "data_type.torch = mlem.contrib.torch:TorchTensorDataType",
            "data_reader.torch = mlem.contrib.torch:TorchTensorReader",
            "data_writer.torch = mlem.contrib.torch:TorchTensorWriter",
            "data_type.xgboost_dmatrix = mlem.contrib.xgboost:DMatrixDataType",
            "model_type.xgboost = mlem.contrib.xgboost:XGBoostModel",
            "model_io.xgboost_io = mlem.contrib.xgboost:XGBoostModelIO",
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
