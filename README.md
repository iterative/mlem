![image](https://user-images.githubusercontent.com/6797716/165590476-994d4d93-8e98-4afb-b5f8-6f42b9d56efc.png)


[![Check, test and release](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml/badge.svg)](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml)
[![codecov](https://codecov.io/gh/iterative/mlem/branch/main/graph/badge.svg?token=WHU4OAB6O2)](https://codecov.io/gh/iterative/mlem)
[![PyPi](https://img.shields.io/pypi/v/mlem.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/mlem)
[![License: Apache 2.0](https://img.shields.io/github/license/iterative/mlem)](https://github.com/iterative/mlem/blob/master/LICENSE)
<!-- [![Maintainability](https://codeclimate.com/github/iterative/mlem/badges/gpa.svg)](https://codeclimate.com/github/iterative/mlem) -->

MLEM helps you package and deploy machine learning models.
It saves ML models in a standard format that can be used in a variety of production scenarios such as real-time REST serving or batch processing.

- **Run your ML models anywhere:**
  Wrap models as a Python package or Docker Image, or deploy them to Heroku (SageMaker, Kubernetes, and more platforms coming soon).
  Switch between platforms transparently, with a single command.

- **Model metadata into YAML automatically:**
  Automatically include Python requirements and input data needs into a human-readable, deployment-ready format.
  Use the same metafile on any ML framework.

- **Stick to your training workflow:**
  MLEM doesn't ask you to rewrite model training code.
  Add just two lines around your Python code: one to import the library and one to save the model.

- **Developer-first experience:**
  Use the CLI when you feel like DevOps, or the API if you feel like a developer.

## Why is MLEM special?

The main reason to use MLEM instead of other tools is to adopt a **GitOps approach** to manage model lifecycles.

- **Git as a single source of truth:**
  MLEM writes model metadata to a plain text file that can be versioned in Git along with code.
  This enables GitFlow and other software engineering best practices.

- **Unify model and software deployment:**
  Release models using the same processes used for software updates (branching, pull requests, etc.).

- **Reuse existing Git infrastructure:**
  Use familiar hosting like Github or Gitlab for model management, instead of having separate services.

- **UNIX philosophy:**
  MLEM is a modular tool that solves one problem very well.
  It integrates well into a larger toolset from Iterative.ai, such as [DVC](https://dvc.org/) and [CML](https://cml.dev/).

## Usage

This a quick walkthrough showcasing deployment and export functionality of MLEM.

Please read [Get Started guide](https://mlem.ai/doc/get-started) for a full version.

### Installation

MLEM requires Python 3.

```console
$ python -m pip install mlem
```

> To install the pre-release version:
>
> ```console
> $ python -m pip install git+https://github.com/iterative/mlem
> ```

### Saving the model

```python
# train.py
from mlem.api import save
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def main():
    data, y = load_iris(return_X_y=True, as_frame=True)
    rf = RandomForestClassifier(
        n_jobs=2,
        random_state=42,
    )
    rf.fit(data, y)

    save(
        rf,
        "rf",
        sample_data=data,
    )

if __name__ == "__main__":
    main()
```

Check out what we have:

```shell
$ ls
rf
rf.mlem
$ cat rf.mlem
```
<details>
  <summary> Click to show `cat` output</summary>

```yaml
artifacts:
  data:
    hash: ea4f1bf769414fdacc2075ef9de73be5
    size: 163651
    uri: rf
model_type:
  methods:
    predict:
      args:
      - name: data
        type_:
          columns:
          - sepal length (cm)
          - sepal width (cm)
          - petal length (cm)
          - petal width (cm)
          dtypes:
          - float64
          - float64
          - float64
          - float64
          index_cols: []
          type: dataframe
      name: predict
      returns:
        dtype: int64
        shape:
        - null
        type: ndarray
    predict_proba:
      args:
      - name: data
        type_:
          columns:
          - sepal length (cm)
          - sepal width (cm)
          - petal length (cm)
          - petal width (cm)
          dtypes:
          - float64
          - float64
          - float64
          - float64
          index_cols: []
          type: dataframe
      name: predict_proba
      returns:
        dtype: float64
        shape:
        - null
        - 3
        type: ndarray
    sklearn_predict:
      args:
      - name: X
        type_:
          columns:
          - sepal length (cm)
          - sepal width (cm)
          - petal length (cm)
          - petal width (cm)
          dtypes:
          - float64
          - float64
          - float64
          - float64
          index_cols: []
          type: dataframe
      name: predict
      returns:
        dtype: int64
        shape:
        - null
        type: ndarray
    sklearn_predict_proba:
      args:
      - name: X
        type_:
          columns:
          - sepal length (cm)
          - sepal width (cm)
          - petal length (cm)
          - petal width (cm)
          dtypes:
          - float64
          - float64
          - float64
          - float64
          index_cols: []
          type: dataframe
      name: predict_proba
      returns:
        dtype: float64
        shape:
        - null
        - 3
        type: ndarray
  type: sklearn
object_type: model
requirements:
- module: sklearn
  version: 1.0.2
- module: pandas
  version: 1.4.1
- module: numpy
  version: 1.22.3
```
</details>

### Deploying the model

If you want to follow this Quick Start, you'll need to sign up on https://heroku.com,
create an API_KEY and populate `HEROKU_API_KEY` env var.

First, create an environment to deploy your model:

```shell
$ mlem declare env heroku staging
💾 Saving env to staging.mlem
```

Now we can [deploy the model with `mlem deploy`](https://mlem.ai/doc/get-started/deploying)
(you need to use different `app_name`, since it's going to be published on https://herokuapp.com):

```shell
$ mlem deployment run mydeploy -m rf -t staging -c app_name=mlem-quick-start
⏳️ Loading deployment from .mlem/deployment/myservice.mlem
🔗 Loading link to .mlem/env/staging.mlem
🔗 Loading link to .mlem/model/rf.mlem
💾 Updating deployment at .mlem/deployment/myservice.mlem
🏛 Creating Heroku App example-mlem-get-started
💾 Updating deployment at .mlem/deployment/myservice.mlem
🛠 Creating docker image for heroku
  💼 Adding model files...
  🛠 Generating dockerfile...
  💼 Adding sources...
  💼 Generating requirements file...
  🛠 Building docker image registry.heroku.com/example-mlem-get-started/web...
  ✅  Built docker image registry.heroku.com/example-mlem-get-started/web
  🔼 Pushed image registry.heroku.com/example-mlem-get-started/web to remote registry at host registry.heroku.com
💾 Updating deployment at .mlem/deployment/myservice.mlem
🛠 Releasing app my-mlem-service formation
💾 Updating deployment at .mlem/deployment/myservice.mlem
✅  Service example-mlem-get-started is up. You can check it out at https://mlem-quick-start.herokuapp.com/
```

### Exporting the model

You could easily [export the model to a different format using `mlem build`](https://mlem.ai/doc/get-started/building):

```
$ mlem build rf docker -c server.type=fastapi -c image.name=sklearn-model
⏳️ Loading model from rf.mlem
🛠 Building MLEM wheel file...
💼 Adding model files...
🛠 Generating dockerfile...
💼 Adding sources...
💼 Generating requirements file...
🛠 Building docker image sklearn-model:latest...
✅  Built docker image sklearn-model:latest
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](https://mlem.ai/doc/contributing/core)
for more details.

Check out the [MLEM weekly board](https://github.com/orgs/iterative/projects/322/views/4)
to learn about what we do, and about the exciting new functionality that is going to be added soon.

Thanks to all our contributors!

## Copyright

This project is distributed under the Apache license version 2.0 (see the LICENSE file in the project root).

By submitting a pull request to this project, you agree to license your contribution under the Apache license version 2.0 to this project.
