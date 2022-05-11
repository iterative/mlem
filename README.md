![image](https://user-images.githubusercontent.com/6797716/165590476-994d4d93-8e98-4afb-b5f8-6f42b9d56efc.png)


[![Check, test and release](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml/badge.svg)](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml)
[![codecov](https://codecov.io/gh/iterative/mlem/branch/main/graph/badge.svg?token=WHU4OAB6O2)](https://codecov.io/gh/iterative/mlem)
[![PyPi](https://img.shields.io/pypi/v/mlem.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/mlem)

MLEM helps you with machine learning model deployment. It saves ML models in a standard format that can be used in a variety of downstream deployment scenarios such as real-time serving through a REST API or batch processing.

- **Run your model anywhere you want:** package it as a Python package, a Docker Image or deploy it to Heroku (SageMaker, Kubernetes and more platforms are coming). Switch between formats and deployment platforms with a single command thanks to unified abstraction.
- **Simple YAML file to save model metadata:** automatically package Python env requirements and input data specifications into a ready-to-deploy format. Use the same human-readable format for any ML framework.
- **Stick to your training workflow:** MLEM doesn't ask you to rewrite your training code. To start using packaging or deployment machinery, add just two lines to your python script: one to import the library and one to save the model.
- **Developer-first experience:** use CLI when you feel like DevOps and API when you feel like a developer.

## Why MLEM?

- MLEM **automatically detects** ML framework, Python requirements, model methods and input/output data specifications, saving your time and preventing manual errors.
- MLEM is designed for **Git-centered** ML models development. Use GitOps with Git as the single source of truth. Enable GitFlow and other software engineering best practices.
- MLEM is made with **Unix philosophy** in mind - one tool solves one problem very well. Plug MLEM into your toolset, easily integrating it with other tools like DVC.

## Usage

### Installation

Install MLEM with pip:

```
$ pip install mlem
```

To install the development version, run:

```
$ pip install git+https://github.com/iterative/mlem
```

### Save your model

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
        tmp_sample_data=data,
        tags=["random-forest", "classifier"],
        description="Random Forest Classifier",
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
  <summary>Click to show `cat` output</summary>

```yaml
artifacts:
  data:
    hash: ea4f1bf769414fdacc2075ef9de73be5
    size: 163651
    uri: rf
description: Random Forest Classifier
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
tags:
- random-forest
- classifier
```
</details>


### Deploy it

Create an environment to deploy your model:

```shell
$ mlem create env heroku staging
💾 Saving env to staging.mlem
```

Define the deployment:

```shell
$ mlem create deployment heroku myservice -c app_name=mlem-quick-start -c model=rf -c env=staging
💾 Saving deployment to myservice.mlem
```

Deploy it:
```shell
$ mlem deploy create myservice
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

### Check the deployment

https://mlem-quick-start.herokuapp.com

Let's save some data first:
```python
# save_data.py
from mlem.api import save
from sklearn.datasets import load_iris

def main():
    data, y = load_iris(return_X_y=True, as_frame=True)
    save(
        data,
        "train.csv",
        description="Training data for Random Forest Classifier",
    )

if __name__ == "__main__":
    main()
```

```
$ mlem apply-remote http train.csv -c host=https://mlem-quick-start.herokuapp.com -c port=80 --json
```

### Stop the deployment

```
$ mlem deploy status myservice.mlem
running
```

```
$ mlem deploy teardown myservice.mlem
⏳️ Loading deployment from myservice.mlem
🔗 Loading link to file://staging.mlem
🔻 Deleting mlem-quick-start heroku app
💾 Updating deployment at myservice.mlem
```

```
$ mlem deploy status myservice.mlem
not_deployed
```
