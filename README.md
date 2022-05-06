![image](https://user-images.githubusercontent.com/6797716/165590476-994d4d93-8e98-4afb-b5f8-6f42b9d56efc.png)


[![Check, test and release](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml/badge.svg)](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml)
[![codecov](https://codecov.io/gh/iterative/mlem/branch/main/graph/badge.svg?token=WHU4OAB6O2)](https://codecov.io/gh/iterative/mlem)
[![PyPi](https://img.shields.io/pypi/v/mlem.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/mlem)

MLEM helps you with model deployment. It saves ML models in a standard format that can be used in a variety of downstream deployment scenarios such as real-time serving through a REST API or batch processing. MLEM format is a human-readable text that helps you use GitOps with Git as the single source of truth.

- **Run your model anywhere you want:** package it as a Python package, a Docker Image or deploy it to Heroku (SageMaker, Kubernetes and more platforms are coming). Switch between formats and deployment platforms with a single command thanks to unified abstraction.
- **Simple text file to save model metadata:** automatically package Python env requirements and input data specifications into a ready-to-deploy format. Use the same human-readable format for any ML framework.
- **Stick to your training workflow:** MLEM doesn't ask you to rewrite your training code. To start using packaging or deployment machinery, add just two lines to your python script: one to import the library and one to save the model.
- **Developer-first experience:** use CLI when you feel like DevOps and API when you feel like a developer.

## Why MLEM?

The main reason to use MLEM instead of other related solutions is that it works well with **GitOps approach** and helps you manage model lifecycle in Git:

- **Git as a single source of truth:** we use plain text to save metadata for models that can be saved and versioned.
- **Reuse existing Git and Github/Gitlab infrastructure** for model management instead of installing separate model management software.
- **Unify model and software deployment.** Deploy models using the same processes and code you use to deploy software.

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
from mlem.api import load, save
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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
  <summary>Output: yaml file contents</summary>
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
$ mlem create deployment staging myservice -c app_name=mlem-quick-start -c model=rf -c env=staging
💾 Saving deployment to service_name.mlem
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
✅  Service example-mlem-get-started is up. You can check it out at https://my-mlem-service.herokuapp.com/
```

### Check the deployment

### Stop the deployment

