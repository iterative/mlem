![image](https://user-images.githubusercontent.com/6797716/165590476-994d4d93-8e98-4afb-b5f8-6f42b9d56efc.png)


[![Check, test and release](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml/badge.svg)](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml)
[![codecov](https://codecov.io/gh/iterative/mlem/branch/main/graph/badge.svg?token=WHU4OAB6O2)](https://codecov.io/gh/iterative/mlem)
[![PyPi](https://img.shields.io/pypi/v/mlem.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/mlem)

MLEM helps you package and deploy machine learning models.
It saves model metadata in a standard, human-readable format that can be used in a variety of deployment scenarios, such as real-time serving through a REST API or a batch processing task.
MLEM lets you keep Git as the single source of truth for code and models.

- **Run your ML models anywhere:**
  Wrap models as a Python package or Docker Image, or deploy them to Heroku (SageMaker, Kubernetes, and more platforms coming soon).
  Switch between platforms transparently, with a single command.

- **Simple text file to save model metadata:**
  Automatically include Python requirements and input data needs into a deployment-ready format.
  Use the same format on any ML framework.

- **Stick to your training workflow:**
  MLEM doesn't ask you to rewrite model training code.
  Just add two lines around your Python code: one to import the library and one to save the model.

- **Developer-first experience:**
  Use the CLI when you feel like DevOps and the API when you feel like a developer.

## Why is MLEM special?

The main reason to use MLEM instead of other tools is to adopt a **GitOps approach**, helping you manage model lifecycles in Git:

- **Git as a single source of truth:** MLEM writes model metadata to a plain text file that can be versioned in a Git repo along with your code.
- **Unify model and software deployment:** Release models using the same processes used for software updates (branching, pull requests, etc.).
- **Reuse existing Git infrastructure:** Use familiar hosting like Github or Gitlab for model management, instead of having separate services.

## Installation

MLEM requires Python 3.

```console
$ pyhon -m pip install mlem
```

To install the bleeding-edge version, use:

```console
$ pyhon -m pip install git+https://github.com/iterative/mlem
```

## Anonymized Usage Analytics

To help us better understand how MLEM is used and improve it, MLEM captures and reports anonymized usage statistics.

MLEM's analytics record the following information per event:

- MLEM version (e.g., `0.1.2+5fb5a3.mod`) and OS version (e.g., `MacOS 10.16`)
- Command name and exception type (e.g., `ls, ValueError` or `get, MLEMRootNotFound`)
- Country, city (e.g., `RU, Moscow`)
- A random user_id (generated with [uuid](https://docs.python.org/3/library/uuid.html))

> The code is viewable in [analytics.py](https://github.com/iterative/mlem/mlem/analytics.py).
> MLEM's analytics are sent through Iterative's proxy to Google BigQuery over HTTPS.

### Opting out

MLEM analytics have no performance impact and help the entire community, so leaving them on is appreciated.
However, to opt out set an environment variable `MLEM_NO_ANALYTICS=true` or add `no_analytics: true` to `.mlem/config.yaml`.

> This will disable it for the project.
> We'll add an option to opt out globally soon.
