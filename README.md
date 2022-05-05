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

## Installation

Install MLEM with pip:

```
$ pip install mlem
```

To install the development version, run:

```
$ pip install git+https://github.com/iterative/mlem
```
