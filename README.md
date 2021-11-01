[![Check, test and release](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml/badge.svg)](https://github.com/iterative/mlem/actions/workflows/check-test-release.yml)
[![codecov](https://codecov.io/gh/iterative/mlem/branch/main/graph/badge.svg?token=WHU4OAB6O2)](https://codecov.io/gh/iterative/mlem)
[![PyPi](https://img.shields.io/pypi/v/mlem.svg?label=pip&logo=PyPI&logoColor=white)](https://pypi.org/project/mlem)

MLEM is in early alpha. Thank you for trying it out! 👋

Alpha include model registry functionality, and upcoming Beta will add model deployment functionality.

## What is MLEM 🐶

MLEM is a tool to help you version and deploy your Machine Learning models. At the top level, MLEM consists of two parts:

1. Model registry part:
    1. Storing model along with information required to use it: environment, methods, input data schema.
    2. Turning your Git repo into a model registry.
2. Deployment part:
    1. Packing a model to use in any serving scenario.
    2. Provider-agnostic deployment.

Speaking generally, the goal of MLEM is to enable easy and error-safe way to transition ML model from training to serving environment.

## Key features

- **MLEM is not intrusive.** It doesn't ask you to rewrite your training code. Just add two lines to your python script: one to import the library and one to save the model.
- **MLEM turns your Git repository into an easy-to-use model registry.** Have a centralized place to store your models along with all metainformation. You don't need to set up a separate backend server to use it as a model registry.
- **Stick to your workflow.** Use Gitflow or any other Git workflow you like. Because MLEM models are saved as mere artifacts, treat them as any other artifact your produce. Commit metainformation to your repo and store model binaries in any other way you usually do.
- Use your model whatever your like:
    - **Turn your model to a python package** with one command. You find that helpful if you use your model embedded in some other Python application.
    - **Use your model for batch scoring.** You can use MLEM CLI to get predictions for a data file or folder with files. The docker container you build will be capable of this by default.
    - **Turn your model to a REST API application** with Dockerfile prepared with one command. If you like, treat it as a separate git repo or build a Docker container from a model directly.
    - **Deploy your model. MLEM is a provider-agnostic deployment tool.** You don't have to learn new providers when you deploy models to a different cloud or PaaS. MLEM abstracts that for you and simplifies the model deployment tasks. If your provider is not listed yet, you can write a simple plugin to work with MLEM or upvote the issue for creating one.

## Installation

Install MLEM with pip:

```
$ pip install mlem
```

To install the development version, run:

```
$ pip install git+https://github.com/iterative/mlem
```

## Anonymized Usage Analytics

To help us better understand how MLEM is used and improve it, MLEM captures and reports anonymized usage statistics. You will be notified the first time you run `mlem init`.

### What
MLEM's analytics record the following information per event:
- MLEM version (e.g., `0.1.2+5fb5a3.mod`) and OS version (e.g., `MacOS 10.16`)
- Command name and exception type (e.g., `ls, ValueError` or `get, MLEMRootNotFound`)
- Country, city (e.g., `RU, Moscow`)
- A random user_id (generated with [uuid](https://docs.python.org/3/library/uuid.html))

### Implementation
The code is viewable in [analytics.py](https://github.com/iterative/mlem/mlem/analytics.py). They are done in a separate background process and fail fast to avoid delaying any execution. They will fail immediately and silently if you have no network connection.

MLEM's analytics are sent through Iterative's proxy to Google BigQuery over HTTPS.

### Opting out
MLEM analytics help the entire community, so leaving it on is appreciated. However, if you want to opt out of MLEM's analytics, you can disable it via setting an environment variable `MLEM_NO_ANALYTICS=true` or by adding `no_analytics: true` to `.mlem/config.yaml`

This will disable it for the project. We'll add an option to opt out globally soon.
