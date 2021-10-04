MLEM is in early alpha. Thank you for trying it out! 👋

Alpha include model registry functionality, and upcoming beta will add model deployment functionality.

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

```bash
% pip install mlem
```

To install the development version, run:

```bash
% pip install git+git://github.com/iterative/mlem
```
