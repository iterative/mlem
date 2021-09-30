import click


@click.group()
def cli():
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialise any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment
    """
