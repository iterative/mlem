"""
This code runs in separate process in isolated dir with model and deps to check that we got all of them
"""
from mlem.core.metadata import load

if __name__ == "__main__":
    model = load("model")

    assert model.__class__.__name__ == "TestM"
