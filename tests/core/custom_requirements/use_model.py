"""
This code runs in separate process in isolated dir with model and deps to check that we got all of them
"""
import dill

if __name__ == "__main__":
    with open("model.pkl", "rb") as f:
        model = dill.load(f)

    model(1)
