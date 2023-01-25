import streamlit

from mlem.contrib.streamlit.utils import model_form
from mlem.runtime.client import HTTPClient

streamlit.title("MLEM Streamlit UI")


@streamlit.cache(hash_funcs={HTTPClient: lambda x: 0})
def get_client():
    return HTTPClient(
        host="{{server_host}}", port=int("{{server_port}}"), raw=True
    )


client = get_client()
model_form(client)

streamlit.write(
    "Built for FastAPI server at `{{server_host}}:{{server_port}}`. Docs: https://mlem.ai/doc"
)
