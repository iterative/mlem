import streamlit

from mlem.contrib.streamlit.utils import model_form
from mlem.runtime.client import HTTPClient

streamlit.set_page_config(
    page_title="{{page_title}}",
)


@streamlit.cache_resource
def get_client():
    return HTTPClient(
        host="{{server_host}}", port=int("{{server_port}}"), raw=True
    )


streamlit.title("{{title}}")
streamlit.write("""{{description}}""")
model_form(get_client())
streamlit.markdown("---")
streamlit.write(
    "Built for FastAPI server at `{{server_host}}:{{server_port}}`. Docs: https://mlem.ai/doc"
)
