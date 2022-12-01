import streamlit
import streamlit_pydantic
from pydantic import BaseModel

from mlem.runtime import InterfaceMethod
from mlem.runtime.client import HTTPClient

client = HTTPClient()

method_name = "{{method_name}}"

streamlit.write(method_name)

method: InterfaceMethod = client.methods[method_name]

arg_values = {}
with streamlit.form(key=method_name):
    for arg in method.args:
        arg_model = arg.get_serializer().get_model()
        key = f"{arg.name}_form"
        if issubclass(arg_model, BaseModel):
            arg_values[arg.name] = streamlit_pydantic.pydantic_input(
                key=key, model=arg_model
            )
        else:
            arg_values[arg.name] = streamlit.number_input(
                key=key, label=arg.name, value=0
            )

    submit_button = streamlit.form_submit_button(label="Submit")

if submit_button:
    streamlit.json(arg_values)
    response = getattr(client, method_name)(**arg_values)
    streamlit.write(response)
