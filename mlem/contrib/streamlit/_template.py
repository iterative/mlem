import streamlit
import streamlit_pydantic
from pydantic import BaseModel

from mlem.contrib.streamlit.server import augment_model
from mlem.runtime import InterfaceMethod
from mlem.runtime.client import HTTPClient

streamlit.title(
    "MLEM Streamlit UI for server at {{server_host}}:{{server_port}}"
)


@streamlit.cache(hash_funcs={HTTPClient: lambda x: 0})
def get_client():
    return HTTPClient(
        host="{{server_host}}", port=int("{{server_port}}"), raw=True
    )


client = get_client()

methods = list(client.methods.keys())
tabs = streamlit.tabs(methods)

for method_name, tab in zip(methods, tabs):

    with tab:
        tab.header(method_name)
        method: InterfaceMethod = client.methods[method_name]

        arg_values = {}
        with streamlit.form(key=method_name):
            for arg in method.args:
                serializer = arg.get_serializer()
                if serializer.serializer.is_binary:
                    arg_values[arg.name] = streamlit.file_uploader(arg.name)
                else:
                    arg_model = serializer.get_model()
                    augment, arg_model_aug = augment_model(arg_model)
                    if arg_model_aug is None:
                        streamlit.write(
                            "Model with lists/arrays are not supported"
                        )
                    else:
                        arg_model = arg_model_aug
                        key = f"{method_name}_{arg.name}_form"
                        if issubclass(arg_model, BaseModel):
                            arg_values[arg.name] = augment(
                                streamlit_pydantic.pydantic_input(
                                    key=key, model=arg_model
                                )
                            )
                        else:
                            if arg_model in (int, float):
                                arg_input = streamlit.number_input(
                                    key=key, label=arg.name, value=0
                                )
                            elif arg_model in (str, complex):
                                arg_input = streamlit.text_input(
                                    key=key, label=arg.name
                                )
                            elif arg_model is bool:
                                arg_input = streamlit.checkbox(
                                    key=key, label=arg.name
                                )
                            else:
                                arg_input = None
                            arg_values[arg.name] = augment(arg_input)

            submit_button = streamlit.form_submit_button(label="Submit")

            if submit_button:
                response = getattr(client, method_name)(
                    **{
                        k: v.dict() if isinstance(v, BaseModel) else v
                        for k, v in arg_values.items()
                    }
                )
                if method.returns.get_serializer().serializer.is_binary:
                    pass
                else:
                    streamlit.write("Response:")
                    streamlit.write(response)
        if (
            submit_button
            and method.returns.get_serializer().serializer.is_binary
        ):
            download_button = streamlit.download_button(
                label="Download", data=response
            )
