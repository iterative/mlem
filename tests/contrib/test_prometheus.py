from mlem.contrib.fastapi import FastAPIServer, Middlewares
from mlem.core.objects import MlemModel
from mlem.runtime.client import Client
from mlem.runtime.interface import ModelInterface
from mlem.runtime.server import ServerInterface


def test_prometheus_fastapi_middleware(create_mlem_client, create_client):
    from mlem.contrib.prometheus import PrometheusFastAPIMiddleware

    model = MlemModel.from_obj(lambda x: x, sample_data=10)
    model_interface = ModelInterface.from_model(model)

    server = FastAPIServer(
        standardize=True,
        middlewares=Middlewares(__root__=[PrometheusFastAPIMiddleware()]),
    )
    interface = ServerInterface.create(server, model_interface)
    client = create_client(server, interface)

    docs = client.get("/openapi.json")
    assert docs.status_code == 200, docs.json()

    # metrics = client.get("/metrics")
    # assert metrics.status_code == 200, metrics
    # assert metrics.text

    mlem_client: Client = create_mlem_client(client)
    remote_interface = mlem_client.interface
    dt = remote_interface.__root__["predict"].args[0].data_type
    response = client.post("/predict", json={"data": dt.serialize(1)})
    assert response.status_code == 200
    resp = remote_interface.__root__["predict"].returns.data_type.deserialize(
        response.json()
    )
    assert resp == 1
