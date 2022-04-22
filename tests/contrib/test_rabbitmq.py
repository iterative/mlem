import time
from threading import Thread

import numpy as np
import pytest
from testcontainers.general import TestContainer

from mlem.api import serve
from mlem.contrib.rabbitmq import RabbitMQClient, RabbitMQServer
from tests.conftest import long

RMQ_PORT = 5672
RMQ_MANAGE_PORT = 15672


@pytest.fixture
def rmq_instance():
    with (
        TestContainer("rabbitmq:3.9-management")
        .with_exposed_ports(RMQ_PORT)
        .with_bind_ports(RMQ_MANAGE_PORT, RMQ_MANAGE_PORT)
    ) as daemon:
        time.sleep(5)
        yield daemon


@long
def test_serving(model_meta_saved_single, rmq_instance):
    server = RabbitMQServer(
        host=rmq_instance.get_container_host_ip(),
        port=int(rmq_instance.get_exposed_port(RMQ_PORT)),
        queue_prefix="aaa",
    )
    t = Thread(target=lambda: serve(model_meta_saved_single, server))
    t.start()
    time.sleep(1)
    client = RabbitMQClient(
        host=server.host, port=server.port, queue_prefix=server.queue_prefix
    )

    res = client.predict(np.array([[1.0, 1.0, 1.0, 1.0]]))
    assert isinstance(res, np.ndarray)
