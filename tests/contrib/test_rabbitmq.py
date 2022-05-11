import time
from threading import Thread

import numpy as np
import pytest
import requests
from pika.exceptions import AMQPError
from requests.exceptions import ConnectionError, HTTPError
from testcontainers.general import TestContainer

from mlem.api import serve
from mlem.contrib.rabbitmq import RabbitMQClient, RabbitMQServer
from tests.conftest import long
from tests.contrib.test_docker.conftest import docker_test

RMQ_PORT = 5672
RMQ_MANAGE_PORT = 15672


@pytest.fixture
def rmq_instance():
    with (
        TestContainer("rabbitmq:3.9-management")
        .with_exposed_ports(RMQ_PORT)
        .with_exposed_ports(RMQ_MANAGE_PORT)
    ) as daemon:
        ready = False
        times = 0
        while not ready and times < 10:
            try:
                r = requests.head(
                    f"http://{daemon.get_container_host_ip()}:{daemon.get_exposed_port(RMQ_MANAGE_PORT)}"
                )
                r.raise_for_status()
                ready = True
            except (HTTPError, ConnectionError):
                time.sleep(0.5)
                times += 1
        time.sleep(1)
        yield daemon


class ServeThread(Thread):
    def __init__(self, model, server):
        super().__init__()
        self.model = model
        self.server = server
        self.dead = True

    def run(self) -> None:
        self.dead = False
        try:
            serve(self.model, self.server)
        finally:
            self.dead = True


@pytest.fixture
def rmq_server(model_meta_saved_single, rmq_instance):
    server = RabbitMQServer(
        host=rmq_instance.get_container_host_ip(),
        port=int(rmq_instance.get_exposed_port(RMQ_PORT)),
        queue_prefix="aaa",
    )
    for _ in range(10):

        t = ServeThread(model_meta_saved_single, server)
        t.start()
        time.sleep(0.5)
        if not t.dead:
            break
        t.join()
    else:
        raise RuntimeError("could not start rmq serving")

    yield server


@long
@docker_test
def test_serving(rmq_server):
    error = None
    for _ in range(20):
        try:
            client = RabbitMQClient(
                host=rmq_server.host,
                port=rmq_server.port,
                queue_prefix=rmq_server.queue_prefix,
            )
            res = client.predict(np.array([[1.0, 1.0, 1.0, 1.0]]))
            assert isinstance(res, np.ndarray)
            break
        except AMQPError as e:
            time.sleep(0.5)
            error = e
    else:
        if error is not None:
            raise error
        pytest.fail("could not connect to server")
