"""RabbitMQ serving
Extension type: serving

RabbitMQServer implementation
"""
import json
from time import time
from typing import Callable, ClassVar, Optional

import pika
from pika import BasicProperties
from pika.adapters.blocking_connection import BlockingChannel
from pydantic import BaseModel, parse_obj_as

from mlem.core.errors import MlemError
from mlem.core.model import Signature
from mlem.runtime import Interface
from mlem.runtime.client import Client
from mlem.runtime.interface import (
    InterfaceDescriptor,
    VersionedInterfaceDescriptor,
)
from mlem.runtime.server import Server
from mlem.ui import EMOJI_NAILS, echo

REQUEST = "_request"
RESPONSE = "_response"
INTERFACE = "interface"
# https://www.rabbitmq.com/direct-reply-to.html
REPLY_TO = "amq.rabbitmq.reply-to"


class RabbitMQMixin(BaseModel):
    host: str
    """Host of RMQ instance"""
    port: int
    """Port of RMQ instance"""
    exchange: str = ""
    """RMQ exchange to use"""
    queue_prefix: str = ""
    """Queue prefix"""
    channel_cache: Optional[BlockingChannel] = None

    class Config:
        fields = {"channel_cache": {"exclude": True}}
        arbitrary_types_allowed = True

    @property
    def channel(self):
        if self.channel_cache is None:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.host, self.port)
            )
            self.channel_cache = connection.channel()
        return self.channel_cache


class RabbitMQServer(Server, RabbitMQMixin):
    """RMQ server that consumes requests and produces model predictions from/to RMQ instance"""

    type: ClassVar = "rmq"

    def _create_handler(
        self, method_name: str, signature: Signature, executor: Callable
    ):
        serializers = {
            arg.name: arg.type_.get_serializer() for arg in signature.args
        }
        response_serializer = signature.returns.get_serializer()
        echo(EMOJI_NAILS + f"Adding queue for {method_name}")

        def handler(ch, method, props, body):
            data = json.loads(body)
            kwargs = {}
            for a in signature.args:
                kwargs[a.name] = serializers[a.name].deserialize(data[a.name])
            result = executor(**kwargs)
            response = response_serializer.serialize(result)
            echo(data)
            ch.basic_publish(
                exchange="",
                routing_key=props.reply_to,
                properties=pika.BasicProperties(
                    correlation_id=props.correlation_id
                ),
                body=json.dumps(response),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        return handler

    def serve(self, interface: Interface):
        self.channel.queue_declare(self.queue_prefix + INTERFACE)
        schema = json.dumps(interface.get_descriptor().dict())
        self.channel.basic_publish(
            self.exchange, self.queue_prefix + INTERFACE, schema.encode("utf8")
        )

        for method, signature in interface.iter_methods():
            self.channel.queue_declare(self.queue_prefix + method + REQUEST)
            self.channel.basic_consume(
                self.queue_prefix + method + REQUEST,
                on_message_callback=self._create_handler(
                    method, signature, interface.get_method_executor(method)
                ),
            )
            self.channel.queue_declare(self.queue_prefix + method + RESPONSE)

        self.channel.start_consuming()


class RabbitMQClient(Client, RabbitMQMixin):
    """Access models served with rmq server"""

    type: ClassVar = "rmq"
    timeout: float = 0
    """Time to wait for response. 0 means indefinite"""

    def _interface_factory(self) -> InterfaceDescriptor:
        res, _, payload = self.channel.basic_get(
            self.queue_prefix + INTERFACE, auto_ack=False
        )
        self.channel.basic_nack(res.delivery_tag)
        return parse_obj_as(VersionedInterfaceDescriptor, json.loads(payload))

    def _call_method(self, name, args):
        response = None

        def consume(ch, mt, pr, body):  # pylint: disable=unused-argument
            nonlocal response
            response = body

        self.channel.basic_consume(REPLY_TO, consume, auto_ack=True)
        self.channel.basic_publish(
            self.exchange,
            self.queue_prefix + name + REQUEST,
            json.dumps(args).encode("utf8"),
            BasicProperties(reply_to=REPLY_TO),
        )
        start = time()
        while response is None:
            self.channel.connection.process_data_events()
            if 0 < self.timeout < time() - start:
                raise MlemError("Request timed out")
        return json.loads(response)
