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
from mlem.runtime.client.base import BaseClient
from mlem.runtime.interface.base import InterfaceDescriptor
from mlem.runtime.server.base import Server
from mlem.ui import EMOJI_NAILS, echo

REQUEST = "_request"
RESPONSE = "_response"
INTERFACE = "interface"
# https://www.rabbitmq.com/direct-reply-to.html
REPLY_TO = "amq.rabbitmq.reply-to"


class RabbitMQMixin(BaseModel):
    host: str
    port: int
    exchange: str = ""
    queue_prefix: str = ""
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


class RabbitMQClient(BaseClient, RabbitMQMixin):
    type: ClassVar = "rmq"
    timeout: float = 0

    def _interface_factory(self) -> InterfaceDescriptor:
        res, _, payload = self.channel.basic_get(
            self.queue_prefix + INTERFACE, auto_ack=False
        )
        self.channel.basic_nack(res.delivery_tag)
        return parse_obj_as(InterfaceDescriptor, json.loads(payload))

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
