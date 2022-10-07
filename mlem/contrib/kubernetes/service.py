from abc import abstractmethod
from typing import ClassVar, Optional, Tuple

from kubernetes import client

from mlem.core.base import MlemABC
from mlem.core.errors import EndpointNotFound, MlemError


def find_index(nodes_list, node_name):
    for i, each_node in enumerate(nodes_list):
        if each_node.metadata.name == node_name:
            return i
    return -1


class ServiceType(MlemABC):
    """Service Type for services inside a Kubernetes Cluster"""

    abs_name: ClassVar = "k8s_service_type"

    class Config:
        type_root = True

    @abstractmethod
    def get_string(self):
        raise NotImplementedError

    @abstractmethod
    def get_host_and_port(
        self, service, namespace="mlem"  # pylint: disable=unused-argument
    ) -> Tuple[Optional[str], Optional[int]]:
        """Returns host and port for the service in Kubernetes"""
        raise NotImplementedError


class NodePortService(ServiceType):
    """NodePort Service implementation for service inside a Kubernetes Cluster"""

    type: ClassVar = "nodeport"

    def get_string(self):
        return "NodePort"

    def get_host_and_port(self, service, namespace="mlem"):
        try:
            port = service.items[0].spec.ports[0].node_port
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine node port of the deployed service"
            ) from e
        try:
            node_name = (
                client.CoreV1Api()
                .list_namespaced_pod(namespace)
                .items[0]
                .spec.node_name
            )
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine name of the node where the pod is deployed"
            ) from e
        node_list = client.CoreV1Api().list_node().items
        node_index = find_index(node_list, node_name)
        if node_index == -1:
            raise MlemError(
                f"Couldn't find the node where pods in namespace {namespace} exists"
            )
        address_dict = node_list[node_index].status.addresses
        for each_address in address_dict:
            if each_address.type == "ExternalIP":
                host = each_address.address
                return host, port
        raise EndpointNotFound(
            f"Node {node_name} doesn't have an externally reachable IP address"
        )


class LoadBalancerService(ServiceType):
    """LoadBalancer Service implementation for service inside a Kubernetes Cluster"""

    type: ClassVar = "loadbalancer"

    def get_string(self):
        return "LoadBalancer"

    def get_host_and_port(self, service, namespace="mlem"):
        try:
            port = service.items[0].spec.ports[0].port
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine port of the deployed service"
            ) from e
        try:
            ingress = service.items[0].status.load_balancer.ingress[0]
            host = ingress.hostname or ingress.ip
        except (IndexError, AttributeError) as e:
            raise MlemError(
                "Couldn't determine IP address of the deployed service"
            ) from e
        return host, port


class ClusterIPService(ServiceType):
    """ClusterIP Service implementation for service inside a Kubernetes Cluster"""

    type: ClassVar = "clusterip"

    def get_string(self):
        return "ClusterIP"

    def get_host_and_port(self, service, namespace="mlem"):
        raise MlemError(
            "Cannot expose service of type ClusterIP outside the Kubernetes Cluster"
        )
