"""
Kubernetes Service + Ingress backend for rollout server registration.

Each call to register_servers creates three resources in the ``xdl`` namespace:

* Headless Service  — provides a stable DNS name for the backend
* Endpoints         — manually points to the rollout worker Pod IPs
* Ingress           — nginx path rule ``/{job_id}/{port}(/|$)(.*)``
                      rewritten to ``/$2`` and forwarded to the Service

Cleanup is label-based: all resources share the label ``roll.ai/job-id``
so they can be found and deleted without tracking instance state.
"""
import hashlib
import logging
import os
import re
from typing import List, Optional, Tuple

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .base import ProxyRouter
from roll.utils.logging import get_logger

logger = get_logger()

NAMESPACE = "xdl"
LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_JOB_ID = "roll.ai/job-id"
LABEL_PORT = "roll.ai/port"


def _safe_k8s_name(job_id: str, port: int) -> str:
    """Return a DNS-label-safe K8s resource name (max 63 chars)."""
    raw = f"roll-{job_id}-{port}"
    if len(raw) <= 63:
        return raw
    # Truncate and append a short hash to keep uniqueness
    h = hashlib.sha256(f"{job_id}:{port}".encode()).hexdigest()[:8]
    return f"roll-{job_id}"[:50] + f"-{h}"


def _parse_address(address: str) -> Tuple[Optional[str], Optional[int]]:
    match = re.match(r'https?://([^:/]+):(\d+)', address)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


class IngressProxyRouter(ProxyRouter):
    """Kubernetes Service + nginx Ingress backend.

    Args:
        namespace: Kubernetes namespace (default: ``roll``). Created automatically
            if it does not exist.
        ingress_class_name: Ingress controller class (default: ``nginx``).
            Written to ``spec.ingressClassName`` as recommended since K8s 1.18.
        tls_secret_name: Name of the TLS Secret for HTTPS. When set, ``spec.tls``
            is populated so the Ingress terminates TLS. Leave ``None`` for plain HTTP.
    """

    def __init__(
        self,
        namespace: str = NAMESPACE,
        ingress_class_name: str = "nginx",
        tls_secret_name: str = None,
    ):
        self.namespace = namespace
        self.ingress_host = os.environ.get('EP_ENDPOINT')
        self.ingress_class_name = ingress_class_name
        self.tls_secret_name = tls_secret_name

        config.load_incluster_config()
        # The cluster's API server certificate may not cover the internal service IP,
        # so disable SSL verification for in-cluster communication.
        configuration = client.Configuration.get_default_copy()
        configuration.verify_ssl = False
        client.Configuration.set_default(configuration)
        self._core = client.CoreV1Api()
        self._networking = client.NetworkingV1Api()

        self._ensure_namespace()

    # ------------------------------------------------------------------
    # ProxyRouter interface
    # ------------------------------------------------------------------

    def register_servers(self, addresses: List[str], job_id: str, port: int) -> dict:
        """Create Service, Endpoints, and Ingress for the given rollout workers.

        Args:
            addresses: Worker addresses, e.g. ``["http://10.0.0.1:8080"]``
            job_id: Unique job identifier used in resource names and Ingress path
            port: Backend port number

        Returns:
            Dict with ``service``, ``ingress``, ``namespace``, and ``registered_servers``
        """
        name = _safe_k8s_name(job_id, port)
        labels = {
            LABEL_MANAGED_BY: "roll",
            LABEL_JOB_ID: job_id,
            LABEL_PORT: str(port),
        }
        self._ensure_service(name, labels, port)
        self._ensure_endpoints(name, labels, addresses, port)
        self._ensure_ingress(name, labels, job_id, port)
        logger.info("Registered %d servers: service/ingress=%s namespace=%s", len(addresses), name, self.namespace)
        return {
            "service": name,
            "ingress": name,
            "namespace": self.namespace,
            "registered_servers": addresses,
        }

    def list_active_job_ids(self) -> List[str]:
        """Return job IDs read from the ``roll.ai/job-id`` label on active Ingress objects.

        Lists all Ingress resources managed by roll in this namespace and
        extracts unique job IDs from their labels.
        """
        label_selector = f"{LABEL_MANAGED_BY}=roll"
        try:
            items = self._networking.list_namespaced_ingress(
                self.namespace, label_selector=label_selector
            ).items
        except ApiException as e:
            logger.error("Failed to list Ingress objects for active job IDs: %s", e)
            return []

        job_ids = []
        for item in items:
            job_id = (item.metadata.labels or {}).get(LABEL_JOB_ID)
            if job_id:
                job_ids.append(job_id)
        return job_ids

    def get_callback_url(self, job_id: str, port: int) -> Optional[str]:
        """Build the K8s Ingress callback URL for agents to reach the rollout server."""
        if not self.ingress_host:
            return None
        return f"http://{self.ingress_host}/{job_id}/{port}/v1"

    def cleanup_resources(self, job_id: str) -> dict:
        """Delete all Ingress, Endpoints, and Service objects labelled with job_id.

        Deletion order: Ingress → Endpoints → Service, so routing rules are
        removed before the backend disappears.

        Args:
            job_id: Job whose resources should be removed

        Returns:
            Dict with ``cleaned_up`` count and ``errors`` list
        """
        label_selector = f"{LABEL_JOB_ID}={job_id}"
        cleaned, errors = 0, []

        # (list_fn, delete_fn, kind) — order matters
        steps = [
            (
                lambda: self._networking.list_namespaced_ingress(self.namespace, label_selector=label_selector),
                lambda n: self._networking.delete_namespaced_ingress(n, self.namespace),
                "Ingress",
            ),
            (
                lambda: self._core.list_namespaced_endpoints(self.namespace, label_selector=label_selector),
                lambda n: self._core.delete_namespaced_endpoints(n, self.namespace),
                "Endpoints",
            ),
            (
                lambda: self._core.list_namespaced_service(self.namespace, label_selector=label_selector),
                lambda n: self._core.delete_namespaced_service(n, self.namespace),
                "Service",
            ),
        ]

        for list_fn, delete_fn, kind in steps:
            try:
                for item in list_fn().items:
                    name = item.metadata.name
                    try:
                        delete_fn(name)
                        logger.info("Deleted %s: %s", kind, name)
                        cleaned += 1
                    except ApiException as e:
                        if e.status == 404:
                            logger.info("%s/%s already gone", kind, name)
                        else:
                            errors.append(f"{kind}/{name}: {e}")
            except ApiException as e:
                errors.append(f"List {kind}: {e}")

        return {"cleaned_up": cleaned, "errors": errors}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_namespace(self) -> None:
        """Create the namespace if it does not already exist.

        Namespace is a cluster-scoped resource. If the ServiceAccount only has
        namespace-scoped permissions (Role/RoleBinding), read_namespace and
        create_namespace will raise 403. In that case we log a warning and
        continue — the namespace must have been pre-created by an admin.
        """
        try:
            self._core.read_namespace(self.namespace)
        except ApiException as e:
            if e.status == 404:
                try:
                    ns = client.V1Namespace(
                        metadata=client.V1ObjectMeta(
                            name=self.namespace,
                            labels={LABEL_MANAGED_BY: "roll"},
                        )
                    )
                    self._core.create_namespace(ns)
                    logger.info("Created namespace: %s", self.namespace)
                except ApiException as ce:
                    if ce.status == 403:
                        logger.warning(
                            "No permission to create namespace %s (ClusterRole required). "
                            "Assuming it was pre-created by an admin.",
                            self.namespace,
                        )
                    else:
                        raise
            elif e.status == 403:
                logger.warning(
                    "No permission to read namespace %s (ClusterRole required). "
                    "Assuming it exists and was pre-created by an admin.",
                    self.namespace,
                )
            else:
                raise

    def _ensure_service(self, name: str, labels: dict, port: int) -> None:
        svc = client.V1Service(
            metadata=client.V1ObjectMeta(name=name, namespace=self.namespace, labels=labels),
            spec=client.V1ServiceSpec(
                cluster_ip="None",  # headless — no kube-proxy load balancing
                ports=[client.V1ServicePort(port=port, protocol="TCP")],
            ),
        )
        try:
            self._core.create_namespaced_service(self.namespace, svc)
            logger.info("Created Service: %s", name)
        except ApiException as e:
            if e.status == 409:
                logger.warning("Service %s already exists, skipping", name)
            else:
                raise

    def _ensure_endpoints(self, name: str, labels: dict, addresses: List[str], port: int) -> None:
        endpoint_addrs = []
        for addr in addresses:
            ip, _ = _parse_address(addr)
            if ip:
                endpoint_addrs.append(client.V1EndpointAddress(ip=ip))
            else:
                logger.warning("Cannot parse address %s, skipping", addr)

        if not endpoint_addrs:
            raise ValueError(f"No valid addresses parsed from: {addresses}")

        ep = client.V1Endpoints(
            metadata=client.V1ObjectMeta(name=name, namespace=self.namespace, labels=labels),
            subsets=[
                client.V1EndpointSubset(
                    addresses=endpoint_addrs,
                    ports=[client.CoreV1EndpointPort(port=port, protocol="TCP")],
                )
            ],
        )
        try:
            self._core.create_namespaced_endpoints(self.namespace, ep)
            logger.info("Created Endpoints: %s (%d addresses)", name, len(endpoint_addrs))
        except ApiException as e:
            if e.status == 409:
                logger.warning("Endpoints %s already exists, patching", name)
                self._core.patch_namespaced_endpoints(name, self.namespace, ep)
            else:
                raise

    def _ensure_ingress(self, name: str, labels: dict, job_id: str, port: int) -> None:
        # Escape job_id so special regex chars (e.g. '.') don't break the pattern.
        # K8s names only contain [a-z0-9-] but the raw job_id passed by the caller
        # may not have been sanitised yet.
        safe_job_id = re.escape(job_id)
        # Capture group $2 holds the remainder after /{job_id}/{port}/
        path_pattern = f"/{safe_job_id}/{port}(/|$)(.*)"

        tls = (
            [client.V1IngressTLS(hosts=[self.ingress_host] if self.ingress_host else [], secret_name=self.tls_secret_name)]
            if self.tls_secret_name
            else None
        )

        ingress = client.V1Ingress(
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                labels=labels,
                annotations={
                    "nginx.ingress.kubernetes.io/use-regex": "true",
                    "nginx.ingress.kubernetes.io/rewrite-target": "/$2",
                },
            ),
            spec=client.V1IngressSpec(
                ingress_class_name=self.ingress_class_name,
                tls=tls,
                rules=[
                    client.V1IngressRule(
                        host=self.ingress_host,
                        http=client.V1HTTPIngressRuleValue(
                            paths=[
                                client.V1HTTPIngressPath(
                                    path=path_pattern,
                                    path_type="ImplementationSpecific",
                                    backend=client.V1IngressBackend(
                                        service=client.V1IngressServiceBackend(
                                            name=name,
                                            port=client.V1ServiceBackendPort(number=port),
                                        )
                                    ),
                                )
                            ]
                        ),
                    )
                ],
            ),
        )
        try:
            self._networking.create_namespaced_ingress(self.namespace, ingress)
            logger.info("Created Ingress: %s host=%s path=%s", name, self.ingress_host, path_pattern)
        except ApiException as e:
            if e.status == 409:
                logger.warning("Ingress %s already exists, patching", name)
                self._networking.patch_namespaced_ingress(name, self.namespace, ingress)
            else:
                raise


if __name__ == "__main__":
    import sys
    import socket

    os.environ.setdefault('EP_ENDPOINT', '')
    os.environ.setdefault('TASK_ID', 'ingress-test-123')

    # 用本机 IP 作为测试 Endpoints 地址
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    test_addresses = [f"http://{local_ip}:19876"]
    job_id = os.environ['TASK_ID']

    print(f"Testing IngressProxyRouter: addresses={test_addresses} job_id={job_id}")

    try:
        backend = IngressProxyRouter()
        result = backend.register_servers(addresses=test_addresses, job_id=job_id, port=19876)
        print(f"register_servers result: {result}")
        url = backend.get_callback_url(job_id, 19876)
        print(f"callback_url: {url}")
        cleanup = backend.cleanup_resources(job_id)
        print(f"cleanup result: {cleanup}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
