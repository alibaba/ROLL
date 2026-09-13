import os
from abc import ABC, abstractmethod
from typing import List, Optional

from roll.utils.logging import get_logger

logger = get_logger()


def batch_cleanup(backend: "ProxyRouter") -> dict:
    """Clean up resources for all jobs registered in the backend.

    Args:
        backend: Any ProxyRouter implementation (ALB or Ingress)

    Returns:
        Dict with ``cleaned_up`` (int), ``errors`` (list), and
        ``checked_jobs`` (list of job IDs that were evaluated)
    """
    job_ids = list(dict.fromkeys(backend.list_active_job_ids()))  # deduplicate, preserve order
    logger.info(f"Found {len(job_ids)} job(s) to clean up: {job_ids}")

    cleaned, errors, checked = 0, [], []

    for job_id in job_ids:
        checked.append(job_id)
        logger.info(f"Cleaning up resources for job {job_id}")
        result = backend.cleanup_resources(job_id)
        cleaned += result.get('cleaned_up', 0)
        errors.extend(result.get('errors', []))

    return {'cleaned_up': cleaned, 'errors': errors, 'checked_jobs': checked}


class ProxyRouter(ABC):
    """Abstract base class for proxy router registration.

    Implementations register rollout server IP:port backends to a load balancer
    (ALB, Ingress, etc.) and clean them up when the job finishes.
    """

    @abstractmethod
    def register_servers(self, addresses: List[str], job_id: str, port: int) -> dict:
        """Register rollout servers to the load balancer backend.

        Args:
            addresses: Server addresses, e.g. ["http://10.0.0.1:8080"]
            job_id: Unique job identifier used for resource naming and routing
            port: Backend port number

        Returns:
            Dict describing created resources
        """
        ...

    @abstractmethod
    def cleanup_resources(self, job_id: str) -> dict:
        """Clean up all backend resources created for a job.

        Args:
            job_id: Job identifier whose resources should be removed

        Returns:
            Dict with keys ``cleaned_up`` (int) and ``errors`` (list)
        """
        ...

    @abstractmethod
    def list_active_job_ids(self) -> List[str]:
        """Return job IDs that currently have registered resources in this backend.

        Used by batch cleanup to discover which jobs still have live resources,
        so their status can be checked and stale ones cleaned up.

        Returns:
            List of job ID strings (may contain duplicates if multiple ports
            are registered per job; callers should deduplicate if needed)
        """
        ...

    @abstractmethod
    def get_callback_url(self, job_id: str, port: int) -> Optional[str]:
        """Return the URL that agents should use to reach the rollout server.

        Args:
            job_id: Job identifier whose resources are already registered
            port: Backend port number

        Returns:
            Full callback URL string, or None if the URL cannot be determined
        """
        ...


def create_proxy_router() -> "ProxyRouter":
    """Instantiate the proxy router selected by the PROXY_ROUTER_TYPE env var.

    Supported values: ``ingress`` (Kubernetes nginx Ingress), ``alb`` (default, Alibaba Cloud ALB).
    """
    if os.environ.get('PROXY_ROUTER_TYPE') == 'ingress':
        from .ingress_proxy_router import IngressProxyRouter
        logger.info("[ProxyRouter] Using Ingress router")
        return IngressProxyRouter()
    logger.info("[ProxyRouter] Using ALB router")
    from .alb_proxy_router import ALBProxyRouter
    return ALBProxyRouter()
