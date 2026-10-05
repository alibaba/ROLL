import errno
import os
import random
import socket


def get_node_ip():
    import ray

    if ray.is_initialized():
        # Use the address Ray actually bound, including on isolated networks
        # and hosts with multiple interfaces. This needs no external route.
        return ray.util.get_node_ip_address()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def collect_free_port():
    """Find a listener port, optionally outside the host's ephemeral range.

    ROLL_PORT_RANGE=start:end includes both endpoints. It is useful when a
    rendezvous listener starts long after discovery: an outbound connection
    can otherwise claim the unreserved ephemeral port in the meantime. The
    caller still owns coordination with other listener processes.
    """
    configured = os.environ.get("ROLL_PORT_RANGE")
    if configured is None:
        candidates = [0]
    else:
        try:
            start, end = map(int, configured.split(":"))
            if not 1 <= start <= end <= 65535:
                raise ValueError
        except ValueError as error:
            raise ValueError("ROLL_PORT_RANGE must be start:end with 1 <= start <= end <= 65535") from error
        candidates = list(range(start, end + 1))
        # Port allocation must not consume the training sampler's RNG state.
        random.SystemRandom().shuffle(candidates)
    for port in candidates:
        with socket.socket() as sock:
            try:
                sock.bind(("", port))
            except OSError as error:
                if configured is None or error.errno not in (errno.EADDRINUSE, errno.EACCES):
                    raise
                continue
            return sock.getsockname()[1]
    raise RuntimeError(f"No available port in ROLL_PORT_RANGE={configured}")
