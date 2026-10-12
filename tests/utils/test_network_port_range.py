"""A configured listener range avoids delayed rendezvous using ephemeral ports."""
import socket

import pytest

from roll.utils.network_utils import collect_free_port


def test_configured_port_is_used_and_busy_range_fails(monkeypatch):
    # Select one real unused port, then reserve it while repeating discovery.
    with socket.socket() as available:
        available.bind(("", 0))
        port = available.getsockname()[1]
    monkeypatch.setenv("ROLL_PORT_RANGE", f"{port}:{port}")
    assert collect_free_port() == port
    with socket.socket() as occupied:
        occupied.bind(("", port))
        occupied.listen()
        with pytest.raises(RuntimeError, match="No available port"):
            collect_free_port()


@pytest.mark.parametrize("value", ["", "20000", "abc:30000", "0:5", "30000:20000", "65000:65536"])
def test_invalid_range_fails_explicitly(monkeypatch, value):
    monkeypatch.setenv("ROLL_PORT_RANGE", value)
    with pytest.raises(ValueError, match="ROLL_PORT_RANGE"):
        collect_free_port()
