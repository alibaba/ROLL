import threading
import time
import weakref
from typing import Any, Callable, Optional

from roll.utils.logging import get_logger

logger = get_logger()


class RuntaLease:
    """
    Renewable server-side lease that keeps a Runta runtime alive only while its owner is.

    ROLL stops its workers with ``ray stop --force`` (SIGKILL), which no in-process hook
    survives, so the runtime itself has to notice that its worker is gone. It is created
    with a short ``auto_shutdown_after_secs`` (``initial_seconds``) and a daemon thread
    re-arms that clock every ``lease_seconds / 3`` through
    ``runtime.update_timeout_policies``: Runta restarts a policy's clock whenever the
    policy is saved. A killed worker stops renewing, and Runta shuts the runtime down
    within one lease.

    ``max_lifetime_seconds`` stays the hard cap. It is counted from the moment the lease
    object is built (build it right before the create request): a renewal never asks for
    more than the lifetime that is left, and renewing stops once none is left.

    The heartbeat talks to Runta through a client and a runtime handle of its own. The
    SDK does not document its sync client as thread-safe and a ``Runtime`` handle
    rewrites its cached metadata on every call, so nothing is shared with the thread
    that executes commands, and no lock of the owner is ever taken. A renewal that fails
    is logged and attempted again on the next beat; the heartbeat's client does not
    retry on its own, so one bad request cannot outlast the beat interval.
    """

    def __init__(
        self,
        lease_seconds: int,
        max_lifetime_seconds: int,
        endpoint: Optional[str] = None,
        owner: Optional[object] = None,
        interval: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            lease_seconds: Seconds the runtime survives without a renewal.
            max_lifetime_seconds: Hard cap on the runtime's lifetime, renewals included.
            endpoint: Runta API endpoint; defaults to ``RUNTA_ENDPOINT`` or the SDK default.
            owner: Object the runtime belongs to. It is only weakly referenced, and the
                heartbeat stops when it is garbage collected.
            interval: Seconds between renewals; defaults to ``lease_seconds / 3``.
            clock: Monotonic clock, replaceable in tests.
        """
        if lease_seconds < 1:
            # Runta reads 0 as "never shut down", which would bring the leak back.
            raise ValueError(f"lease_seconds must be at least 1, got {lease_seconds}")
        self.lease_seconds = lease_seconds
        self.max_lifetime_seconds = max_lifetime_seconds
        self.endpoint = endpoint
        self.interval = lease_seconds / 3 if interval is None else interval
        self.runtime_id: Optional[str] = None

        self._clock = clock
        self._expires_at = clock() + max_lifetime_seconds
        self._owner = weakref.ref(owner) if owner is not None else None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._runtime: Any = None

    @property
    def initial_seconds(self) -> int:
        """``auto_shutdown_after_secs`` to create the runtime with."""
        return min(self.lease_seconds, self.max_lifetime_seconds)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def start(self, runtime_id: str) -> None:
        """Starts renewing the lease of ``runtime_id`` in a daemon thread."""
        self.runtime_id = runtime_id
        self._thread = threading.Thread(target=self._run, name=f"runta-lease-{runtime_id[:8]}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops renewing. Never blocks: a renewal in flight finishes on its own client."""
        self._stop.set()
        if self._thread is None or not self._thread.is_alive():
            self._close_client()

    def join(self, timeout: Optional[float] = None) -> bool:
        """Waits for the heartbeat thread to end and tells whether it did."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def beat(self) -> bool:
        """Renews the lease once. Returns False when it must not be renewed again."""
        if self._stop.is_set() or (self._owner is not None and self._owner() is None):
            return False

        # Rounded down, so the runtime never outlives the cap. Below one second there is
        # nothing left to ask for, and 0 would switch the shutdown policy off.
        seconds = int(min(self.lease_seconds, self._expires_at - self._clock()))
        if seconds < 1:
            logger.info(
                "Runta runtime %s used up its max_lifetime_seconds (%d); its lease is no longer renewed.",
                self.runtime_id,
                self.max_lifetime_seconds,
            )
            return False

        try:
            self._handle().update_timeout_policies(auto_shutdown_after_secs=seconds)
        except Exception as e:
            # A runtime deleted by close() while this renewal was in flight is not a failure.
            if self._stop.is_set():
                return False
            logger.warning(
                "Could not renew the lease of Runta runtime %s (%s: %s); retrying in %.0f seconds.",
                self.runtime_id,
                type(e).__name__,
                e,
                self.interval,
            )
        return True

    def _handle(self):
        """The heartbeat's own handle of the runtime, on the heartbeat's own client."""
        if self._runtime is None:
            from runta import Runta

            if self._client is None:
                kwargs = {"endpoint": self.endpoint} if self.endpoint else {}
                self._client = Runta(retry_max_attempts=1, **kwargs)
            self._runtime = self._client.runtimes.get(self.runtime_id)
        return self._runtime

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                if not self.beat():
                    break
        except Exception:
            logger.exception("The lease heartbeat of Runta runtime %s stopped unexpectedly.", self.runtime_id)
        finally:
            self._close_client()

    def _close_client(self) -> None:
        client, self._client, self._runtime = self._client, None, None
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Failed to close the lease client of Runta runtime {self.runtime_id}: {e}")
