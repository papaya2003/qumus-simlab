"""Clock abstraction so the testbench can fast-forward time (spec 11.2).

All flow, safety and runtime code must take time from a Clock, never from
time.time() / time.sleep() directly.

Clocks may explicitly declare event_clock_basis and event_clock_domain for
saved evidence comparison. A domain's latest writer selects its current basis;
an undeclared custom clock remains valid for execution but has unknown freshness.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4


class Clock(Protocol):
    def now(self) -> float:
        """Seconds since the epoch (may be simulated)."""

    def sleep(self, seconds: float) -> None: ...


class RealClock:
    event_clock_basis = "utc"
    event_clock_domain = "utc"

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimClock:
    """Simulated clock.

    speed=0 means fully manual: sleep() advances simulated time instantly.
    speed=N>0 means simulated time runs N times faster than wall time and
    sleep() really waits seconds/N (useful for demos).
    """

    def __init__(self, start: float = 1_800_000_000.0, speed: float = 0.0) -> None:
        # Independent/restarted simulations do not share an observation epoch.
        self.event_clock_basis = "simulation:" + uuid4().hex
        self.event_clock_domain = "simulation"
        self._t = start
        self._speed = speed
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += max(0.0, seconds)

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._speed > 0:
            time.sleep(seconds / self._speed)
        self.advance(seconds)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")
