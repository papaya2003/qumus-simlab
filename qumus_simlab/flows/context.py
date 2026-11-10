"""What a Step gets to work with."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from qumus_simlab.core.clock import Clock
from qumus_simlab.core.errors import StopRequested
from qumus_simlab.perception.base import MicroscopeAnalyzer, ObjectLocator
from qumus_simlab.runtime.store import RunStore
from qumus_simlab.safety.gate import SafetyGate


@dataclass
class FlowContext:
    run_id: str
    store: RunStore
    gate: SafetyGate
    clock: Clock
    locator: ObjectLocator
    bench: dict[str, Any]  # config/bench.yaml
    verification: dict[str, Any]  # config/verification.yaml
    params: dict[str, Any] = field(default_factory=dict)
    project_id: str | None = None
    analyzer: MicroscopeAnalyzer | None = None  # microscope perception (focus, corner, flakes)
    cancel: threading.Event = field(default_factory=threading.Event)

    def check_cancel(self) -> None:
        """Call between device moves; the stop button sets `cancel`. The job
        then pauses with this step in progress (runtime/jobs.py)."""
        if self.cancel.is_set():
            raise StopRequested("stopped by user")
