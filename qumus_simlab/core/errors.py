"""Structured errors (spec 4.5, 5.2).

Hardware and flow code never returns "❌ ..." strings. It raises one of these,
and the category decides how the job service reacts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCategory(StrEnum):
    TRANSIENT = "transient"  # serial timeout, dropped frame, rate limit -> retry with backoff
    CORRECTABLE = "correctable"  # placed askew, out of focus -> correction ladder (spec 4.3)
    NEEDS_HUMAN = "needs_human"  # out of material, broken stamp -> park job, notify
    DANGEROUS = "dangerous"  # limit exceeded, runaway heater -> safe state, no auto-recovery


class QumusError(Exception):
    category: ErrorCategory = ErrorCategory.NEEDS_HUMAN

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "category": str(self.category),
            "message": self.message,
            "details": self.details,
        }


class TransientError(QumusError):
    category = ErrorCategory.TRANSIENT


class CorrectableError(QumusError):
    category = ErrorCategory.CORRECTABLE


class NeedsHumanError(QumusError):
    category = ErrorCategory.NEEDS_HUMAN


class DangerousError(QumusError):
    category = ErrorCategory.DANGEROUS


class SafetyViolation(DangerousError):
    """A device command was refused by the SafetyGate (spec 7.2)."""


class StopRequested(QumusError):
    """The person pressed Stop: the step ends at this safe point, still in
    progress, and the job pauses there (runtime/jobs.py). Not a failure: the
    StepRunner lets it through and Continue resumes the step like a restart."""


class PreconditionFailed(QumusError):
    """A step's preconditions do not hold; nothing was moved (spec 4.2)."""

    category = ErrorCategory.CORRECTABLE


class DeviceUnavailable(TransientError):
    """A single device is offline. Only steps that use it are affected (spec 5.4)."""
