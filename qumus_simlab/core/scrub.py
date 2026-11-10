"""Remove keys and home paths from text that leaves the process (recordings,
error messages shown in the UI)."""

from __future__ import annotations

import os
import re

SECRET_RES = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|auth[_-]?token)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{12,}"),
]
HOME_RE = re.compile(r"(/Users|/home)/[^/\s\"']+/")
SECRET_ENV = ("QUMUS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
              "QUMUS_OPENAI_API_KEY", "OPENAI_API_KEY", "TOKEN_STATION_API_KEY")


def env_secrets() -> list[str]:
    """Values of the key variables set in this process (long enough to be keys)."""
    return [v for k in SECRET_ENV if len(v := os.environ.get(k, "")) >= 8]


def scrub_text(s: str, secrets: list[str] | None = None) -> str:
    for v in env_secrets() if secrets is None else secrets:
        s = s.replace(v, "[redacted]")
    for r in SECRET_RES:
        s = r.sub("[redacted]", s)
    return HOME_RE.sub("~/", s)
