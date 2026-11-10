"""Shared TLS configuration for read-only model gateway requests."""

from __future__ import annotations

import ssl


def tls_context() -> ssl.SSLContext:
    """Use certifi when available, otherwise the system certificate store."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
