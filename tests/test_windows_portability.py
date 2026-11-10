"""qumus-simlab must start and run on Windows (the lab PC) as on macOS/Linux:
every text file is read and written as UTF-8 (Windows defaults to GBK there)
and nothing POSIX-only runs unguarded. Same scans as qumus-agent's."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_ROOTS = ("qumus_simlab", "tests")


def _python_files():
    for root in CODE_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel.startswith("v1compare/v1/") or "__pycache__" in rel:
                continue   # v1compare/v1 is the byte-identical frozen v1 copy
            yield rel, ast.parse(path.read_text(encoding="utf-8"))


# ----- scans -----------------------------------------------------------------

# where encoding= is (positionally) the n-th argument; Path.open has no file argument
_ENCODING_POSITION = {"read_text": 1, "write_text": 2, "open": 4}
_NOT_FILE_OPEN = {"os", "Image", "gzip", "zipfile", "tarfile", "io", "webbrowser", "sqlite3",
                  "self", "opener", "_opener", "request", "urllib"}


def _missing_encoding(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id == "open":
        name, path_method = "open", False
    elif isinstance(func, ast.Attribute) and func.attr in _ENCODING_POSITION:
        name, path_method = func.attr, func.attr == "open"
        owner = func.value.id if isinstance(func.value, ast.Name) else \
            func.value.attr if isinstance(func.value, ast.Attribute) else None
        if path_method and owner in _NOT_FILE_OPEN:
            return False
    else:
        return False
    if any(k.arg in (None, "encoding") for k in call.keywords):
        return False
    if len(call.args) >= _ENCODING_POSITION[name] - path_method:
        return False
    if name == "open":
        mode = next((k.value for k in call.keywords if k.arg == "mode"), None)
        position = 0 if path_method else 1
        if mode is None and len(call.args) > position:
            mode = call.args[position]
        if isinstance(mode, ast.Constant) and "b" in str(mode.value):
            return False
    return True


def test_every_text_file_is_read_and_written_as_utf8():
    """Windows reads text with the locale code page (GBK on a Chinese system):
    bench.yaml's Chinese comment and experience.yaml's "°C" would break.
    ruff's PLW1514 catches only calls on a known Path; this scans every call."""
    found = [f"{rel}:{node.lineno}" for rel, tree in _python_files()
             for node in ast.walk(tree) if isinstance(node, ast.Call) and _missing_encoding(node)]
    assert found == [], "add encoding='utf-8' to: " + ", ".join(found)


# POSIX-only names and where they are allowed (behind a Windows check)
_POSIX_ATTRS = {"SIGKILL", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "killpg", "getpgid",
                "setsid", "setpgrp", "fork", "WNOHANG", "pwrite", "getuid", "geteuid"}
_POSIX_MODULES = {"fcntl", "termios", "pwd", "grp", "resource"}
_POSIX_KEYWORDS = {"pass_fds", "preexec_fn"}
_GUARDED = {
    # every use sits behind a guard; listed after checking it
    "qumus_simlab/simlab/server.py": {"SIGUSR1", "resource"},   # hasattr / try-import
    "tests/test_windows_portability.py": {"SIGKILL", ".venv/bin", "libSystem"},   # the scan itself
}


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant):
            ids.add(id(body[0].value))
    return ids


def test_no_new_posix_only_calls():
    found = []
    for rel, tree in _python_files():
        allowed = _GUARDED.get(rel, set())
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            hit = None
            if isinstance(node, ast.Attribute) and node.attr in _POSIX_ATTRS:
                hit = node.attr
            elif isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                hit = next((n for n in names if n.split(".")[0] in _POSIX_MODULES), None)
            elif isinstance(node, ast.keyword) and node.arg in _POSIX_KEYWORDS:
                hit = node.arg
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docs and (".venv/bin" in node.value
                                                  or "/usr/lib/libSystem" in node.value):
                hit = ".venv/bin" if ".venv/bin" in node.value else "libSystem"
            if hit and hit not in allowed:
                found.append(f"{rel}:{getattr(node, 'lineno', '?')} {hit}")
    assert found == [], ("POSIX-only (breaks on Windows); guard it and list it here: " + ", ".join(found))


