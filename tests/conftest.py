"""pytest fixtures for integration tests — launches Rust risk engine as subprocess."""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator

import pytest

logger = logging.getLogger(__name__)

# Path to the compiled Rust binary — relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_RUST_BIN = os.path.join(
    _PROJECT_ROOT,
    "sentinel-x",
    "rust",
    "target",
    "release",
    "sentinel-risk.exe" if sys.platform == "win32" else "sentinel-risk",
)


def _find_free_port() -> int:
    """Find a random available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    """Wait until the TCP port is accepting connections (blocking)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting {timeout}s for Rust engine on {host}:{port}"
                )
            time.sleep(0.2)


@pytest.fixture(scope="session")
def rust_engine_addr() -> Generator[str, None, None]:
    """Start the Rust risk engine as a subprocess and yield its TCP address.

    Session-scoped — the engine starts once and is reused across all tests.
    Teardown kills the subprocess.
    """
    if not os.path.isfile(_RUST_BIN):
        pytest.skip(
            f"Rust binary not found at {_RUST_BIN}. "
            "Build it with: cd sentinel-x/rust && cargo build --release"
        )

    port = _find_free_port()
    addr = f"127.0.0.1:{port}"

    env = os.environ.copy()
    env["RISK_TCP_ADDR"] = addr
    # Prometheus metrics on a separate random port
    env.pop("RISK_METRICS_ADDR", None)  # engine falls back to :9091 if not set

    proc = subprocess.Popen(
        [_RUST_BIN],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_port("127.0.0.1", port, timeout=15.0)
        logger.info("Rust risk engine started at %s (pid=%d)", addr, proc.pid)
        yield addr
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Rust engine didn't terminate — killing")
                proc.kill()
                proc.wait()

        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "Rust engine exited with code %d\nstdout:\n%s\nstderr:\n%s",
                proc.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
