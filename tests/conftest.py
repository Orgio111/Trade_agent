"""pytest fixtures — shared resources across all test modules."""
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

# ── Rust risk engine binary path ──────────────────────────────────────────────
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


def _wait_for_port(host: str, port: int, timeout: float = 8.0) -> None:
    """Wait until the TCP port is accepting connections (blocking)."""
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except (ConnectionRefusedError, OSError) as exc:
            last_err = exc
            time.sleep(0.3)
    raise TimeoutError(
        f"Timed out waiting {timeout}s for port {host}:{port}: {last_err}"
    )


def _kill_proc(proc: subprocess.Popen) -> None:
    """Kill a subprocess (cross-platform)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=5,
        )
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture(scope="session")
def rust_engine_addr() -> Generator[str, None, None]:
    """Start the Rust risk engine as a subprocess and yield its TCP address.

    Session-scoped — engine starts once and is reused across all tests.
    Tests are skipped if the binary is missing or fails to start.
    """
    if not os.path.isfile(_RUST_BIN):
        pytest.skip(
            f"Rust binary not found at {_RUST_BIN}. "
            "Build it with: cd sentinel-x/rust && cargo build --release"
        )

    port = _find_free_port()
    addr = f"127.0.0.1:{port}"
    metrics_port = _find_free_port()

    env = os.environ.copy()
    env["RISK_TCP_ADDR"] = addr
    env["RISK_METRICS_ADDR"] = f"127.0.0.1:{metrics_port}"

    logger.info("Starting Rust risk engine on %s (metrics :%d) ...", addr, metrics_port)
    proc = subprocess.Popen(
        [_RUST_BIN],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_port("127.0.0.1", port, timeout=8.0)
        logger.info("Rust risk engine started at %s (pid=%d)", addr, proc.pid)
        yield addr
    except TimeoutError:
        _kill_proc(proc)
        stdout, stderr = proc.communicate()
        logger.warning(
            "Rust engine failed to start.\nstdout:\n%s\nstderr:\n%s",
            stdout.decode(errors="replace")[:500],
            stderr.decode(errors="replace")[:500],
        )
        pytest.skip(f"Rust engine failed to start on {addr}")
    finally:
        _kill_proc(proc)
        stdout, stderr = proc.communicate()
        if proc.returncode and proc.returncode != 0:
            logger.debug(
                "Rust engine exited with code %d", proc.returncode,
            )
