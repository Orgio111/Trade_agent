"""Docker-backed, resumable 24-hour chaos and seven-day paper controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import RemoteDisconnected
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from packages.certification.core import (
    CHAOS_PHASE,
    PAPER_PHASE,
    Thresholds,
    advance_state,
    create_state,
    evaluate_external_gates,
    record_fault,
    record_sample,
    sign_document,
    verify_document,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT_PATTERN = re.compile(r"^trade_agent_canonical(?:_[a-z0-9]+)?$")
RUN_ID_PATTERN = re.compile(r"^cert-[0-9]{8}t[0-9]{6}z-[0-9a-f]{8}$")
MAX_EVIDENCE_BYTES = 1024 * 1024


def utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError(f"evidence file is too large: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence root must be an object: {path.name}")
    return payload


def _new_private_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(path, 0o600)


def ensure_signing_key(path: Path) -> bytes:
    """Load or create a local-only HMAC key without overwriting an existing key."""

    if not path.exists():
        try:
            _new_private_file(path, secrets.token_bytes(32))
        except FileExistsError:
            pass
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("certification signing key must contain at least 32 bytes")
    return key


def load_signing_key(path: Path) -> bytes:
    """Load an existing key; verification must never invent missing authority."""

    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("certification signing key must contain at least 32 bytes")
    return key


class ProcessLock:
    """Small cross-platform single-controller lock with stale-PID recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    @staticmethod
    def _running(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            process_query_limited_information = 0x1000
            still_active = 259
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def __enter__(self) -> ProcessLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                try:
                    pid = int(self.path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    pid = -1
                if self._running(pid):
                    raise RuntimeError(
                        f"certification controller already runs as PID {pid}"
                    )
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(str(os.getpid()))
            self.acquired = True
            return self
        raise RuntimeError("could not acquire certification controller lock")

    def __exit__(self, *_: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


class SleepInhibitor:
    """Prevent automatic Windows sleep while an elapsed certification is active."""

    def __init__(self) -> None:
        self.active = False

    def __enter__(self) -> SleepInhibitor:
        if os.name != "nt":
            return self
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        es_continuous = 0x80000000
        es_system_required = 0x00000001
        if kernel32.SetThreadExecutionState(es_continuous | es_system_required) == 0:
            raise OSError(ctypes.get_last_error(), "could not inhibit automatic sleep")
        self.active = True
        return self

    def __exit__(self, *_: object) -> None:
        if os.name != "nt" or not self.active:
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        es_continuous = 0x80000000
        kernel32.SetThreadExecutionState(es_continuous)
        self.active = False


@dataclass(frozen=True)
class ControllerPaths:
    state_directory: Path
    restore_evidence: Path
    alert_webhook_url_file: Path | None = None
    promoted_artifact: Path = ROOT / "artifacts" / "24x7-certification-latest.json"

    @property
    def state(self) -> Path:
        return self.state_directory / "state.json"

    @property
    def journal(self) -> Path:
        return self.state_directory / "samples.jsonl"

    @property
    def signed_status(self) -> Path:
        return self.state_directory / "signed-status.json"

    @property
    def signing_key(self) -> Path:
        return self.state_directory / "certification-signing.key"

    @property
    def lock(self) -> Path:
        return self.state_directory / "controller.lock"


@dataclass(frozen=True)
class RuntimeConfig:
    project: str
    env_file: Path
    runtime_url: str = "http://127.0.0.1:8001/api/v1/runtime"
    outage_seconds: int = 30
    readiness_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not PROJECT_PATTERN.fullmatch(self.project):
            raise ValueError(
                "project must use the isolated trade_agent_canonical namespace"
            )
        if self.outage_seconds <= 0 or self.readiness_timeout_seconds <= 0:
            raise ValueError("fault and readiness durations must be positive")
        parsed = urlsplit(self.runtime_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("runtime URL must be loopback HTTP")


class DockerRuntime:
    """Read and fault only the explicitly bound canonical Compose project."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def _run(
        self,
        command: list[str],
        *,
        capture: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=check,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )

    def compose(
        self,
        *args: str,
        capture: bool = False,
        check: bool = True,
    ) -> str:
        completed = self._run(
            [
                "docker",
                "compose",
                "--env-file",
                str(self.config.env_file),
                "-p",
                self.config.project,
                *args,
            ],
            capture=capture,
            check=check,
        )
        return completed.stdout or ""

    def preflight(self) -> None:
        if not self.config.env_file.is_file():
            raise FileNotFoundError(self.config.env_file)
        self._run(
            [
                sys.executable,
                str(ROOT / "scripts" / "config_preflight.py"),
                "--env-file",
                str(self.config.env_file),
            ]
        )
        self.compose("config", "--quiet")

    def start(self) -> None:
        self.preflight()
        self.compose("up", "-d")
        self.wait_readiness(expected=True)

    def runtime(self, *, timeout_seconds: float = 5.0) -> dict[str, Any] | None:
        try:
            with urlopen(self.config.runtime_url, timeout=timeout_seconds) as response:
                payload = json.load(response)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            RemoteDisconnected,
            json.JSONDecodeError,
        ):
            return None
        return payload if isinstance(payload, dict) else None

    def wait_readiness(
        self,
        *,
        expected: bool,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + (
            timeout_seconds or self.config.readiness_timeout_seconds
        )
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            last = self.runtime()
            if last is not None and last.get("ready") is expected:
                return last
            time.sleep(2)
        raise RuntimeError(
            f"readiness did not become {expected}; last_ready="
            f"{last.get('ready') if last else None}"
        )

    def database_evidence(self) -> dict[str, int]:
        query = """
        SELECT json_build_object(
          'risk_decisions', (SELECT COUNT(*) FROM risk_decisions),
          'order_intents', (SELECT COUNT(*) FROM order_intents),
          'fills', (SELECT COUNT(*) FROM fills),
          'duplicate_inbox', (SELECT COUNT(*) FROM (
            SELECT 1 FROM event_inbox
            GROUP BY stream_name,durable_name,event_id HAVING COUNT(*) > 1
          ) duplicate_rows),
          'duplicate_fills', (SELECT COUNT(*) FROM (
            SELECT 1 FROM fills GROUP BY id HAVING COUNT(*) > 1
          ) duplicate_rows),
          'reconciliation_discrepancies', (
            SELECT COUNT(*) FROM reconciliation_runs
            WHERE status = 'discrepancy'
          ),
          'failed_reconciliations', (
            SELECT COUNT(*) FROM reconciliation_runs WHERE status = 'failed'
          ),
          'dead_letters', (SELECT COUNT(*) FROM dead_letter_events),
          'pending_outbox', (
            SELECT COUNT(*) FROM event_outbox WHERE status <> 'published'
          ),
          'expired_leases', (
            SELECT COUNT(*) FROM worker_leases WHERE lease_expires_at <= NOW()
          )
        );
        """
        output = self.compose(
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "quantex",
            "-d",
            "quantex",
            "-Atc",
            query,
            capture=True,
        )
        parsed = json.loads(output.strip())
        return {str(key): int(value) for key, value in parsed.items()}

    def container_inventory(self) -> dict[str, dict[str, Any]]:
        ids = [
            line.strip()
            for line in self.compose("ps", "-q", capture=True).splitlines()
            if line.strip()
        ]
        if not ids:
            raise RuntimeError("canonical project has no running containers")
        output = self._run(["docker", "inspect", *ids], capture=True).stdout or "[]"
        containers = json.loads(output)
        result: dict[str, dict[str, Any]] = {}
        for container in containers:
            labels = container.get("Config", {}).get("Labels", {})
            if labels.get("com.docker.compose.project") != self.config.project:
                raise RuntimeError("container escaped the bound Compose project")
            service = labels.get("com.docker.compose.service")
            if not isinstance(service, str):
                raise RuntimeError("container is missing its Compose service label")
            result[service] = {
                "container_id": str(container.get("Id", "")),
                "image_id": str(container.get("Image", "")),
                "restart_count": int(container.get("RestartCount", 0)),
                "started_at": str(container.get("State", {}).get("StartedAt", "")),
            }
        return result

    def restart_counts(self) -> dict[str, int]:
        return {
            service: int(item["restart_count"])
            for service, item in self.container_inventory().items()
        }

    def baseline(self) -> dict[str, Any]:
        database = self.database_evidence()
        inventory = self.container_inventory()
        return {
            **database,
            "restart_counts": {
                service: int(item["restart_count"])
                for service, item in inventory.items()
            },
            "container_ids": {
                service: str(item["container_id"])
                for service, item in inventory.items()
            },
            "image_ids": {
                service: str(item["image_id"]) for service, item in inventory.items()
            },
        }

    def capture_sample(self, *, expected_fault: bool = False) -> dict[str, Any]:
        captured = utc_now().isoformat()
        try:
            runtime = self.runtime()
            database = self.database_evidence()
            inventory = self.container_inventory()
            if runtime is None:
                raise RuntimeError("runtime endpoint unavailable")
            policy_violation = None
            if runtime.get("mode") != "paper_live":
                policy_violation = "runtime mode is not paper_live"
            return {
                "captured_at": captured,
                "ready": runtime.get("ready") is True,
                "execution_enabled": runtime.get("execution_enabled") is True,
                "mode": runtime.get("mode"),
                "dependencies": runtime.get("dependencies", {}),
                "database": database,
                "restart_counts": {
                    service: int(item["restart_count"])
                    for service, item in inventory.items()
                },
                "container_ids": {
                    service: str(item["container_id"])
                    for service, item in inventory.items()
                },
                "image_ids": {
                    service: str(item["image_id"])
                    for service, item in inventory.items()
                },
                "expected_fault": expected_fault,
                "policy_violation": policy_violation,
            }
        except Exception as exc:
            return {
                "captured_at": captured,
                "ready": False,
                "expected_fault": expected_fault,
                "controller_error": type(exc).__name__,
            }

    def _fault_dependency(self, service: str) -> tuple[datetime, datetime]:
        started = utc_now()
        self.compose("stop", service)
        self.wait_readiness(expected=False, timeout_seconds=90)
        time.sleep(self.config.outage_seconds)
        self.compose("start", service)
        self.wait_readiness(expected=True)
        return started, utc_now()

    def _fault_candidate(self) -> tuple[datetime, datetime]:
        container_id = self.compose(
            "ps", "-q", "candidate-worker", capture=True
        ).strip()
        if not container_id:
            raise RuntimeError("candidate worker container is unavailable")
        before = self.restart_counts().get("candidate-worker", 0)
        started = utc_now()
        self._run(["docker", "kill", container_id], capture=True)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            current = self.restart_counts().get("candidate-worker", 0)
            if current > before:
                self.wait_readiness(expected=True)
                return started, utc_now()
            time.sleep(2)
        self.compose("start", "candidate-worker", check=False)
        self.wait_readiness(expected=True)
        raise RuntimeError("candidate worker did not auto-restart after crash")

    def inject_fault(self, name: str) -> tuple[datetime, datetime]:
        if name == "nats_outage":
            return self._fault_dependency("nats")
        if name == "postgres_outage":
            return self._fault_dependency("postgres")
        if name == "candidate_worker_crash":
            return self._fault_candidate()
        raise ValueError(f"unsupported fault: {name}")


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _source_lock_sha256() -> str:
    path = ROOT / "project.manifest.lock.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source_lock() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "knowledge.py"), "check"],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _run_id(now: datetime) -> str:
    return f"cert-{now.strftime('%Y%m%dt%H%M%Sz')}-{secrets.token_hex(4)}"


def _destination_allowed(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme == "https" and parsed.hostname:
        return True
    return bool(
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    )


def send_alert_canary(
    url_file: Path | None,
    *,
    run_id: str,
    event: str,
) -> dict[str, Any] | None:
    """Send a bounded canary without persisting its potentially secret URL."""

    if url_file is None or not url_file.is_file():
        return None
    if url_file.stat().st_size > 4096:
        raise ValueError("alert webhook URL file exceeds 4096 bytes")
    url = url_file.read_text(encoding="utf-8").strip()
    if not _destination_allowed(url):
        raise ValueError("alert webhook must use HTTPS or loopback HTTP")
    payload = {
        "event": event,
        "run_id": run_id,
        "nonce": secrets.token_hex(16),
        "sent_at": utc_now().isoformat(),
        "severity": "test",
        "paper_only": True,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read(4096)
            status = int(response.status)
    except (HTTPError, URLError, TimeoutError, RemoteDisconnected) as exc:
        return {
            "acknowledged": False,
            "error": type(exc).__name__,
            "destination_id": hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
            "attempted_at": utc_now().isoformat(),
        }
    return {
        "acknowledged": 200 <= status < 300,
        "status_code": status,
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "destination_id": hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "attempted_at": utc_now().isoformat(),
    }


class CertificationController:
    """Persist samples and automatically move through both elapsed phases."""

    def __init__(
        self,
        *,
        runtime: DockerRuntime,
        paths: ControllerPaths,
        thresholds: Thresholds,
        start_runtime: bool,
    ) -> None:
        self.runtime = runtime
        self.paths = paths
        self.thresholds = thresholds
        self.start_runtime = start_runtime
        self.key = ensure_signing_key(paths.signing_key)

    def _write_journal(self, record: Mapping[str, Any]) -> None:
        self.paths.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True))
            stream.write("\n")

    def _persist(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.paths.state, state)
        signed = sign_document(
            {
                "schema_version": 1,
                "generated_at": utc_now().isoformat(),
                "certification_state": state,
            },
            self.key,
        )
        _atomic_json(self.paths.signed_status, signed)
        if state.get("status") == "promoted":
            _atomic_json(self.paths.promoted_artifact, signed)

    def _load_or_create(self) -> dict[str, Any]:
        _verify_source_lock()
        if self.start_runtime:
            self.runtime.start()
        else:
            self.runtime.preflight()
            self.runtime.wait_readiness(expected=True)
        if self.paths.state.exists():
            state = _load_json(self.paths.state)
            if state.get("project") != self.runtime.config.project:
                raise RuntimeError("stored state belongs to another Compose project")
            if state.get("git_commit") != _git_commit():
                raise RuntimeError(
                    "repository HEAD changed during certification; start a new run"
                )
            if state.get("source_lock_sha256") != _source_lock_sha256():
                raise RuntimeError(
                    "project manifest lock changed during certification; start a new run"
                )
            if state.get("thresholds") != self.thresholds.__dict__:
                raise RuntimeError("stored thresholds differ from the requested run")
            return state
        now = utc_now()
        state = create_state(
            run_id=_run_id(now),
            now=now,
            thresholds=self.thresholds,
            baseline=self.runtime.baseline(),
            git_commit=_git_commit(),
            source_lock_sha256=_source_lock_sha256(),
            project=self.runtime.config.project,
        )
        if not RUN_ID_PATTERN.fullmatch(state["run_id"]):
            raise RuntimeError("generated run ID violated the certification namespace")
        state["external_evidence"]["alert_delivery"] = send_alert_canary(
            self.paths.alert_webhook_url_file,
            run_id=state["run_id"],
            event="certification_started",
        )
        self._persist(state)
        return state

    @staticmethod
    def _phase_elapsed(state: Mapping[str, Any], now: datetime) -> float:
        started = datetime.fromisoformat(str(state["phase_started_at"]))
        return max(0.0, (now - started.astimezone(UTC)).total_seconds())

    def _due_fault(self, state: Mapping[str, Any], now: datetime) -> str | None:
        if state.get("phase") != CHAOS_PHASE:
            return None
        elapsed = self._phase_elapsed(state, now)
        for fault in state["faults"]:
            if fault["status"] == "pending" and elapsed >= float(
                fault["scheduled_offset_seconds"]
            ):
                return str(fault["name"])
        return None

    def _attempt_fault(self, state: dict[str, Any], name: str) -> None:
        started = utc_now()
        try:
            actual_started, recovered = self.runtime.inject_fault(name)
        except Exception as exc:
            recovered = utc_now()
            record_fault(
                state,
                name=name,
                started_at=started,
                recovered_at=recovered,
                passed=False,
                detail=type(exc).__name__,
            )
        else:
            record_fault(
                state,
                name=name,
                started_at=actual_started,
                recovered_at=recovered,
                passed=True,
                detail="readiness recovered",
            )
        alert = send_alert_canary(
            self.paths.alert_webhook_url_file,
            run_id=state["run_id"],
            event=f"fault_{name}",
        )
        if alert is not None:
            state["external_evidence"]["alert_delivery"] = alert
        fault_passed = next(
            fault["status"] == "passed"
            for fault in state["faults"]
            if fault["name"] == name
        )
        if not fault_passed:
            state["status"] = "failed"
            state["failure"] = f"scheduled fault did not recover: {name}"
        self._write_journal(
            {
                "type": "fault",
                "name": name,
                "recorded_at": utc_now().isoformat(),
                "result": next(
                    fault for fault in state["faults"] if fault["name"] == name
                ),
            }
        )
        self._persist(state)

    def _external_gates(self, state: dict[str, Any], now: datetime) -> dict[str, Any]:
        alert = state["external_evidence"].get("alert_delivery")
        if not alert or alert.get("acknowledged") is not True:
            alert = send_alert_canary(
                self.paths.alert_webhook_url_file,
                run_id=state["run_id"],
                event="promotion_gate",
            )
            state["external_evidence"]["alert_delivery"] = alert
        restore = (
            _load_json(self.paths.restore_evidence)
            if self.paths.restore_evidence.is_file()
            else None
        )
        state["external_evidence"]["restore"] = restore
        return evaluate_external_gates(
            alert_delivery=alert,
            restore=restore,
            thresholds=self.thresholds,
            run_started_at=state["started_at"],
            now=now,
        )

    def _sample_and_advance(self, state: dict[str, Any]) -> None:
        sample = self.runtime.capture_sample()
        self._write_journal({"type": "sample", **sample})
        record_sample(state, sample)
        if sample.get("policy_violation"):
            state["status"] = "failed"
            state["failure"] = str(sample["policy_violation"])
        database = sample.get("database", {})
        if isinstance(database, Mapping) and (
            int(database.get("duplicate_inbox", 0)) > 0
            or int(database.get("duplicate_fills", 0)) > 0
            or int(database.get("reconciliation_discrepancies", 0)) > 0
            or int(state["summary"].get("max_failed_reconciliations_delta", 0)) > 0
        ):
            state["status"] = "failed"
            state["failure"] = "hard data-integrity invariant failed"
        now = utc_now()
        external = (
            self._external_gates(state, now)
            if state.get("phase") == PAPER_PHASE
            and self._phase_elapsed(state, now) >= self.thresholds.paper_seconds
            else None
        )
        baseline = state["summary"]["baseline"]
        if isinstance(sample.get("database"), Mapping) and isinstance(
            sample.get("restart_counts"), Mapping
        ):
            baseline = {
                **sample["database"],
                "restart_counts": sample["restart_counts"],
                "container_ids": sample.get("container_ids", {}),
                "image_ids": sample.get("image_ids", {}),
            }
        advance_state(
            state,
            now=now,
            baseline=baseline,
            external_gates=external,
        )
        self._persist(state)

    def run(self) -> int:
        with ProcessLock(self.paths.lock), SleepInhibitor():
            state = self._load_or_create()
            if state.get("status") == "promoted":
                self._persist(state)
                return 0
            while state.get("status") not in {"promoted", "failed"}:
                due = self._due_fault(state, utc_now())
                if due:
                    self._attempt_fault(state, due)
                else:
                    self._sample_and_advance(state)
                if state.get("status") in {"promoted", "failed"}:
                    break
                time.sleep(self.thresholds.sample_seconds)
            return 0 if state.get("status") == "promoted" else 1


def read_status(paths: ControllerPaths, *, verify: bool = True) -> dict[str, Any]:
    """Read and optionally authenticate the latest bounded status snapshot."""

    if not paths.signed_status.is_file():
        raise FileNotFoundError(paths.signed_status)
    document = _load_json(paths.signed_status)
    if verify:
        key = load_signing_key(paths.signing_key)
        if not verify_document(document, key):
            raise RuntimeError("certification status signature is invalid")
    return document
