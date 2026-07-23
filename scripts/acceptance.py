"""Run isolated canonical paper-runtime readiness and NATS fault acceptance."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PROJECT_PATTERN = re.compile(r"^tradeagent_acceptance_[a-z0-9_]+$")
CONTROL_PORT = 18094


def validate_project_name(value: str) -> str:
    if not PROJECT_PATTERN.fullmatch(value):
        raise ValueError("project name must use tradeagent_acceptance_<id> namespace")
    return value


def prepare_compose(
    rendered: dict[str, Any], *, password: str, runtime_password: str | None = None
) -> dict[str, Any]:
    runtime_password = runtime_password or password
    result = deepcopy(rendered)
    for service in result.get("services", {}).values():
        service.pop("container_name", None)
        service.pop("ports", None)
        environment = service.get("environment")
        if not isinstance(environment, dict):
            continue
        for key in ("POSTGRES_PASSWORD", "PGPASSWORD"):
            if key in environment:
                environment[key] = password
        if "DB_RUNTIME_PASSWORD" in environment:
            environment["DB_RUNTIME_PASSWORD"] = runtime_password
        database_url = environment.get("DATABASE_URL")
        if isinstance(database_url, str):
            environment["DATABASE_URL"] = re.sub(
                r"(postgresql://[^:/@]+:)[^@]*(@)",
                rf"\g<1>{runtime_password}\2",
                database_url,
            )
    control = result["services"]["control-plane"]
    control["ports"] = [
        {
            "target": 8001,
            "published": str(CONTROL_PORT),
            "host_ip": "127.0.0.1",
            "protocol": "tcp",
            "mode": "ingress",
        }
    ]
    for section in ("volumes", "networks"):
        for item in result.get(section, {}).values():
            item.pop("name", None)
            item.pop("external", None)
    return result


def _run(
    command: list[str], *, env: dict[str, str] | None = None, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def _compose(project: str, path: Path, *args: str, capture: bool = False) -> str:
    completed = _run(
        ["docker", "compose", "-p", project, "-f", str(path), *args],
        capture=capture,
    )
    return completed.stdout or ""


def _runtime(timeout_seconds: float = 3.0) -> dict[str, Any] | None:
    try:
        with urlopen(
            f"http://127.0.0.1:{CONTROL_PORT}/api/v1/runtime",
            timeout=timeout_seconds,
        ) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, RemoteDisconnected, json.JSONDecodeError):
        return None


def _wait_for_readiness(expected: bool, *, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = _runtime()
        if last is not None and last.get("ready") is expected:
            return last
        time.sleep(2)
    raise RuntimeError(f"readiness did not become {expected}; last={last}")


def _database_evidence(project: str, compose_path: Path) -> dict[str, int]:
    query = """
    SELECT json_build_object(
      'risk_decisions', (SELECT COUNT(*) FROM risk_decisions),
      'order_intents', (SELECT COUNT(*) FROM order_intents),
      'fills', (SELECT COUNT(*) FROM fills),
      'duplicate_inbox', (SELECT COUNT(*) FROM (
        SELECT 1 FROM event_inbox GROUP BY stream_name,durable_name,event_id HAVING COUNT(*) > 1
      ) duplicate_rows),
      'duplicate_fills', (SELECT COUNT(*) FROM (
        SELECT 1 FROM fills GROUP BY id HAVING COUNT(*) > 1
      ) duplicate_rows),
      'dead_letters', (SELECT COUNT(*) FROM dead_letter_events),
      'pending_outbox', (SELECT COUNT(*) FROM event_outbox WHERE status <> 'published'),
      'expired_leases', (SELECT COUNT(*) FROM worker_leases WHERE lease_expires_at <= NOW()),
      'failed_reconciliations', (SELECT COUNT(*) FROM reconciliation_runs WHERE status <> 'matched')
    );
    """
    output = _compose(
        project,
        compose_path,
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
    return {key: int(value) for key, value in json.loads(output.strip()).items()}


def _render_isolated_compose(password: str, runtime_password: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["POSTGRES_PASSWORD"] = password
    env["DB_RUNTIME_PASSWORD"] = runtime_password
    rendered = _run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "config",
            "--format",
            "json",
        ],
        env=env,
        capture=True,
    )
    return prepare_compose(
        json.loads(rendered.stdout),
        password=password,
        runtime_password=runtime_password,
    )


def run_acceptance(*, project: str, keep: bool = False) -> dict[str, Any]:
    validate_project_name(project)
    password = secrets.token_urlsafe(24)
    runtime_password = secrets.token_urlsafe(24)
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "project": project,
        "paper_trading_only": True,
        "live_trading_enabled": False,
        "checks": {},
    }
    with tempfile.TemporaryDirectory(prefix="tradeagent-acceptance-") as directory:
        compose_path = Path(directory) / "compose.json"
        compose_path.write_text(
            json.dumps(_render_isolated_compose(password, runtime_password)), encoding="utf-8"
        )
        try:
            _compose(project, compose_path, "down", "-v", "--remove-orphans")
            _compose(project, compose_path, "up", "-d", "--build")
            baseline = _wait_for_readiness(True, timeout_seconds=180)
            report["checks"]["baseline_readiness"] = baseline

            _compose(project, compose_path, "stop", "nats")
            outage_started = time.monotonic()
            time.sleep(30)
            outage = _wait_for_readiness(False, timeout_seconds=10)
            outage_seconds = time.monotonic() - outage_started
            if outage.get("dependencies", {}).get("nats", {}).get("healthy") is not False:
                raise RuntimeError("NATS outage did not fail its readiness dependency")
            report["checks"]["nats_outage"] = {
                "duration_seconds": round(outage_seconds, 3),
                "runtime": outage,
            }

            _compose(project, compose_path, "start", "nats")
            recovered = _wait_for_readiness(True, timeout_seconds=120)
            report["checks"]["recovered_readiness"] = recovered
            database = _database_evidence(project, compose_path)
            if any(
                database[key]
                for key in (
                    "duplicate_inbox",
                    "duplicate_fills",
                    "expired_leases",
                    "failed_reconciliations",
                )
            ):
                raise RuntimeError(f"post-recovery invariant failed: {database}")
            report["checks"]["database"] = database
            report["passed"] = True
        except Exception as exc:
            report["passed"] = False
            report["failure"] = f"{type(exc).__name__}: {exc}"
            try:
                report["logs_tail"] = _compose(
                    project, compose_path, "logs", "--no-color", "--tail=200", capture=True
                )[-20_000:]
            except Exception as log_error:
                report["logs_error"] = type(log_error).__name__
            raise
        finally:
            report["finished_at"] = datetime.now(UTC).isoformat()
            ARTIFACTS.mkdir(exist_ok=True)
            (ARTIFACTS / "acceptance-latest.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            if not keep:
                _compose(project, compose_path, "down", "-v", "--remove-orphans")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--project",
        type=validate_project_name,
        default=f"tradeagent_acceptance_{int(time.time())}",
    )
    result.add_argument("--keep", action="store_true", help="keep isolated stack for debugging")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = run_acceptance(project=args.project, keep=args.keep)
    except Exception as exc:
        print(f"acceptance failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
