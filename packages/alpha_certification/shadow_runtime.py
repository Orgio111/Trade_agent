"""Docker-backed sampler and resumable controller for alpha paper shadow."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

from packages.certification.controller import (
    DockerRuntime,
    ProcessLock,
    SleepInhibitor,
    ensure_signing_key,
    utc_now,
)
from packages.certification.core import sign_document

from .shadow import (
    ShadowThresholds,
    create_shadow_state,
    evaluate_shadow_gates,
    finalize_shadow_state,
    record_shadow_sample,
)


_ACCOUNT_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")
_CONTAINER_ARTIFACT = "/run/alpha-candidate/candidate.joblib"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f"shadow evidence file is too large: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shadow evidence root must be an object")
    return payload


class AlphaShadowDockerSampler:
    """Observe the exact paper runtime and candidate artifact without mutation."""

    def __init__(
        self,
        runtime: DockerRuntime,
        *,
        artifact_sha256: str,
        account_id: str = "paper-main",
    ) -> None:
        if len(artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be lowercase SHA-256")
        if not _ACCOUNT_PATTERN.fullmatch(account_id):
            raise ValueError("invalid paper account identifier")
        self.runtime = runtime
        self.artifact_sha256 = artifact_sha256
        self.account_id = account_id

    def baseline_image_ids(self) -> dict[str, str]:
        baseline = self.runtime.baseline()
        images = baseline.get("image_ids")
        if not isinstance(images, Mapping) or not images:
            raise RuntimeError("canonical runtime has no image inventory")
        return {str(name): str(value) for name, value in images.items()}

    def _artifact_identity(self) -> tuple[str, str]:
        code = (
            "import hashlib,os,sys;"
            "p=sys.argv[1];"
            "h=hashlib.sha256();"
            "f=open(p,'rb');"
            "[h.update(c) for c in iter(lambda:f.read(1048576),b'')];"
            "f.close();"
            "print(os.environ.get('CANDIDATE_PROVIDER',''));"
            "print(h.hexdigest())"
        )
        output = self.runtime.compose(
            "exec",
            "-T",
            "candidate-worker",
            "python",
            "-c",
            code,
            _CONTAINER_ARTIFACT,
            capture=True,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if len(lines) != 2:
            raise RuntimeError(
                "candidate worker artifact probe returned invalid output"
            )
        return lines[0], lines[1]

    def _database_evidence(self, *, started_at: datetime) -> dict[str, Any]:
        started = started_at.astimezone(UTC).isoformat()
        query = f"""
        WITH candidate_events AS (
          SELECT payload
          FROM event_outbox
          WHERE subject = 'signals.candidate.v1'
            AND created_at >= '{started}'::timestamptz
        ),
        duplicate_signals AS (
          SELECT payload->'candidate'->>'signal_id' AS signal_id
          FROM candidate_events
          GROUP BY payload->'candidate'->>'signal_id'
          HAVING COUNT(*) > 1
        ),
        equity_points AS (
          SELECT equity,
                 MAX(equity) OVER (
                   ORDER BY created_at, source_sequence
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                 ) AS peak
          FROM portfolio_snapshots
          WHERE account_id = '{self.account_id}'
            AND created_at >= '{started}'::timestamptz
        )
        SELECT json_build_object(
          'candidate_events', (SELECT COUNT(*) FROM candidate_events),
          'candidate_digest_mismatches', (
            SELECT COUNT(*) FROM candidate_events
            WHERE payload->>'model_digest' IS DISTINCT FROM
                  '{self.artifact_sha256}'
          ),
          'duplicate_signals', (SELECT COUNT(*) FROM duplicate_signals),
          'fills', (
            SELECT COUNT(*) FROM fills
            WHERE received_ts >= '{started}'::timestamptz
          ),
          'duplicate_fills', (SELECT COUNT(*) FROM (
            SELECT venue_fill_id FROM fills
            WHERE received_ts >= '{started}'::timestamptz
            GROUP BY order_id, venue_fill_id HAVING COUNT(*) > 1
          ) duplicates),
          'reconciliation_mismatches', (
            SELECT COUNT(*) FROM reconciliation_runs
            WHERE started_at >= '{started}'::timestamptz
              AND status IN ('discrepancy', 'failed')
          ),
          'max_drawdown', COALESCE((
            SELECT MAX((peak - equity) / NULLIF(peak, 0))
            FROM equity_points
          ), 0)
        );
        """
        output = self.runtime.compose(
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
        payload = json.loads(output.strip())
        if not isinstance(payload, dict):
            raise RuntimeError("shadow database probe returned invalid JSON")
        return payload

    def capture(self, *, started_at: datetime) -> dict[str, Any]:
        captured_at = utc_now().isoformat()
        try:
            base = self.runtime.capture_sample()
            if base.get("controller_error"):
                raise RuntimeError(str(base["controller_error"]))
            provider, artifact_sha256 = self._artifact_identity()
            database = self._database_evidence(started_at=started_at)
            base_database = base.get("database")
            if isinstance(base_database, Mapping):
                database["duplicate_fills"] = max(
                    int(database.get("duplicate_fills", 0)),
                    int(base_database.get("duplicate_fills", 0)),
                )
            return {
                **base,
                "captured_at": captured_at,
                "paper_only": (
                    base.get("mode") == "paper_live"
                    and base.get("execution_enabled") is True
                ),
                "candidate_provider": provider,
                "artifact_sha256": artifact_sha256,
                "database": database,
            }
        except Exception as exc:
            return {
                "captured_at": captured_at,
                "ready": False,
                "execution_enabled": False,
                "paper_only": False,
                "controller_error": type(exc).__name__,
            }


class ShadowControllerPaths:
    """Artifact-specific bounded state and append-only sample journal."""

    def __init__(self, state_directory: Path) -> None:
        self.state_directory = state_directory.resolve()

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
        return self.state_directory / "shadow-signing.key"

    @property
    def lock(self) -> Path:
        return self.state_directory / "controller.lock"


class AlphaShadowCertificationController:
    """Collect seven days of exact artifact evidence and sign the result."""

    _HARD_GATES = frozenset(
        {
            "controller_errors",
            "paper_only",
            "artifact_digest",
            "candidate_provider",
            "immutable_images",
            "duplicate_signals",
            "duplicate_fills",
            "reconciliation_mismatches",
            "candidate_digest_mismatches",
            "max_drawdown",
        }
    )

    def __init__(
        self,
        *,
        sampler: AlphaShadowDockerSampler,
        paths: ShadowControllerPaths,
        thresholds: ShadowThresholds,
        artifact_sha256: str,
        alpha_model_card_sha256: str,
        reliability_document_sha256: str,
        git_commit: str,
        code_sha256: str,
    ) -> None:
        self.sampler = sampler
        self.paths = paths
        self.thresholds = thresholds
        self.artifact_sha256 = artifact_sha256
        self.alpha_model_card_sha256 = alpha_model_card_sha256
        self.reliability_document_sha256 = reliability_document_sha256
        self.git_commit = git_commit
        self.code_sha256 = code_sha256
        self.key = ensure_signing_key(paths.signing_key)

    def _persist(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.paths.state, state)
        signed = sign_document(
            {
                "schema_version": 1,
                "document_type": "alpha-shadow-paper",
                "generated_at": utc_now().isoformat(),
                "shadow_state": state,
            },
            self.key,
        )
        _atomic_json(self.paths.signed_status, signed)

    def _journal(self, sample: Mapping[str, Any]) -> None:
        self.paths.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(sample, sort_keys=True))
            stream.write("\n")

    def _load_or_create(self) -> dict[str, Any]:
        self.sampler.runtime.preflight()
        self.sampler.runtime.wait_readiness(expected=True)
        if self.paths.state.is_file():
            state = _load_json(self.paths.state)
            expected = {
                "artifact_sha256": self.artifact_sha256,
                "alpha_model_card_sha256": self.alpha_model_card_sha256,
                "reliability_document_sha256": self.reliability_document_sha256,
                "git_commit": self.git_commit,
                "code_sha256": self.code_sha256,
                "thresholds": {
                    name: getattr(self.thresholds, name)
                    for name in self.thresholds.__dataclass_fields__
                },
            }
            for name, value in expected.items():
                if state.get(name) != value:
                    raise RuntimeError(f"stored shadow state changed identity: {name}")
            return state
        state = create_shadow_state(
            artifact_sha256=self.artifact_sha256,
            alpha_model_card_sha256=self.alpha_model_card_sha256,
            reliability_document_sha256=self.reliability_document_sha256,
            git_commit=self.git_commit,
            code_sha256=self.code_sha256,
            now=utc_now(),
            thresholds=self.thresholds,
            baseline_image_ids=self.sampler.baseline_image_ids(),
        )
        self._persist(state)
        return state

    def _record(self, state: dict[str, Any]) -> None:
        started_at = datetime.fromisoformat(str(state["started_at"])).astimezone(UTC)
        sample = self.sampler.capture(started_at=started_at)
        self._journal(sample)
        record_shadow_sample(state, sample)
        gates = evaluate_shadow_gates(state, now=utc_now())
        failed_hard_gates = [
            name for name in self._HARD_GATES if gates[name]["passed"] is not True
        ]
        if failed_hard_gates:
            state["status"] = "failed"
            state["failure"] = {
                "reason": "hard shadow invariant failed",
                "gates": sorted(failed_hard_gates),
            }
            state["gates"] = gates
            state["completed_at"] = utc_now().isoformat()
        else:
            finalize_shadow_state(state, now=utc_now())
        self._persist(state)

    def run(self) -> int:
        with ProcessLock(self.paths.lock), SleepInhibitor():
            state = self._load_or_create()
            while state.get("status") == "running":
                self._record(state)
                if state.get("status") != "running":
                    break
                time.sleep(self.thresholds.sample_seconds)
            return 0 if state.get("status") == "passed" else 1
