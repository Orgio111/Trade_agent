"""Strict alpha gates, signed model cards, and dual-evidence promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from packages.certification.core import (
    canonical_json,
    sign_document,
    verify_document,
)

from .data import DatasetSnapshot
from .replay import ReplayMetrics
from .training import CandidateArtifact
from .walk_forward import WalkForwardConfig


@dataclass(frozen=True, slots=True)
class AlphaGateThresholds:
    min_folds: int = 3
    min_trades_per_fold: int = 30
    min_profit_factor: float = 1.1
    max_drawdown: float = 0.15
    min_fold_pass_ratio: float = 1.0
    min_positive_fold_ratio: float = 0.8
    max_worst_fold_loss: float = 0.02
    min_baseline_excess_return: float = 0.0
    min_oos_regimes: int = 3
    min_regimes_per_fold: int = 2
    min_shadow_seconds: int = 7 * 24 * 60 * 60

    def __post_init__(self) -> None:
        if (
            self.min_folds < 1
            or self.min_trades_per_fold < 1
            or self.min_oos_regimes < 1
            or self.min_regimes_per_fold < 1
        ):
            raise ValueError("minimum folds and trades must be positive")
        if self.min_profit_factor <= 0:
            raise ValueError("min_profit_factor must be positive")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        for name in ("min_fold_pass_ratio", "min_positive_fold_ratio"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_worst_fold_loss < 0:
            raise ValueError("max_worst_fold_loss must be non-negative")
        if self.min_shadow_seconds < 1:
            raise ValueError("min_shadow_seconds must be positive")

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    passed: bool
    gates: Mapping[str, Mapping[str, Any]]

    def to_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "gates": {name: dict(value) for name, value in self.gates.items()},
        }


def _gate(passed: bool, observed: Any, required: Any) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _profit_factor(value: float | None) -> float:
    return math.inf if value is None else value


def evaluate_alpha_gates(
    folds: Sequence[ReplayMetrics],
    thresholds: AlphaGateThresholds,
) -> PromotionDecision:
    """Fail closed unless profitability, stability, and integrity all pass."""

    count = len(folds)
    qualified = [
        fold
        for fold in folds
        if fold.trades >= thresholds.min_trades_per_fold
        and _profit_factor(fold.profit_factor) >= thresholds.min_profit_factor
        and fold.max_drawdown <= thresholds.max_drawdown
        and (
            fold.net_return - fold.buy_hold_return
            >= thresholds.min_baseline_excess_return
        )
    ]
    fold_pass_ratio = len(qualified) / count if count else 0.0
    positive_ratio = (
        sum(fold.net_return > 0 for fold in folds) / count if count else 0.0
    )
    worst_return = min((fold.net_return for fold in folds), default=-math.inf)
    duplicate_signals = sum(fold.duplicate_signals for fold in folds)
    duplicate_fills = sum(fold.duplicate_fills for fold in folds)
    reconciliation_mismatches = sum(fold.reconciliation_mismatches for fold in folds)
    next_bar_violations = sum(fold.next_bar_fill_violations for fold in folds)
    gates = {
        "fold_count": _gate(count >= thresholds.min_folds, count, thresholds.min_folds),
        "fold_quality": _gate(
            fold_pass_ratio >= thresholds.min_fold_pass_ratio,
            round(fold_pass_ratio, 6),
            thresholds.min_fold_pass_ratio,
        ),
        "profit_factor": _gate(
            all(
                _profit_factor(fold.profit_factor) >= thresholds.min_profit_factor
                for fold in folds
            )
            and bool(folds),
            [fold.profit_factor for fold in folds],
            f"every fold >= {thresholds.min_profit_factor}",
        ),
        "max_drawdown": _gate(
            all(fold.max_drawdown <= thresholds.max_drawdown for fold in folds)
            and bool(folds),
            [round(fold.max_drawdown, 8) for fold in folds],
            f"every fold <= {thresholds.max_drawdown}",
        ),
        "baseline_excess": _gate(
            all(
                fold.net_return - fold.buy_hold_return
                >= thresholds.min_baseline_excess_return
                for fold in folds
            )
            and bool(folds),
            [round(fold.net_return - fold.buy_hold_return, 8) for fold in folds],
            f"every fold >= {thresholds.min_baseline_excess_return}",
        ),
        "fold_stability": _gate(
            positive_ratio >= thresholds.min_positive_fold_ratio
            and worst_return >= -thresholds.max_worst_fold_loss,
            {
                "positive_fold_ratio": round(positive_ratio, 6),
                "worst_fold_return": (
                    round(worst_return, 8) if math.isfinite(worst_return) else None
                ),
            },
            {
                "positive_fold_ratio_min": thresholds.min_positive_fold_ratio,
                "worst_fold_return_min": -thresholds.max_worst_fold_loss,
            },
        ),
        "minimum_trades": _gate(
            all(fold.trades >= thresholds.min_trades_per_fold for fold in folds)
            and bool(folds),
            [fold.trades for fold in folds],
            f"every fold >= {thresholds.min_trades_per_fold}",
        ),
        "duplicate_signals": _gate(duplicate_signals == 0, duplicate_signals, 0),
        "duplicate_fills": _gate(duplicate_fills == 0, duplicate_fills, 0),
        "reconciliation_mismatches": _gate(
            reconciliation_mismatches == 0,
            reconciliation_mismatches,
            0,
        ),
        "next_bar_fill": _gate(next_bar_violations == 0, next_bar_violations, 0),
    }
    return PromotionDecision(
        passed=all(gate["passed"] is True for gate in gates.values()),
        gates=gates,
    )


def reliability_is_promoted(
    document: Mapping[str, Any] | None,
    key: bytes | None,
    *,
    git_commit: str | None = None,
    source_lock_sha256: str | None = None,
) -> bool:
    if document is None or key is None or not verify_document(document, key):
        return False
    state = document.get("certification_state")
    return bool(
        isinstance(state, Mapping)
        and state.get("status") == "promoted"
        and state.get("paper_only") is True
        and (git_commit is None or state.get("git_commit") == git_commit)
        and (
            source_lock_sha256 is None
            or state.get("source_lock_sha256") == source_lock_sha256
        )
    )


def shadow_is_passed(
    document: Mapping[str, Any] | None,
    key: bytes | None,
    *,
    artifact_sha256: str,
    minimum_seconds: int,
    alpha_model_card_sha256: str | None = None,
    reliability_document_sha256: str | None = None,
    git_commit: str | None = None,
    code_sha256: str | None = None,
) -> bool:
    """Authenticate model-specific paper shadow evidence."""

    if document is None or key is None or not verify_document(document, key):
        return False
    if document.get("document_type") != "alpha-shadow-paper":
        return False
    state = document.get("shadow_state")
    gates = state.get("gates") if isinstance(state, Mapping) else None
    return bool(
        isinstance(state, Mapping)
        and state.get("status") == "passed"
        and state.get("paper_only") is True
        and state.get("artifact_sha256") == artifact_sha256
        and isinstance(state.get("elapsed_seconds"), (int, float))
        and float(state["elapsed_seconds"]) >= minimum_seconds
        and state.get("duplicate_signals") == 0
        and state.get("duplicate_fills") == 0
        and state.get("reconciliation_mismatches") == 0
        and isinstance(gates, Mapping)
        and bool(gates)
        and all(
            isinstance(gate, Mapping) and gate.get("passed") is True
            for gate in gates.values()
        )
        and (
            alpha_model_card_sha256 is None
            or state.get("alpha_model_card_sha256") == alpha_model_card_sha256
        )
        and (
            reliability_document_sha256 is None
            or state.get("reliability_document_sha256") == reliability_document_sha256
        )
        and (git_commit is None or state.get("git_commit") == git_commit)
        and (code_sha256 is None or state.get("code_sha256") == code_sha256)
    )


class PromotionRegistry:
    """Write signed evidence always; copy a model only after both certifications."""

    def __init__(self, root: Path, signing_key: bytes) -> None:
        if len(signing_key) < 32:
            raise ValueError("alpha signing key must contain at least 32 bytes")
        self.root = root.resolve()
        self.signing_key = signing_key

    def record(
        self,
        *,
        artifact: CandidateArtifact,
        datasets: Sequence[DatasetSnapshot],
        walk_forward: WalkForwardConfig,
        fold_metrics: Sequence[ReplayMetrics],
        fold_evidence: Sequence[Mapping[str, Any]],
        alpha_decision: PromotionDecision,
        thresholds: AlphaGateThresholds,
        git_commit: str,
        code_sha256: str,
        source_lock_sha256: str,
        reliability_document: Mapping[str, Any] | None = None,
        reliability_key: bytes | None = None,
        shadow_document: Mapping[str, Any] | None = None,
        shadow_key: bytes | None = None,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], Path, Path | None, Path | None]:
        if not git_commit.strip():
            raise ValueError("git_commit cannot be blank")
        if len(code_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in code_sha256
        ):
            raise ValueError("code_sha256 must be lowercase SHA-256")
        if len(source_lock_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_lock_sha256
        ):
            raise ValueError("source_lock_sha256 must be lowercase SHA-256")
        artifact_bytes = artifact.path.read_bytes()
        if hashlib.sha256(artifact_bytes).hexdigest() != artifact.sha256:
            raise ValueError("candidate artifact changed before promotion")
        reliability_passed = reliability_is_promoted(
            reliability_document,
            reliability_key,
            git_commit=git_commit,
            source_lock_sha256=source_lock_sha256,
        )
        unbound_shadow_valid = shadow_is_passed(
            shadow_document,
            shadow_key,
            artifact_sha256=artifact.sha256,
            minimum_seconds=thresholds.min_shadow_seconds,
        )
        # The exact signed alpha-card digest is created below. No pre-existing
        # shadow document can therefore satisfy this card's promotion gate.
        shadow_passed = False
        if not alpha_decision.passed:
            status = "rejected"
        elif not reliability_passed:
            status = "alpha_certified_awaiting_reliability"
        else:
            # A shadow report must bind the hash of this exact signed alpha card.
            # That hash does not exist until this immutable staging card is written,
            # so final promotion is deliberately a separate operation.
            status = "alpha_certified_awaiting_shadow"
        timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        unsigned = {
            "schema_version": 1,
            "document_type": "alpha-model-card",
            "status": status,
            "created_at": timestamp,
            "git_commit": git_commit,
            "code_sha256": code_sha256,
            "source_lock_sha256": source_lock_sha256,
            "artifact": artifact.to_payload(),
            "datasets": [
                {
                    "manifest_file": snapshot.manifest_path.name,
                    "manifest_sha256": snapshot.manifest_sha256,
                    "data_sha256": snapshot.data_sha256,
                    "symbol": snapshot.request.symbol,
                    "interval": snapshot.request.interval,
                    "start": snapshot.request.start.isoformat(),
                    "end": snapshot.request.end.isoformat(),
                    "gaps": [value.isoformat() for value in snapshot.gaps],
                }
                for snapshot in datasets
            ],
            "walk_forward": walk_forward.to_payload(),
            "fold_metrics": [fold.to_payload() for fold in fold_metrics],
            "fold_evidence": [dict(fold) for fold in fold_evidence],
            "thresholds": thresholds.to_payload(),
            "alpha_gates": alpha_decision.to_payload(),
            "reliability_gate": {
                "passed": reliability_passed,
                "required": "valid signed paper reliability status=promoted",
                "source_document_sha256": (
                    hashlib.sha256(canonical_json(reliability_document)).hexdigest()
                    if isinstance(reliability_document, Mapping)
                    else None
                ),
                "source_key_id": (
                    reliability_document.get("signature", {}).get("key_id")
                    if isinstance(reliability_document, Mapping)
                    and isinstance(reliability_document.get("signature"), Mapping)
                    else None
                ),
            },
            "shadow_gate": {
                "passed": shadow_passed,
                "preexisting_document_valid_but_unbound": unbound_shadow_valid,
                "required": {
                    "signed": True,
                    "paper_only": True,
                    "artifact_sha256": artifact.sha256,
                    "alpha_model_card_sha256": "this exact signed card",
                    "elapsed_seconds_min": thresholds.min_shadow_seconds,
                    "duplicate_signals": 0,
                    "duplicate_fills": 0,
                    "reconciliation_mismatches": 0,
                },
                "source_document_sha256": (
                    hashlib.sha256(canonical_json(shadow_document)).hexdigest()
                    if isinstance(shadow_document, Mapping)
                    else None
                ),
            },
        }
        signed = sign_document(unsigned, self.signing_key)
        card_sha256 = hashlib.sha256(canonical_json(signed)).hexdigest()
        reports = self.root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        report_path = reports / (
            f"{artifact.sha256}-{card_sha256[:16]}-model-card.json"
        )
        report_bytes = (
            json.dumps(
                signed,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        self._write_immutable(report_path, report_bytes)
        staged_path: Path | None = None
        promoted_path: Path | None = None
        if alpha_decision.passed:
            staged = self.root / "staged" / artifact.sha256
            staged.mkdir(parents=True, exist_ok=True)
            staged_path = staged / "candidate.joblib"
            self._copy_immutable(artifact.path, staged_path, artifact.sha256)
            self._write_immutable(
                staged / f"model-card-{card_sha256}.json",
                report_bytes,
            )
        return signed, report_path, staged_path, promoted_path

    def promote_staged(
        self,
        *,
        alpha_model_card: Mapping[str, Any],
        reliability_document: Mapping[str, Any],
        reliability_key: bytes,
        shadow_document: Mapping[str, Any],
        shadow_key: bytes,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], Path, Path]:
        """Promote one immutable staged artifact without retraining it."""

        if not self.verify_card(alpha_model_card, self.signing_key):
            raise ValueError("alpha model card signature is invalid")
        if alpha_model_card.get("document_type") != "alpha-model-card":
            raise ValueError("unexpected alpha model-card document type")
        if alpha_model_card.get("status") not in {
            "alpha_certified_awaiting_reliability",
            "alpha_certified_awaiting_shadow",
        }:
            raise ValueError("alpha model card is not awaiting promotion evidence")
        alpha_gates = alpha_model_card.get("alpha_gates")
        if (
            not isinstance(alpha_gates, Mapping)
            or alpha_gates.get("passed") is not True
        ):
            raise ValueError("alpha gates did not pass")
        artifact = alpha_model_card.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("alpha model card is missing artifact identity")
        artifact_sha256 = str(artifact.get("artifact_sha256", ""))
        if len(artifact_sha256) != 64:
            raise ValueError("alpha model card has an invalid artifact digest")

        staged_path = self.root / "staged" / artifact_sha256 / "candidate.joblib"
        if not staged_path.is_file():
            raise FileNotFoundError(staged_path)
        if hashlib.sha256(staged_path.read_bytes()).hexdigest() != artifact_sha256:
            raise ValueError("staged candidate artifact hash mismatch")

        alpha_card_sha256 = hashlib.sha256(canonical_json(alpha_model_card)).hexdigest()
        reliability_sha256 = hashlib.sha256(
            canonical_json(reliability_document)
        ).hexdigest()
        thresholds_payload = alpha_model_card.get("thresholds")
        if not isinstance(thresholds_payload, Mapping):
            raise ValueError("alpha model card is missing promotion thresholds")
        minimum_seconds = int(thresholds_payload.get("min_shadow_seconds", 0))
        git_commit = str(alpha_model_card.get("git_commit", ""))
        source_lock_sha256 = str(alpha_model_card.get("source_lock_sha256", ""))
        if len(source_lock_sha256) != 64:
            raise ValueError("alpha model card has an invalid source lock digest")
        reliability_passed = reliability_is_promoted(
            reliability_document,
            reliability_key,
            git_commit=git_commit,
            source_lock_sha256=source_lock_sha256,
        )
        shadow_passed = shadow_is_passed(
            shadow_document,
            shadow_key,
            artifact_sha256=artifact_sha256,
            minimum_seconds=minimum_seconds,
            alpha_model_card_sha256=alpha_card_sha256,
            reliability_document_sha256=reliability_sha256,
            git_commit=git_commit,
            code_sha256=str(alpha_model_card.get("code_sha256", "")),
        )
        if not reliability_passed or not shadow_passed:
            raise ValueError(
                "signed reliability and artifact-bound shadow gates are required"
            )

        unsigned = dict(alpha_model_card)
        unsigned.pop("signature", None)
        unsigned["status"] = "promoted"
        unsigned["promoted_at"] = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        unsigned["promotion"] = {
            "prior_alpha_model_card_sha256": alpha_card_sha256,
            "reliability_document_sha256": reliability_sha256,
            "shadow_document_sha256": hashlib.sha256(
                canonical_json(shadow_document)
            ).hexdigest(),
            "artifact_sha256": artifact_sha256,
        }
        signed = sign_document(unsigned, self.signing_key)
        promoted_card_sha256 = hashlib.sha256(canonical_json(signed)).hexdigest()
        report_bytes = (
            json.dumps(
                signed,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        reports = self.root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        report_path = reports / (
            f"{artifact_sha256}-{promoted_card_sha256[:16]}-promotion.json"
        )
        self._write_immutable(report_path, report_bytes)

        destination = self.root / "models" / artifact_sha256
        destination.mkdir(parents=True, exist_ok=True)
        promoted_path = destination / "candidate.joblib"
        self._copy_immutable(staged_path, promoted_path, artifact_sha256)
        self._write_immutable(
            destination / f"model-card-{promoted_card_sha256}.json",
            report_bytes,
        )
        return signed, report_path, promoted_path

    @staticmethod
    def verify_card(document: Mapping[str, Any], key: bytes) -> bool:
        return verify_document(document, key)

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"immutable registry collision: {path}")
            return
        with path.open("xb") as handle:
            handle.write(payload)

    @staticmethod
    def _copy_immutable(source: Path, destination: Path, expected_sha256: str) -> None:
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise FileExistsError(
                    f"immutable promoted artifact collision: {destination}"
                )
            return
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError("promoted artifact hash verification failed")
