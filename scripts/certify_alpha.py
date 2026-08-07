"""Download immutable data, replay canonical candidates, and certify alpha."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from packages.alpha_certification.controller import (  # noqa: E402
    AlphaCertificationConfig,
    AlphaCertificationController,
)
from packages.alpha_certification.data import (  # noqa: E402
    BinanceHistoricalDownloader,
    GapPolicy,
    HistoricalRequest,
    load_snapshot,
)
from packages.alpha_certification.promotion import (  # noqa: E402
    AlphaGateThresholds,
    PromotionRegistry,
    reliability_is_promoted,
)
from packages.alpha_certification.replay import (  # noqa: E402
    CanonicalAlphaReplay,
    CanonicalCandidateAdapter,
    ReplayConfig,
    ReplayCostModel,
)
from packages.alpha_certification.training import (  # noqa: E402
    CandidateTrainingConfig,
)
from packages.alpha_certification.shadow import ShadowThresholds  # noqa: E402
from packages.alpha_certification.shadow_runtime import (  # noqa: E402
    AlphaShadowCertificationController,
    AlphaShadowDockerSampler,
    ShadowControllerPaths,
)
from packages.alpha_certification.walk_forward import (  # noqa: E402
    WalkForwardConfig,
)
from packages.certification.controller import (  # noqa: E402
    DockerRuntime,
    RuntimeConfig,
)
from packages.certification.core import (  # noqa: E402
    canonical_json,
    verify_document,
)
from packages.risk import InstrumentConstraints, RiskPolicy  # noqa: E402
from workers.candidate.fallback import (  # noqa: E402
    DeterministicBaselineCandidateProducer,
)
from workers.candidate.ollama import OllamaCandidateClient  # noqa: E402
from workers.candidate.runtime import TypedCandidateProducer  # noqa: E402


DEFAULT_ROOT = SCRIPT_ROOT / ".local" / "alpha-certification"


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unit_ratio(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1]")
    return parsed


def _non_negative(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _constraints(symbol: str) -> InstrumentConstraints:
    normalized = symbol.upper()
    if normalized == "BTCUSDT":
        step = Decimal("0.00001")
        minimum = Decimal("0.00001")
    elif normalized == "ETHUSDT":
        step = Decimal("0.0001")
        minimum = Decimal("0.0001")
    else:
        raise ValueError(f"no audited default constraints for {normalized}")
    return InstrumentConstraints(
        venue="binance",
        market_type="spot",
        instrument=normalized,
        tick_size=Decimal("0.01"),
        step_size=step,
        min_quantity=minimum,
        min_notional=Decimal("10"),
    )


def _risk_policy() -> RiskPolicy:
    return RiskPolicy(
        version="alpha-certification-v1",
        min_confidence=Decimal("0.55"),
        max_data_age_seconds=Decimal("5"),
        max_portfolio_age_seconds=Decimal("30"),
        max_spread_bps=Decimal("10"),
        max_slippage_bps=Decimal("10"),
    )


def _load_or_create_key(path: Path) -> bytes:
    path = path.resolve()
    if path.exists():
        key = path.read_bytes()
        if len(key) < 32:
            raise ValueError("signing key must contain at least 32 bytes")
        return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, key)
    finally:
        os.close(descriptor)
    return key


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SCRIPT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _code_sha256() -> str:
    paths = {
        *SCRIPT_ROOT.glob("packages/alpha_certification/*.py"),
        SCRIPT_ROOT / "packages" / "certification" / "core.py",
        SCRIPT_ROOT / "packages" / "domain" / "events.py",
        SCRIPT_ROOT / "packages" / "execution" / "models.py",
        *SCRIPT_ROOT.glob("packages/risk/*.py"),
        SCRIPT_ROOT / "workers" / "features" / "runtime.py",
        SCRIPT_ROOT / "workers" / "config.py",
        SCRIPT_ROOT / "workers" / "contracts.py",
        SCRIPT_ROOT / "workers" / "candidate" / "__main__.py",
        SCRIPT_ROOT / "workers" / "candidate" / "runtime.py",
        SCRIPT_ROOT / "workers" / "candidate" / "service.py",
        SCRIPT_ROOT / "workers" / "candidate" / "fallback.py",
        SCRIPT_ROOT / "infra" / "workers" / "Dockerfile",
        SCRIPT_ROOT / "docker-compose.yml",
        SCRIPT_ROOT / "pyproject.toml",
        SCRIPT_ROOT / "uv.lock",
        Path(__file__).resolve(),
    }
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(SCRIPT_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_lock_sha256() -> str:
    path = SCRIPT_ROOT / "project.manifest.lock.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source_lock() -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / "scripts" / "knowledge.py"),
            "check",
        ],
        cwd=SCRIPT_ROOT,
        check=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download")
    download.add_argument("--symbol", required=True)
    download.add_argument("--interval", default="1h")
    download.add_argument("--start", required=True, type=_aware_datetime)
    download.add_argument("--end", required=True, type=_aware_datetime)
    download.add_argument(
        "--gap-policy",
        choices=[policy.value for policy in GapPolicy],
        default=GapPolicy.REJECT.value,
    )
    download.add_argument(
        "--allow-gap",
        action="append",
        default=[],
        type=_aware_datetime,
    )
    download.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_ROOT / "datasets",
    )

    replay = commands.add_parser("replay")
    replay.add_argument("--dataset-manifest", type=Path, required=True)
    replay.add_argument(
        "--provider",
        choices=["deterministic_baseline", "ollama"],
        default="deterministic_baseline",
    )
    replay.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    replay.add_argument("--initial-equity", default="10000")
    replay.add_argument("--fee-bps", default="4")
    replay.add_argument("--spread-bps", default="4")
    replay.add_argument("--slippage-bps", default="3")
    replay.add_argument("--max-holding-bars", type=_positive_int, default=12)
    replay.add_argument("--output", type=Path)

    certify = commands.add_parser("certify")
    certify.add_argument(
        "--dataset-manifest",
        type=Path,
        action="append",
        required=True,
    )
    certify.add_argument("--train-days", type=_positive_int, default=730)
    certify.add_argument("--test-days", type=_positive_int, default=180)
    certify.add_argument("--purge-hours", type=_positive_int)
    certify.add_argument("--embargo-hours", type=_positive_int, default=24)
    certify.add_argument(
        "--expanding",
        action="store_true",
        help="use expanding training history instead of the bounded rolling default",
    )
    certify.add_argument("--horizon-bars", type=_positive_int, default=12)
    certify.add_argument("--min-training-samples", type=_positive_int, default=500)
    certify.add_argument("--n-estimators", type=_positive_int, default=300)
    certify.add_argument("--min-test-samples", type=_positive_int, default=100)
    certify.add_argument("--min-folds", type=_positive_int, default=3)
    certify.add_argument("--min-trades", type=_positive_int, default=30)
    certify.add_argument("--min-profit-factor", type=float, default=1.1)
    certify.add_argument("--max-drawdown", type=_unit_ratio, default=0.15)
    certify.add_argument("--min-fold-pass-ratio", type=_unit_ratio, default=1.0)
    certify.add_argument(
        "--min-positive-fold-ratio",
        type=_unit_ratio,
        default=0.8,
    )
    certify.add_argument("--max-worst-fold-loss", type=_non_negative, default=0.02)
    certify.add_argument(
        "--min-baseline-excess-return",
        type=float,
        default=0.0,
    )
    certify.add_argument("--min-oos-regimes", type=_positive_int, default=3)
    certify.add_argument("--min-regimes-per-fold", type=_positive_int, default=2)
    certify.add_argument("--fee-bps", default="4")
    certify.add_argument("--spread-bps", default="4")
    certify.add_argument("--slippage-bps", default="3")
    certify.add_argument("--cost-stress-multiplier", default="1.5", type=Decimal)
    certify.add_argument(
        "--candidate-workspace",
        type=Path,
        default=DEFAULT_ROOT / "candidates",
    )
    certify.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_ROOT / "registry",
    )
    certify.add_argument(
        "--signing-key",
        type=Path,
        default=DEFAULT_ROOT / "alpha-signing.key",
    )
    certify.add_argument("--reliability-document", type=Path)
    certify.add_argument("--reliability-key", type=Path)
    certify.add_argument("--shadow-document", type=Path)
    certify.add_argument("--shadow-key", type=Path)
    certify.add_argument("--min-shadow-days", type=_positive_int, default=7)

    shadow = commands.add_parser("shadow-run")
    shadow.add_argument("--model-card", type=Path, required=True)
    shadow.add_argument(
        "--alpha-signing-key",
        type=Path,
        default=DEFAULT_ROOT / "alpha-signing.key",
    )
    shadow.add_argument("--reliability-document", type=Path, required=True)
    shadow.add_argument("--reliability-key", type=Path, required=True)
    shadow.add_argument("--project", default="trade_agent_canonical")
    shadow.add_argument(
        "--env-file",
        type=Path,
        default=SCRIPT_ROOT / ".local" / "canonical.env",
    )
    shadow.add_argument(
        "--runtime-url",
        default="http://127.0.0.1:8001/api/v1/runtime",
    )
    shadow.add_argument("--state-directory", type=Path)
    shadow.add_argument("--account-id", default="paper-main")
    shadow.add_argument("--duration-seconds", type=_positive_int, default=7 * 86400)
    shadow.add_argument("--sample-seconds", type=_positive_int, default=60)
    shadow.add_argument("--sample-coverage-ratio", type=_unit_ratio, default=0.90)
    shadow.add_argument("--availability-ratio", type=_unit_ratio, default=0.995)
    shadow.add_argument("--max-drawdown", type=_unit_ratio, default=0.15)
    shadow.add_argument("--min-candidate-events", type=_positive_int, default=1)
    shadow.add_argument("--min-fills", type=_positive_int, default=1)
    shadow.add_argument("--start-runtime", action="store_true")

    promote = commands.add_parser("promote")
    promote.add_argument("--model-card", type=Path, required=True)
    promote.add_argument(
        "--alpha-signing-key",
        type=Path,
        default=DEFAULT_ROOT / "alpha-signing.key",
    )
    promote.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_ROOT / "registry",
    )
    promote.add_argument("--reliability-document", type=Path, required=True)
    promote.add_argument("--reliability-key", type=Path, required=True)
    promote.add_argument("--shadow-document", type=Path, required=True)
    promote.add_argument("--shadow-key", type=Path, required=True)

    shadow_verify = commands.add_parser("shadow-verify")
    shadow_verify.add_argument("--shadow-document", type=Path, required=True)
    shadow_verify.add_argument("--shadow-key", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--model-card", type=Path, required=True)
    verify.add_argument(
        "--signing-key",
        type=Path,
        default=DEFAULT_ROOT / "alpha-signing.key",
    )
    verify.add_argument("--artifact", type=Path)
    return result


async def _download(args: argparse.Namespace) -> int:
    request = HistoricalRequest(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        gap_policy=GapPolicy(args.gap_policy),
        allowed_gaps=tuple(args.allow_gap),
    )
    snapshot = await BinanceHistoricalDownloader().download(
        request,
        args.output_directory,
    )
    print(
        json.dumps(
            {
                "manifest": str(snapshot.manifest_path),
                "manifest_sha256": snapshot.manifest_sha256,
                "data": str(snapshot.data_path),
                "data_sha256": snapshot.data_sha256,
                "rows": len(snapshot.candles),
                "gaps": [value.isoformat() for value in snapshot.gaps],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


async def _replay(args: argparse.Namespace) -> int:
    snapshot = load_snapshot(args.dataset_manifest)
    constraints = _constraints(snapshot.request.symbol)
    costs = ReplayCostModel(
        taker_fee_bps=Decimal(args.fee_bps),
        spread_bps=Decimal(args.spread_bps),
        slippage_bps=Decimal(args.slippage_bps),
    )
    client: OllamaCandidateClient | None = None
    if args.provider == "deterministic_baseline":
        fallback_path = SCRIPT_ROOT / "workers" / "candidate" / "fallback.py"
        canonical = DeterministicBaselineCandidateProducer(
            artifact_digest=hashlib.sha256(fallback_path.read_bytes()).hexdigest()
        )
    else:
        client = OllamaCandidateClient(base_url=args.ollama_url)
        canonical = TypedCandidateProducer(
            client,
            model_digest=await client.model_digest(),
        )
    try:
        result = await CanonicalAlphaReplay(
            ReplayConfig(
                symbol=snapshot.request.symbol,
                interval=snapshot.request.interval,
                interval_seconds=snapshot.request.interval_seconds,
                constraints=constraints,
                risk_policy=_risk_policy(),
                initial_equity=Decimal(args.initial_equity),
                costs=costs,
                max_holding_bars=args.max_holding_bars,
            )
        ).run(snapshot.candles, CanonicalCandidateAdapter(canonical))
    finally:
        if client is not None:
            await client.aclose()
    payload = {
        "dataset_sha256": snapshot.data_sha256,
        "provider": args.provider,
        "costs": costs.to_payload(),
        "metrics": result.metrics.to_payload(),
        "trace_digest": result.trace_digest,
    }
    encoded = (
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True).encode() + b"\n"
    )
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(encoded)
    print(encoded.decode(), end="")
    return 0


async def _certify(args: argparse.Namespace) -> int:
    _verify_source_lock()
    snapshots = tuple(load_snapshot(path) for path in args.dataset_manifest)
    constraints = {
        snapshot.request.symbol: _constraints(snapshot.request.symbol)
        for snapshot in snapshots
    }
    max_interval_seconds = max(
        snapshot.request.interval_seconds for snapshot in snapshots
    )
    purge = (
        timedelta(hours=args.purge_hours)
        if args.purge_hours is not None
        else timedelta(seconds=max_interval_seconds * args.horizon_bars)
    )
    signing_key = _load_or_create_key(args.signing_key)
    reliability_document: dict[str, Any] | None = None
    reliability_key: bytes | None = None
    shadow_document: dict[str, Any] | None = None
    shadow_key: bytes | None = None
    if bool(args.reliability_document) != bool(args.reliability_key):
        raise ValueError(
            "reliability-document and reliability-key must be supplied together"
        )
    if args.reliability_document:
        reliability_document = json.loads(args.reliability_document.read_text())
        reliability_key = args.reliability_key.read_bytes()
    if bool(args.shadow_document) != bool(args.shadow_key):
        raise ValueError("shadow-document and shadow-key must be supplied together")
    if args.shadow_document:
        shadow_document = json.loads(args.shadow_document.read_text())
        shadow_key = args.shadow_key.read_bytes()
    controller = AlphaCertificationController(
        config=AlphaCertificationConfig(
            walk_forward=WalkForwardConfig(
                train_window=timedelta(days=args.train_days),
                test_window=timedelta(days=args.test_days),
                purge=purge,
                embargo=timedelta(hours=args.embargo_hours),
                min_train_samples=args.min_training_samples,
                min_test_samples=args.min_test_samples,
                expanding=args.expanding,
            ),
            training=CandidateTrainingConfig(
                horizon_bars=args.horizon_bars,
                min_training_samples=args.min_training_samples,
                n_estimators=args.n_estimators,
            ),
            gates=AlphaGateThresholds(
                min_folds=args.min_folds,
                min_trades_per_fold=args.min_trades,
                min_profit_factor=args.min_profit_factor,
                max_drawdown=args.max_drawdown,
                min_fold_pass_ratio=args.min_fold_pass_ratio,
                min_positive_fold_ratio=args.min_positive_fold_ratio,
                max_worst_fold_loss=args.max_worst_fold_loss,
                min_baseline_excess_return=args.min_baseline_excess_return,
                min_oos_regimes=args.min_oos_regimes,
                min_regimes_per_fold=args.min_regimes_per_fold,
                min_shadow_seconds=args.min_shadow_days * 24 * 60 * 60,
            ),
            costs=ReplayCostModel(
                taker_fee_bps=Decimal(args.fee_bps),
                spread_bps=Decimal(args.spread_bps),
                slippage_bps=Decimal(args.slippage_bps),
            ),
            cost_stress_multiplier=args.cost_stress_multiplier,
            candidate_workspace=args.candidate_workspace,
        ),
        constraints_by_symbol=constraints,
        risk_policy=_risk_policy(),
        registry=PromotionRegistry(args.registry, signing_key),
    )
    result = await controller.certify(
        snapshots,
        git_commit=_git_commit(),
        code_sha256=_code_sha256(),
        source_lock_sha256=_source_lock_sha256(),
        reliability_document=reliability_document,
        reliability_key=reliability_key,
        shadow_document=shadow_document,
        shadow_key=shadow_key,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "model_card": str(result.report_path),
                "staged_model": (
                    None
                    if result.staged_model_path is None
                    else str(result.staged_model_path)
                ),
                "promoted_model": (
                    None
                    if result.promoted_model_path is None
                    else str(result.promoted_model_path)
                ),
                "folds": len(result.fold_metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result.status == "promoted":
        return 0
    if result.status in {
        "alpha_certified_awaiting_reliability",
        "alpha_certified_awaiting_shadow",
    }:
        return 3
    return 1


def _document(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f"evidence file is too large: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence root must be an object: {path.name}")
    return value


def _shadow_run(args: argparse.Namespace) -> int:
    _verify_source_lock()
    alpha_card = _document(args.model_card)
    alpha_key = args.alpha_signing_key.read_bytes()
    if not PromotionRegistry.verify_card(alpha_card, alpha_key):
        raise ValueError("alpha model card signature is invalid")
    alpha_gates = alpha_card.get("alpha_gates")
    if not isinstance(alpha_gates, dict) or alpha_gates.get("passed") is not True:
        raise ValueError("alpha model card did not pass offline gates")
    current_git_commit = _git_commit()
    current_code_sha256 = _code_sha256()
    if alpha_card.get("git_commit") != current_git_commit:
        raise ValueError("repository commit differs from the alpha model card")
    if alpha_card.get("code_sha256") != current_code_sha256:
        raise ValueError("runtime code differs from the alpha model card")
    current_source_lock_sha256 = _source_lock_sha256()
    if alpha_card.get("source_lock_sha256") != current_source_lock_sha256:
        raise ValueError("source lock differs from the alpha model card")
    artifact = alpha_card.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("alpha model card is missing artifact identity")
    artifact_sha256 = str(artifact.get("artifact_sha256", ""))
    reliability = _document(args.reliability_document)
    reliability_key = args.reliability_key.read_bytes()
    if not reliability_is_promoted(
        reliability,
        reliability_key,
        git_commit=current_git_commit,
        source_lock_sha256=current_source_lock_sha256,
    ):
        raise ValueError("promoted reliability evidence must match Git and source lock")
    reliability_sha256 = hashlib.sha256(canonical_json(reliability)).hexdigest()
    alpha_card_sha256 = hashlib.sha256(canonical_json(alpha_card)).hexdigest()
    state_directory = args.state_directory or DEFAULT_ROOT / "shadow" / artifact_sha256
    runtime = DockerRuntime(
        RuntimeConfig(
            project=args.project,
            env_file=args.env_file.resolve(),
            runtime_url=args.runtime_url,
        )
    )
    if args.start_runtime:
        runtime.start()
    controller = AlphaShadowCertificationController(
        sampler=AlphaShadowDockerSampler(
            runtime,
            artifact_sha256=artifact_sha256,
            account_id=args.account_id,
        ),
        paths=ShadowControllerPaths(state_directory),
        thresholds=ShadowThresholds(
            duration_seconds=args.duration_seconds,
            sample_seconds=args.sample_seconds,
            sample_coverage_ratio=args.sample_coverage_ratio,
            availability_ratio=args.availability_ratio,
            max_drawdown=args.max_drawdown,
            min_candidate_events=args.min_candidate_events,
            min_fills=args.min_fills,
        ),
        artifact_sha256=artifact_sha256,
        alpha_model_card_sha256=alpha_card_sha256,
        reliability_document_sha256=reliability_sha256,
        git_commit=current_git_commit,
        code_sha256=current_code_sha256,
    )
    return controller.run()


def _promote(args: argparse.Namespace) -> int:
    _verify_source_lock()
    alpha_card = _document(args.model_card)
    current_git_commit = _git_commit()
    current_code_sha256 = _code_sha256()
    current_source_lock_sha256 = _source_lock_sha256()
    if alpha_card.get("git_commit") != current_git_commit:
        raise ValueError("repository commit differs from the alpha model card")
    if alpha_card.get("code_sha256") != current_code_sha256:
        raise ValueError("runtime code differs from the alpha model card")
    if alpha_card.get("source_lock_sha256") != current_source_lock_sha256:
        raise ValueError("source lock differs from the alpha model card")
    reliability = _document(args.reliability_document)
    shadow = _document(args.shadow_document)
    registry = PromotionRegistry(
        args.registry,
        args.alpha_signing_key.read_bytes(),
    )
    promoted, report_path, model_path = registry.promote_staged(
        alpha_model_card=alpha_card,
        reliability_document=reliability,
        reliability_key=args.reliability_key.read_bytes(),
        shadow_document=shadow,
        shadow_key=args.shadow_key.read_bytes(),
    )
    print(
        json.dumps(
            {
                "status": promoted["status"],
                "promotion_report": str(report_path),
                "promoted_model": str(model_path),
                "artifact_sha256": promoted["artifact"]["artifact_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _shadow_verify(args: argparse.Namespace) -> int:
    document = _document(args.shadow_document)
    if not verify_document(document, args.shadow_key.read_bytes()):
        print("signature=invalid", file=sys.stderr)
        return 1
    state = document.get("shadow_state")
    print("signature=valid")
    print(f"status={state.get('status') if isinstance(state, dict) else 'invalid'}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    document = json.loads(args.model_card.read_text())
    key = args.signing_key.read_bytes()
    if not PromotionRegistry.verify_card(document, key):
        print("signature=invalid", file=sys.stderr)
        return 1
    expected = document.get("artifact", {}).get("artifact_sha256")
    if args.artifact is not None:
        actual = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
        if actual != expected:
            print("artifact=invalid", file=sys.stderr)
            return 1
    print("signature=valid")
    print(f"status={document.get('status')}")
    print(f"artifact_sha256={expected}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "download":
            return asyncio.run(_download(args))
        if args.command == "replay":
            return asyncio.run(_replay(args))
        if args.command == "certify":
            return asyncio.run(_certify(args))
        if args.command == "shadow-run":
            return _shadow_run(args)
        if args.command == "promote":
            return _promote(args)
        if args.command == "shadow-verify":
            return _shadow_verify(args)
        return _verify(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            f"alpha certification failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
