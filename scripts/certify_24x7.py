"""Run or inspect the resumable 24h-chaos then 7d-paper certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from packages.certification.controller import (  # noqa: E402
    ROOT,
    CertificationController,
    ControllerPaths,
    DockerRuntime,
    RuntimeConfig,
    read_status,
)
from packages.certification.core import Thresholds  # noqa: E402


DEFAULT_STATE_DIRECTORY = ROOT / ".local" / "certification"


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1]")
    return parsed


def _paths(args: argparse.Namespace) -> ControllerPaths:
    return ControllerPaths(
        state_directory=args.state_directory.resolve(),
        restore_evidence=args.restore_evidence.resolve(),
        alert_webhook_url_file=(
            args.alert_webhook_url_file.resolve()
            if args.alert_webhook_url_file
            else None
        ),
        promoted_artifact=args.promoted_artifact.resolve(),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--state-directory",
        type=Path,
        default=DEFAULT_STATE_DIRECTORY,
    )
    shared.add_argument(
        "--restore-evidence",
        type=Path,
        default=ROOT / "artifacts" / "restore-drill-latest.json",
    )
    shared.add_argument("--alert-webhook-url-file", type=Path)
    shared.add_argument(
        "--promoted-artifact",
        type=Path,
        default=ROOT / "artifacts" / "24x7-certification-latest.json",
    )

    run = commands.add_parser("run", parents=[shared])
    run.add_argument("--project", default="trade_agent_canonical")
    run.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".local" / "canonical.env",
    )
    run.add_argument(
        "--runtime-url",
        default="http://127.0.0.1:8001/api/v1/runtime",
    )
    run.add_argument("--start-runtime", action="store_true")
    run.add_argument("--chaos-seconds", type=_positive, default=24 * 60 * 60)
    run.add_argument("--paper-seconds", type=_positive, default=7 * 24 * 60 * 60)
    run.add_argument("--sample-seconds", type=_positive, default=60)
    run.add_argument("--outage-seconds", type=_positive, default=30)
    run.add_argument("--readiness-timeout-seconds", type=_positive, default=300)
    run.add_argument("--availability-ratio", type=_ratio, default=0.995)
    run.add_argument("--sample-coverage-ratio", type=_ratio, default=0.90)
    run.add_argument("--max-restart-delta", type=int, default=3)
    run.add_argument("--max-recovery-seconds", type=_positive, default=300)
    run.add_argument("--max-rpo-seconds", type=_positive, default=60 * 60)
    run.add_argument("--max-rto-seconds", type=_positive, default=30 * 60)
    run.add_argument(
        "--restore-evidence-max-age-seconds",
        type=_positive,
        default=24 * 60 * 60,
    )

    commands.add_parser("status", parents=[shared])
    commands.add_parser("verify", parents=[shared])
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = _paths(args)
    if args.command in {"status", "verify"}:
        try:
            document = read_status(paths, verify=True)
        except Exception as exc:
            print(
                f"certification status failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        if args.command == "verify":
            print("signature=valid")
            print(f"status={document['certification_state']['status']}")
        else:
            print(json.dumps(document, indent=2, sort_keys=True))
        return 0

    if args.max_restart_delta < 0:
        print("max restart delta must be non-negative", file=sys.stderr)
        return 2
    thresholds = Thresholds(
        chaos_seconds=args.chaos_seconds,
        paper_seconds=args.paper_seconds,
        sample_seconds=args.sample_seconds,
        availability_ratio=args.availability_ratio,
        sample_coverage_ratio=args.sample_coverage_ratio,
        max_restart_delta=args.max_restart_delta,
        max_recovery_seconds=args.max_recovery_seconds,
        max_rpo_seconds=args.max_rpo_seconds,
        max_rto_seconds=args.max_rto_seconds,
        restore_evidence_max_age_seconds=args.restore_evidence_max_age_seconds,
    )
    runtime = DockerRuntime(
        RuntimeConfig(
            project=args.project,
            env_file=args.env_file.resolve(),
            runtime_url=args.runtime_url,
            outage_seconds=args.outage_seconds,
            readiness_timeout_seconds=args.readiness_timeout_seconds,
        )
    )
    controller = CertificationController(
        runtime=runtime,
        paths=paths,
        thresholds=thresholds,
        start_runtime=args.start_runtime,
    )
    try:
        return controller.run()
    except KeyboardInterrupt:
        print("certification interrupted; state is resumable", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"certification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
