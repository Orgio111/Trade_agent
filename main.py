"""
QUANTEX Trading System — Root CLI entry point.
Orchestrates the full multi-language stack: Rust execution, Python AI, Go realtime.
"""
import os
import sys
import signal
import subprocess
import argparse
from pathlib import Path


def banner():
    print(r"""
╔═══════════════════════════════════════════════════════════╗
║                     QUANTEX v0.1.0                        ║
║           Autonomous AI Trading System                    ║
╚═══════════════════════════════════════════════════════════╝
    """)


def start_infra():
    """Start Docker Compose infrastructure."""
    print("🏗️  Starting infrastructure (PostgreSQL, Redis, NATS)...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "postgres", "redis", "nats"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠️  Infra warning: {result.stderr}")


def start_all():
    """Start all services."""
    banner()
    start_infra()

    # Start Go realtime service
    print("🚀 Starting Go realtime service...")
    go_path = Path("realtime")
    go_proc = subprocess.Popen(
        ["go", "run", "."],
        cwd=go_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Start Python orchestrator
    print("🚀 Starting Python orchestrator...")
    py_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "orchestrator.main:app",
         "--host", "0.0.0.0", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print("\n✅ All services started")
    print("   Orchestrator:  http://localhost:8001")
    print("   Realtime WS:   ws://localhost:8082/ws")
    print("   Health:        http://localhost:8082/health")
    print("\nPress Ctrl+C to stop all services\n")

    def shutdown(sig, frame):
        print("\nShutting down...")
        py_proc.terminate()
        go_proc.terminate()
        subprocess.run(["docker", "compose", "down"], capture_output=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        py_proc.wait()
    except KeyboardInterrupt:
        shutdown(None, None)


def build_rust():
    """Build Rust execution engine."""
    print("🔧 Building Rust execution engine...")
    rust_path = Path("execution")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=rust_path,
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode == 0:
        print("✅ Rust engine built successfully")
    else:
        print(f"❌ Build failed:\n{result.stderr}")


def build_frontend():
    """Build TypeScript frontend."""
    print("🔧 Building frontend...")
    frontend_path = Path("frontend")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=frontend_path,
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode == 0:
        print("✅ Frontend built successfully")
    else:
        print(f"❌ Build failed:\n{result.stderr}")


def cli():
    parser = argparse.ArgumentParser(description="QUANTEX Trading System")
    parser.add_argument("command", nargs="?", default="start",
                        choices=["start", "build", "build-rust", "build-frontend", "infra"])
    args = parser.parse_args()

    if args.command == "start":
        start_all()
    elif args.command == "build":
        build_rust()
        build_frontend()
    elif args.command == "build-rust":
        build_rust()
    elif args.command == "build-frontend":
        build_frontend()
    elif args.command == "infra":
        start_infra()


if __name__ == "__main__":
    cli()
