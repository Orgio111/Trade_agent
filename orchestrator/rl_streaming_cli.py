"""
CLI Entry Point for Real-Time RL Streaming + Meta-Learning + SEG v1.

Usage:
    python -m orchestrator.rl_streaming_cli rl_streaming daemon
    python -m orchestrator.rl_streaming_cli meta_learner analyze
    python -m orchestrator.rl_streaming_cli seg generate --count 10
    python -m orchestrator.rl_streaming_cli seg backtest --strategy strategies/pool/best.json
    python -m orchestrator.rl_streaming_cli seg evolve --generations 50
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import and run CLI
from orchestrator.rl_streaming.cli_commands import cli

if __name__ == "__main__":
    cli()