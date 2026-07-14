"""Dashboard Service — entry point."""

import logging
from services.dashboard import run_dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

if __name__ == "__main__":
    run_dashboard()
