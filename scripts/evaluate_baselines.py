"""Run benchmark comparisons between Baselines and Production Architecture."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Add project root to sys.path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark import BenchmarkSuite


if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_all()

