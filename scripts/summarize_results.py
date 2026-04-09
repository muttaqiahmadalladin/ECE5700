#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ece570_vit_adapters.reporting import summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate result.json files into tables and plots.")
    parser.add_argument("--results-root", required=True, help="Root directory containing experiment outputs.")
    return parser.parse_args()


def main() -> int:
    summary_files = summarize_results(Path(parse_args().results_root))
    for label, path in summary_files.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

