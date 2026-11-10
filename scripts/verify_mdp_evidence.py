"""Compare two benchmark runs, ignoring machine-dependent timing fields.

Usage: python scripts/verify_mdp_evidence.py first.json second.json
"""

import argparse
import json
from pathlib import Path


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if key not in ("elapsed_ms", "median_elapsed_ms")}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first = without_timing(json.loads(args.first.read_text(encoding="utf-8")))
    second = without_timing(json.loads(args.second.read_text(encoding="utf-8")))
    if first != second:
        raise SystemExit("FAIL: quality metrics, actions or provenance differ")
    print("PASS: all quality metrics, chains, intervals and provenance match")


if __name__ == "__main__":
    main()
