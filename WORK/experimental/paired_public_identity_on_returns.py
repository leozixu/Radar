"""Check if public identity protection adds value after paired-return assignment."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from lightgbm import Booster
from paired_public_identity import compare, summarize
from paired_returns import ASSETS, HERE, BirthOnline, PairReturns
from algorithm import load_runtime
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from replay_validation import file_sha256


def evaluate(path):
    weights = [Booster(model_file=str(asset)) for asset in ASSETS.values()]
    paired = PairReturns(Booster(model_file=str(HERE / "models/paired_returns.txt")), .985, .95, early=True)
    tracking = BirthOnline(*weights, .86, "supplement", paired_birth=True, joint=True, two_hit_pairs=True, pairing=paired)
    return compare(path, tracking.track, load_runtime().process_circle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    sources = ("paired_public_identity.py", "paired_public_identity_on_returns.py", "paired_returns.py",
               "causal_birth_paths.py")
    source_sha256 = {name: file_sha256(HERE / name) for name in sources}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        evaluations = list(pool.map(evaluate, paths))
    report = summarize(evaluations, args.packets, ASSETS | {"paired_returns": HERE / "models/paired_returns.txt"})
    report.update({"source_sha256": source_sha256, "start_floor": .985, "link_floor": .95, "early": True})
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
