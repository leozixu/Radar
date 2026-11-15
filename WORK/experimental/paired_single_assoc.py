"""Replay changes to single-track association after trusted paired-return locks."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from joint_birth_residual import ResidualAfterLocks
from paired_returns import ASSETS, HERE, PairReturns
from causal_birth_paths import BirthOnline, parallel
from algorithm import load_runtime
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay
import tracker_compat


def evaluate(spec):
    from lightgbm import Booster

    path, args = spec
    upstream = [Booster(model_file=str(asset)) for asset in ASSETS.values()]
    pair_returns = PairReturns(Booster(model_file=str(args.paired_returns)), .985, .95)
    tracker = BirthOnline(*upstream, .86, "supplement", paired_birth=True, joint=True, two_hit_pairs=True,
                          pairing=pair_returns)
    if args.blend:
        model = Booster(model_file=str(args.residual))
        tracker.matcher.base = ResidualAfterLocks(tracker.matcher.base, model, args.blend)
    with (patch.object(tracker_compat, "baseline_circle", tracker.track),
          patch.object(replay, "load_handler", return_value=load_runtime().process_circle)):
        result = replay.replay_packet(path, allowed=TUNING_PACKETS)
    result["pair_counters"] = dict(pair_returns.stats)
    print(path.name, round(result["score"]["total"], 5), result["id_switches"], result["split_truth_tracks"],
          round(result["max_circle_ms"], 2), result["pair_counters"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-returns", type=Path, default=HERE / "models/paired_returns.txt")
    parser.add_argument("--residual", type=Path, default=HERE / "models/closed_loop_assoc.txt")
    parser.add_argument("--packets", nargs="+", type=int, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--blend", type=float, default=.55)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.blend <= 1:
        parser.error("blend must be between 0 and 1")
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    reports = parallel(evaluate, [(path, args) for path in paths], args.jobs)
    summary = aggregate_packets(reports)
    result = {
        "metadata": dict(packets=args.packets, blend=args.blend, jobs=args.jobs, score_version=SCORE_VERSION,
                         source_sha256={path.name: replay.file_sha256(path) for path in (
                             Path(__file__), HERE / "paired_returns.py", HERE / "causal_birth_paths.py",
                             HERE / "joint_birth_residual.py", HERE / "closed_loop_assoc.py")},
                         model_sha256={key: replay.file_sha256(path) for key, path in
                                       {**ASSETS, "paired_returns": args.paired_returns,
                                        "residual": args.residual}.items()}),
        "summary": summary,
        "packets": [{key: value for key, value in item.items() if key not in ("tracks", "_circle_times")}
                    for item in reports],
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("score", summary["score"]["total"], "switches", summary["id_switches"],
          "splits", summary["split_truth_tracks"], "IDF1", summary["idf1"], flush=True)


if __name__ == "__main__":
    main()
