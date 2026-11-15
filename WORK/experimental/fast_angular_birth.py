"""Confirm high-angular-speed births left open by the ordinary three-hit tracker."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lightgbm import Booster

from paired_returns import ASSETS, HERE, PairReturns
from causal_birth_paths import LINK_WIDTH, POINT, POSITION, SCAN, BirthOnline, parallel
from raw_birth_stage_audit import EMISSION_MODELS, PAIR_MODEL, PROFILE, RAW_MODEL, emissions, models
from raw_timing_detection import RawReturnGate
from emission_identity import EmissionFeatures
from identity_rejoin import load_quality_reference
from paired_single_start import mature_competitor
from baseline_cv import OUTPUT_FIELDS, residual
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from tracker_assoc import spatial_edges
import replay_validation as replay
import tracker_compat


WIDE_FIRST = np.array([.110, .220, .022])
PRECISE_THIRD = np.array([.024, .105, .0046])
RETURN_FLOOR = .98
MARGIN = .14


class FastAngularBirths(BirthOnline):
    """Use only still-unclaimed returns, after completing existing births and associations."""

    def __init__(self, *upstream, **kwargs):
        super().__init__(*upstream, **kwargs)
        self.angular_starts = 0
        self.added = set()

    def track(self, points, state, observe=None):
        outputs = super().track(points, state, observe)
        circle = state["circle"]
        recent = state.setdefault("fast_angular_returns", {})
        for old in tuple(recent):
            if circle - old > 2:
                del recent[old]
        q = state["raw_return_probability"]
        recent[circle] = {point["PntID"] for point, value in zip(points, q) if value >= RETURN_FLOOR}
        pool = state["birth_path_pool"]
        scans = {scan: rows[np.isin(rows[:, POINT], tuple(recent.get(scan, ())))]
                 for scan, rows in pool.observations}
        if any(not len(scans.get(scan, ())) for scan in range(circle - 2, circle + 1)):
            return outputs
        first, second, third = (scans[scan] for scan in range(circle - 2, circle + 1))
        active, tentatives, seeds = state["tracks"]
        # Dense bursts remain on the standard path. A confirmed incumbent vetoes the fallback.
        if sum(map(len, (first, second, third))) > 500:
            return outputs
        third = third[~mature_competitor(active, third, circle)]
        trajectories = self.propose(first, second, third)
        if not trajectories:
            return outputs
        next_id = state["next_batch_id"]
        matched, claimed = {}, set()
        for history in trajectories:
            self.confirm(history, active, matched, next_id, observe)
            next_id += 1
            claimed.update(int(row[POINT]) for row in history)
        state["next_batch_id"] = next_id
        pool.add(np.empty((0, 7)), circle, np.fromiter(claimed, dtype=int))
        history = state["young_keys"]
        tentatives, seeds = ([track for track in tracks if not claimed.intersection(history[id(track)])]
                             for tracks in (tentatives, seeds))
        state["tracks"] = active, tentatives, seeds
        state["young_keys"] = {id(track): history[id(track)] for track in (*tentatives, *seeds)}
        state["previous_pairs"] = [
            replace(group, owners=tuple(matched.get(int(pid), old)
                                        for pid, old in zip(group.observations[:, POINT], group.owners)))
            if group.observations[0, SCAN] == circle else group
            for group in state["previous_pairs"]
        ]
        self.angular_starts += len(trajectories)
        self.added.update(matched)
        outputs.extend({**{key: point[key] for key in OUTPUT_FIELDS}, "BatchID": matched[point["PntID"]]}
                       for point in points if point["PntID"] in matched)
        return outputs

    @staticmethod
    def propose(first, second, third):
        if not len(third):
            return []
        rows, cols = spatial_edges(second[:, POSITION], first[:, POSITION], WIDE_FIRST[None, :])
        velocity = residual(second[cols, POSITION], first[rows, POSITION])
        distance = np.square(velocity / WIDE_FIRST).sum(axis=1)
        fast = np.square(velocity / LINK_WIDTH).sum(axis=1) > 1.
        keep = (distance <= 1.) & fast
        if not keep.any():
            return []
        a, b, speed, start_cost = first[rows[keep]], second[cols[keep]], velocity[keep], distance[keep]
        forecast = b[:, POSITION] + speed
        rows, cols = spatial_edges(third[:, POSITION], forecast, PRECISE_THIRD[None, :])
        error = residual(third[cols, POSITION], forecast[rows])
        tail_cost = np.square(error / PRECISE_THIRD).sum(axis=1)
        keep = tail_cost <= 1.
        if not keep.any():
            return []
        first, second, third = a[rows[keep]], b[rows[keep]], third[cols[keep]]
        cost = tail_cost[keep] + .15 * start_cost[rows[keep]]
        order = np.argsort(cost, kind="stable")
        resources = np.column_stack((first[:, POINT], second[:, POINT], third[:, POINT])).astype(int)
        rivals = {}
        for i in order:
            for point in resources[i]:
                contenders = rivals.setdefault(point, [])
                if len(contenders) < 2:
                    contenders.append(i)
        return [(first[i], second[i], third[i]) for i in order
                if all(rivals[point][0] == i and
                       (len(rivals[point]) == 1 or cost[rivals[point][1]] - cost[i] >= MARGIN)
                       for point in resources[i])]


def evaluate(spec):
    path, trial, allowed = spec
    weights = models()
    emission_weights = {name: Booster(model_file=str(path)) for name, path in EMISSION_MODELS.items()}
    pairing = PairReturns(Booster(model_file=str(PAIR_MODEL)), .985, .95, wide=True,
                          fallback_floor=PROFILE["fallback_floor"])
    gate = RawReturnGate(Booster(model_file=str(RAW_MODEL)), PROFILE["raw_floor"])
    engine = FastAngularBirths if trial else BirthOnline
    tracker = engine(*weights.values(), PROFILE["birth_floor"], "supplement", paired_birth=True, joint=True,
                     two_hit_pairs=True, pairing=pairing, raw_gate=gate, association_floor=PROFILE["association_floor"])
    history, reference = EmissionFeatures(), load_quality_reference()
    labels, circles, times, published = {}, {}, {}, {}
    state = {}
    with patch.object(tracker_compat, "baseline_circle", tracker.track):
        for _, payload in replay.iter_circle_payloads(path, labels, circles, allowed):
            points = [dict(zip(replay.POINT_KEYS, row)) for row in replay.POINT.iter_unpack(payload)]
            times.update((p["PntID"], p["Time"]) for p in points)
            proposals, quality = tracker_compat.decisions_with_features(points, state)
            source = {(p["Time"], p["CircleID"], p["PntID"]): (p["Azimuth"], p["Elevation"], p["Distance"])
                      for p in points}
            frame = replay.encode_tracks(proposals, source)
            accepted = emissions(points, state, proposals, quality, history, weights["clock"],
                                 emission_weights, reference)
            published.update((row[-1], row[-3]) for row in replay.TRACK.iter_unpack(memoryview(frame)[5:-4])
                             if row[-1] in accepted)
    result = evaluate_packet(labels, published, circles, times)
    additions = tracker.added if trial else set()
    stats = Counter("truth" if labels[pid] > 0 else "clutter" if labels[pid] == 0 else "unknown" for pid in additions)
    print(path.name, "angular" if trial else "control", round(result["score"]["total"], 5), dict(stats), flush=True)
    return result, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--phase", choices=("train", "tune"), required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    allowed = TRAIN_PACKETS if args.phase == "train" else TUNING_PACKETS
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, allowed)
    results = parallel(evaluate, [(path, trial, allowed) for trial in (False, True) for path in paths], args.jobs)
    split = len(paths)
    report = {"phase": args.phase, "packets": args.packets, "score_version": SCORE_VERSION, "profile": PROFILE,
              "strategy": {"first_width": WIDE_FIRST.tolist(), "third_width": PRECISE_THIRD.tolist(),
                           "raw_floor": RETURN_FLOOR, "margin": MARGIN},
              "control": aggregate_packets(row[0] for row in results[:split]),
              "trial": aggregate_packets(row[0] for row in results[split:]),
              "direct_start_labels": sum((row[1] for row in results[split:]), Counter()),
              "source_sha256": {name: replay.file_sha256(HERE / name) for name in (
                  Path(__file__).name, "causal_birth_paths.py", "paired_returns.py", "raw_birth_stage_audit.py",
                  "emission_identity.py")},
              "model_sha256": {key: replay.file_sha256(path) for key, path in (
                  ASSETS | {"paired_returns": PAIR_MODEL, "raw_timing": RAW_MODEL} | EMISSION_MODELS).items()},
              }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
