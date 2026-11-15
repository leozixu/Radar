"""Recover rapid paired births using the exact signed amplitude signature."""

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lightgbm import Booster

from paired_returns import ASSETS, HERE, PairReturns, pair_offset
from causal_birth_paths import POINT, POSITION, SCAN, BirthOnline, parallel
from paired_single_start import mature_competitor
from raw_birth_stage_audit import EMISSION_MODELS, PAIR_MODEL, PROFILE, RAW_MODEL, emissions, models
from raw_timing_detection import RawReturnGate
from emission_identity import EmissionFeatures
from identity_rejoin import load_quality_reference
from baseline_cv import residual
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
import replay_validation as replay
import tracker_compat


ORDINARY_WIDTH = np.array([.038, .140, .020])
FAST_WIDTH = np.array([.110, .220, .020])
ROLE_WIDTH = np.array([.004, .0018, .0025])
AMPLITUDE_ERROR = .0000026


class FastPairCenter(PairReturns):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = []

    def begin(self, tracks, raw, times, circle, state):
        self.raw = raw
        frame = super().begin(tracks, raw, times, circle, state)
        used_prior = {id(prior) for prior, _ in frame.decisions}
        used_points = {point for _, index in frame.decisions for point in frame.columns[index]}
        early = [prior for prior in state.get("previous_pairs", ())
                 if id(prior) not in used_prior and not any(prior.owners)
                 and prior.observations[0, SCAN] == circle - 1]
        open_pairs = np.array([index for index, (a, b) in enumerate(frame.columns[:frame.strict_count])
                               if a not in used_points and b not in used_points], dtype=int)
        self.fast = set()
        if not early or not len(open_pairs):
            return frame

        before = np.stack([prior.observations for prior in early])
        after = raw[frame.columns[open_pairs]]
        past_amp, current_amp = pair_offset(before), pair_offset(after)
        past_role = residual(before[:, 1, POSITION], before[:, 0, POSITION])
        current_role = residual(after[:, 1, POSITION], after[:, 0, POSITION])
        rows, cols = np.nonzero(
            (np.abs(past_amp[:, None] - current_amp[None, :]) <= AMPLITUDE_ERROR)
            & (past_amp[:, None] < 0.) & (current_amp[None, :] < 0.)
            & (past_role[:, None, 1] >= 0.) & (current_role[None, :, 1] >= 0.))
        if not len(rows):
            return frame
        movement = np.abs(residual(after[cols, :, POSITION], before[rows, :, POSITION]).mean(axis=1))
        shape = np.abs(residual(current_role[cols], past_role[rows]))
        valid = (~(movement <= ORDINARY_WIDTH).all(axis=1) & (movement <= FAST_WIDTH).all(axis=1)
                 & (shape <= ROLE_WIDTH).all(axis=1))
        rows, cols = rows[valid], cols[valid]
        if not len(rows):
            return frame
        occupied = mature_competitor(tracks, after[cols].reshape(-1, 7), circle).reshape(-1, 2).any(axis=1)
        rows, cols = rows[~occupied], cols[~occupied]
        prior_counts = Counter(int(row) for row in rows)
        point_counts = Counter(int(point) for col in cols for point in frame.columns[open_pairs[col]])
        for row, col in zip(rows, cols):
            index = int(open_pairs[col])
            if prior_counts[int(row)] != 1 or any(point_counts[int(point)] != 1 for point in frame.columns[index]):
                continue
            self.fast.add(index)
            frame.decisions.append((early[row], index))
        return frame

    def starts(self, frame, remaining):
        fast_points = {tuple(frame.columns[index]) for index in self.fast}
        for before, current in super().starts(frame, remaining):
            if tuple(current) in fast_points:
                previous_ids = tuple(before[:, POINT].astype(int))
                self.started.append(previous_ids + tuple(self.raw[current, POINT].astype(int)))
            yield before, current


def evaluate(spec):
    path, trial, allowed = spec
    weights = models()
    emission_weights = {name: Booster(model_file=str(path)) for name, path in EMISSION_MODELS.items()}
    pair_type = FastPairCenter if trial else PairReturns
    pairing = pair_type(Booster(model_file=str(PAIR_MODEL)), .985, .95, wide=True,
                        fallback_floor=PROFILE["fallback_floor"])
    gate = RawReturnGate(Booster(model_file=str(RAW_MODEL)), PROFILE["raw_floor"])
    tracker = BirthOnline(*weights.values(), PROFILE["birth_floor"], "supplement", paired_birth=True, joint=True,
                          two_hit_pairs=True, pairing=pairing, raw_gate=gate,
                          association_floor=PROFILE["association_floor"])
    history, reference = EmissionFeatures(), load_quality_reference()
    labels, circles, times, published, state = {}, {}, {}, {}, {}
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
    stats = Counter()
    if trial:
        for a, b, c, d in pairing.started:
            stats["confirmed_pairs"] += 1
            stats["correct_positive_pairs"] += labels[a] == labels[c] > 0 and labels[b] == labels[d] > 0
            stats["public_birth_points"] += (c in published) + (d in published)
    print(path.name, "wide_center" if trial else "control", round(result["score"]["total"], 5),
          result["id_switches"], dict(stats), flush=True)
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
    report = {
        "phase": args.phase, "packets": args.packets, "score_version": SCORE_VERSION, "profile": PROFILE,
        "strategy": {"center_width": FAST_WIDTH.tolist(), "role_width": ROLE_WIDTH.tolist(),
                     "amplitude_error": AMPLITUDE_ERROR},
        "control": aggregate_packets(row[0] for row in results[:split]),
        "trial": aggregate_packets(row[0] for row in results[split:]),
        "direct_start_labels": sum((row[1] for row in results[split:]), Counter()),
        "per_packet": {str(packet): {"control": base[0]["score"]["total"], "trial": cand[0]["score"]["total"],
                                     "starts": dict(cand[1])}
                       for packet, base, cand in zip(args.packets, results[:split], results[split:])},
        "source_sha256": {name: replay.file_sha256(HERE / name) for name in (
            Path(__file__).name, "causal_birth_paths.py", "paired_returns.py", "raw_birth_stage_audit.py",
            "emission_identity.py")},
        "model_sha256": {key: replay.file_sha256(path) for key, path in (
            ASSETS | {"paired_returns": PAIR_MODEL, "raw_timing": RAW_MODEL} | EMISSION_MODELS).items()},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
