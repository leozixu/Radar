"""Add causal azimuth/range residual forecasts to existing mature association."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import joint_elevation_pairs as joint
import pair_swap_probe as pairs
import teacher_assoc as teacher
from identity_rejoin import load_quality_reference
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
import replay_validation as replay
from temporal_motion_probe import WINDOW, LAGS


AXES = (0, 2)
TIME = np.arange(-WINDOW, 0)
WINDOW_WEIGHTS = 1 / WINDOW - TIME.mean() * (TIME - TIME.mean()) / np.var(TIME) / WINDOW


def forecast(history, coefficients):
    previous = np.asarray(history).copy()
    previous[:, 0] = np.unwrap(previous[:, 0], period=1.)
    windows = sliding_window_view(previous, WINDOW + 1, axis=0)
    predicted = windows[:, :, :-1] @ WINDOW_WEIGHTS
    residual = previous[WINDOW:] - predicted
    return WINDOW_WEIGHTS @ previous[-WINDOW:] + np.sum(residual * coefficients, axis=0)


class TemporalEvidence:
    def __init__(self, predictor, coefficients, gain):
        self.predictor = predictor
        self.coefficients = coefficients
        self.gain = gain
        self.history = {}
        self.edge_evidence = None
        self.statistics = Counter()
        self.update_track = teacher.update
        self.make_features = pairs.candidate_features

    def observe(self, track, position, amplitude, circle, probability=1., robust=0.):
        key = id(track)
        if key not in self.history:
            saved = deque(((scan, p[list(AXES)].copy()) for scan, _, p in track.history), maxlen=WINDOW + LAGS)
            self.history[key] = saved
        saved = self.history[key]
        if saved and saved[-1][0] != circle - 1:
            saved.clear()
        if probability >= .60:
            saved.append((circle, position[list(AXES)].copy()))
        self.update_track(track, position, amplitude, circle, probability, robust)

    def prune(self, state):
        live = {id(t) for collection in state.get("tracks", ((), (), ())) for t in collection}
        self.history = {key: history for key, history in self.history.items() if key in live}

    def candidate_features(self, tracks, positions, amplitudes, circle):
        row, col, x = self.make_features(tracks, positions, amplitudes, circle)
        self.edge_evidence = np.zeros(len(row))
        if not len(row):
            return row, col, x
        predicted = np.full((len(tracks), 2), np.nan)
        for index, track in enumerate(tracks):
            hist = self.history.get(id(track))
            if hist is not None and len(hist) == WINDOW + LAGS and hist[-1][0] == circle - 1:
                predicted[index] = forecast([p for _, p in hist], self.coefficients)
        eligible = np.isfinite(predicted[row, 0])
        self.statistics["candidate_edges"] += len(row)
        self.statistics["forecast_edges"] += int(eligible.sum())
        if not eligible.any():
            return row, col, x
        expected = predicted[row[eligible]]
        error = positions[col[eligible]][:, AXES] - expected
        error[:, 0] = (error[:, 0] + .5) % 1. - .5
        error = abs(error) / teacher.NOISE[list(AXES)]
        old_kf = x[eligible, :][:, [13, 15]]
        old_short = x[eligible, :][:, [16, 18]]
        old_best, old_worst = np.minimum(old_kf, old_short), np.maximum(old_kf, old_short)

        # An AR fit provides additional evidence only outside the envelope of
        # the two existing forecasts, and only for a noticeable tail residual.
        better = np.maximum(0., old_best - np.maximum(error, .85))
        worse = np.maximum(0., error - np.maximum(old_worst, 1.25))
        relative = np.minimum(better, 1.4) - .50 * np.minimum(worse, 1.4)
        self.edge_evidence[eligible] = np.clip(relative.sum(axis=1), -.85, 1.2) * self.gain
        self.statistics["positive_evidence"] += int((self.edge_evidence > 0).sum())
        self.statistics["negative_evidence"] += int((self.edge_evidence < 0).sum())
        return row, col, x

    def predict(self, features, num_threads):
        confidence = self.predictor.predict(features, num_threads=num_threads)
        if self.edge_evidence is None or len(self.edge_evidence) != len(confidence):
            raise RuntimeError("mature association requested probabilities with different candidate geometry")
        mask = (confidence >= .06) & (self.edge_evidence != 0)
        result = confidence.copy()
        if mask.any():
            odds = confidence[mask] / np.maximum(1.e-6, 1 - confidence[mask])
            odds *= np.exp(self.edge_evidence[mask])
            result[mask] = odds / (1 + odds)
        return result


def run(spec):
    from lightgbm import Booster

    path, args, coefficients, gain = spec
    quality = load_quality_reference()
    original = Booster(model_file=str(args.teacher))
    correction = TemporalEvidence(original, coefficients, gain)
    base = pairs.PairCorrector(Booster(model_file=str(args.pair)), .85)
    joint_matcher = joint.JointElevationMatcher(pairs, base)
    mature = teacher.Online(correction if gain else original, .20, 0., 7, .30, 4.)

    def apply_scan(points, state):
        output = quality.process_circle(points, state)
        correction.prune(state)
        return output

    with (patch.object(teacher, "assign_confirmed", joint_matcher.associate),
          patch("tracker_compat.baseline_circle", mature.track),
          patch.object(replay, "load_handler", return_value=apply_scan),
          patch.object(teacher, "update", correction.observe if gain else correction.update_track),
          patch.object(pairs, "candidate_features", correction.candidate_features if gain else correction.make_features)):
        result = replay.replay_packet(path, allowed=TUNING_PACKETS)
    print(gain, path.name, round(result["score"]["total"], 4), result["id_switches"], dict(correction.statistics),
          flush=True)
    return result, correction.statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--gains", type=float, nargs="+", default=(0., .55))
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    if min(args.gains) < 0 or args.report.suffix.lower() != ".json":
        parser.error("choose nonnegative gains and a JSON report")
    artifact = json.loads(args.coefficients.read_text(encoding="utf-8"))
    if any(number not in TRAIN_PACKETS for number in (*artifact["fit_packets"], *artifact["probe_packets"])):
        parser.error("forecast model metadata includes packets outside the training split")
    coefficients = np.asarray(artifact["autoregressive_coefficients"])[:, AXES]
    if coefficients.shape != (LAGS, 2) or artifact["window"] != WINDOW:
        parser.error("forecast model is incompatible with the history format")
    tasks = [(path, args, coefficients, gain) for gain in args.gains for path in paths]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        output = list(pool.map(run, tasks))
    variants = {}
    for j, gain in enumerate(args.gains):
        selected, diagnostics = zip(*output[j * len(paths):(j + 1) * len(paths)])
        stats = aggregate_packets(selected)
        print(gain, stats["score"]["total"], stats["id_switches"], stats["split_truth_tracks"],
              stats["merged_truth_points"], flush=True)
        variants[str(gain)] = {"summary": stats, "evidence": sum(diagnostics, Counter()),
                               "by_packet": [{k: v for k, v in s.items() if k not in ("tracks", "_circle_times")}
                                             for s in selected]}
    report = {"packet_ids": args.packets, "variants": variants,
              "sha256": {name: replay.file_sha256(getattr(args, name)) for name in ("teacher", "pair", "coefficients")}}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
