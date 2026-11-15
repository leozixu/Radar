"""Recover registered, still-live pairs after larger measured motion."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from lightgbm import Booster
import numpy as np

from paired_returns import ASSETS, HERE, MAX_GAP, PairReturns, pair_offset
from causal_birth_paths import POINT, POSITION, SCAN, BirthOnline
from baseline_cv import residual
from emission_identity import FEATURES, RAW_RESCUE, EmissionFeatures, accepted_emissions, current_raw_probabilities
from identity_rejoin import load_quality_reference
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from raw_timing_detection import RawReturnGate
import replay_validation as replay
import tracker_compat


ORDINARY_WIDTH = np.array((.038, .140, .020))
BROAD_WIDTH = np.array((.105, .280, .036))
FORECAST_WIDTH = np.array((.058, .160, .021))
ROLE_CHANGE_WIDTH = np.array((.012, .004, .0035))
AMPLITUDE_ERROR = .0000026
PROBABILITY = .97
RAW_MODEL = HERE / "models/raw_timing_detection.txt"
PAIR_MODEL = HERE / "models/paired_returns.txt"
EMISSION_MODELS = {name: HERE / "models" / f"emission_{name}.txt" for name in ("detection", "ownership")}
PROFILE = {"association_floor": .035, "raw_floor": .025, "fallback_floor": .96, "emission": "rescue25_raw80"}


def compatible_recoveries(frame, tracks, raw, circle, prior_pairs):
    """Return only exclusive wide-motion links that preserve the original frame."""
    if not frame.strict_count or not prior_pairs:
        return []
    taken_pairs = {id(prior) for prior, _ in frame.decisions}
    taken_tracks = {row for row, _, _ in frame.locked}
    taken_points = {point for _, index in frame.decisions for point in frame.columns[index]}
    taken_points.update(point for _, point, _ in frame.locked)
    active = {track.batch_id: index for index, track in enumerate(tracks) if index not in taken_tracks}
    registered = [prior for prior in prior_pairs
                  if prior.confirmed and id(prior) not in taken_pairs
                  and 0 < circle - prior.observations[0, SCAN] <= MAX_GAP
                  and prior.owners[0] != prior.owners[1] and all(owner in active for owner in prior.owners)]
    if not registered:
        return []
    columns = np.asarray([index for index, pair in enumerate(frame.columns[:frame.strict_count])
                          if not any(point in taken_points for point in pair)], dtype=int)
    if not len(columns):
        return []
    previous = np.stack([prior.observations for prior in registered])
    current = raw[frame.columns[columns]]
    old_amp, new_amp = pair_offset(previous), pair_offset(current)
    old_role = residual(previous[:, 1, POSITION], previous[:, 0, POSITION])
    new_role = residual(current[:, 1, POSITION], current[:, 0, POSITION])
    rows, cols = np.nonzero(
        (np.abs(old_amp[:, None] - new_amp[None, :]) <= AMPLITUDE_ERROR)
        & (old_amp[:, None] < 0.) & (new_amp[None, :] < 0.)
        & (old_role[:, None, 1] >= -1e-7) & (new_role[None, :, 1] >= -1e-7)
        & (np.abs(new_role[None, :, 2]) <= .0038))
    if not len(rows):
        return []
    elapsed = circle - previous[rows, 0, SCAN]
    observed = current[cols, :, POSITION]
    motion = np.abs(residual(observed, previous[rows, :, POSITION]).mean(axis=1))
    width = (1. + .60 * (elapsed - 1.))[:, None]
    shape = np.abs(residual(new_role[cols], old_role[rows]))
    viable = (~(motion <= ORDINARY_WIDTH * width).all(axis=1) & (motion <= BROAD_WIDTH * width).all(axis=1)
              & (shape <= ROLE_CHANGE_WIDTH).all(axis=1))
    rows, cols, observed = rows[viable], cols[viable], observed[viable]
    if not len(rows):
        return []
    choices = []
    for index, col, measured in zip(rows, cols, observed):
        provenance = registered[index]
        owners = tuple(active[owner] for owner in provenance.owners)
        tracks_for_pair = [tracks[owner] for owner in owners]
        lag = max(circle - track.circle for track in tracks_for_pair)
        prediction = np.stack([track.position + (circle - track.circle) * track.velocity
                               for track in tracks_for_pair])
        forecast_error = np.abs(residual(measured, prediction).mean(axis=0))
        if (forecast_error <= FORECAST_WIDTH * (1. + .35 * max(0, lag - 1))).all():
            choices.append((provenance, int(columns[col]), owners))
    row_counts = Counter(owner for _, _, owners in choices for owner in owners)
    point_counts = Counter(point for _, index, _ in choices for point in frame.columns[index])
    return [(prior, index, owners) for prior, index, owners in choices
            if all(row_counts[owner] == 1 for owner in owners)
            and all(point_counts[point] == 1 for point in frame.columns[index])]


class RegisteredPairRecovery(PairReturns):
    """Handles old registrations before an outer class adds unowned starts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovered = []
        self.extra_times = []

    def begin(self, tracks, raw, times, circle, state):
        frame = super().begin(tracks, raw, times, circle, state)
        started = perf_counter()
        for prior, index, owners in compatible_recoveries(frame, tracks, raw, circle, state.get("previous_pairs", ())):
            points = frame.columns[index]
            frame.decisions.append((prior, index))
            frame.locked.extend((owner, int(point), PROBABILITY) for owner, point in zip(owners, points))
            self.recovered.append(tuple(map(int, prior.observations[:, POINT])) + tuple(map(int, raw[points, POINT])))
        self.extra_times.append((perf_counter() - started) * 1000)
        return frame


def pairing_type(variant):
    if variant in ("fast", "both"):
        from fast_pair_center import FastPairCenter

        if variant == "both":
            class BothRecoveries(FastPairCenter, RegisteredPairRecovery):
                pass

            return BothRecoveries
        return FastPairCenter
    return RegisteredPairRecovery if variant == "recovery" else PairReturns


def replay_stage(spec):
    path, variant = spec
    weights = {name: Booster(model_file=str(path)) for name, path in ASSETS.items()}
    output_weights = {name: Booster(model_file=str(path)) for name, path in EMISSION_MODELS.items()}
    gate = RawReturnGate(Booster(model_file=str(RAW_MODEL)), PROFILE["raw_floor"])
    pairing = pairing_type(variant)(Booster(model_file=str(PAIR_MODEL)), .985, .95, wide=True,
                                    fallback_floor=PROFILE["fallback_floor"])
    tracker = BirthOnline(*weights.values(), .86, "supplement", paired_birth=True, joint=True, two_hit_pairs=True,
                          pairing=pairing, raw_gate=gate, association_floor=PROFILE["association_floor"])
    history, reference = EmissionFeatures(), load_quality_reference()
    detection_floor, raw_floor = RAW_RESCUE[PROFILE["emission"]]
    labels, circles, timestamps, emitted, state = {}, {}, {}, {}, {}
    circle_times = []
    with patch.object(tracker_compat, "baseline_circle", tracker.track):
        for circle, payload in replay.iter_circle_payloads(path, labels, circles, TUNING_PACKETS):
            started = perf_counter()
            points = [dict(zip(replay.POINT_KEYS, row)) for row in replay.POINT.iter_unpack(payload)]
            proposals, quality = tracker_compat.decisions_with_features(points, state)
            probabilities = reference.quality_model.predict(quality, num_threads=1) if proposals else np.empty(0)
            features = history.build(points, proposals, quality, probabilities, state, weights["clock"])
            selected = []
            if proposals:
                detection = output_weights["detection"].predict(features, num_threads=1)
                ownership = output_weights["ownership"].predict(features, num_threads=1)
                young = features[:, FEATURES.index("trusted_proposals")] < 3
                raw_probability = current_raw_probabilities(points, proposals, state)
                keep = accepted_emissions(probabilities >= reference.threshold, detection, ownership, young,
                                          (0., detection_floor, 0.), raw_probability, raw_floor)
                selected = [row for row, accepted in zip(proposals, keep) if accepted]
            source = {(row["Time"], row["CircleID"], row["PntID"]): (row["Azimuth"], row["Elevation"], row["Distance"])
                      for row in points}
            frame = replay.encode_tracks(selected, source)
            circle_times.append((circle, (perf_counter() - started) * 1000))
            timestamps.update((row["PntID"], row["Time"]) for row in points)
            emitted.update((row[-1], row[-3]) for row in replay.TRACK.iter_unpack(memoryview(frame)[5:-4]))
    stats = Counter()
    for a, b, c, d in getattr(pairing, "recovered", ()):
        stats["recovered_pairs"] += 1
        stats["same_truth_both_arms"] += labels[a] == labels[c] > 0 and labels[b] == labels[d] > 0
        stats["old_or_new_unscored_arm"] += min(labels[a], labels[b], labels[c], labels[d]) < 0
        stats["emitted_points_from_recovery"] += (c in emitted) + (d in emitted)
    for a, b, c, d in getattr(pairing, "started", ()):
        stats["fast_starts"] += 1
        stats["fast_starts_same_truth"] += labels[a] == labels[c] > 0 and labels[b] == labels[d] > 0
    result = evaluate_packet(labels, emitted, circles, timestamps)
    print(path.stem, variant, round(result["score"]["total"], 5), result["id_switches"],
          result["split_truth_tracks"], dict(stats), flush=True)
    return result, stats, getattr(pairing, "extra_times", []), circle_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--with-fast-start", action="store_true")
    parser.add_argument("--variant", choices=("control", "recovery", "fast", "both"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    variants = ((args.variant,) if args.variant else
                ("both", "control", "recovery", "fast") if args.with_fast_start else ("control", "recovery"))
    sources = ("registered_pair_recovery.py", "paired_returns.py", "causal_birth_paths.py", "emission_identity.py")
    if "both" in variants or "fast" in variants:
        sources += ("fast_pair_center.py",)
    report = {"packets": args.packets, "score_version": SCORE_VERSION, "profile": PROFILE,
              "timing": {"parallel_packet_workers": args.jobs, "prediction_threads_per_worker": 1,
                         "clock": "wall time from point decoding through TRACK encoding, excluding input I/O and scoring"},
              "source_sha256": {name: replay.file_sha256(HERE / name) for name in sources},
              "model_sha256": {name: replay.file_sha256(path) for name, path in
                               (ASSETS | EMISSION_MODELS | {"raw_detection": RAW_MODEL, "pair_returns": PAIR_MODEL,
                                "compat_quality": HERE.parent / "participant_demo/models/quality.txt"}).items()},
              "variants": {}}
    specs = [(path, variant) for variant in variants for path in paths]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        collected = list(pool.map(replay_stage, specs))
    for index, variant in enumerate(variants):
        runs = collected[index * len(paths):(index + 1) * len(paths)]
        results = [result for result, _, _, _ in runs]
        timings = [value for _, _, block, _ in runs for value in block]
        circles = [(path.stem, circle, duration) for path, (_, _, _, measures) in zip(paths, runs)
                   for circle, duration in measures]
        slowest = max(circles, key=lambda event: event[2])
        report["variants"][variant] = {
            "summary": aggregate_packets(results),
            "recovery": sum((stats for _, stats, _, _ in runs), Counter()),
            "extra_p99_ms": replay._p99(timings) if timings else None,
            "extra_max_ms": max(timings) if timings else None,
            "circle_p99_ms": replay._p99([event[2] for event in circles]),
            "slowest_circle": {"packet": slowest[0], "circle": slowest[1], "ms": slowest[2]},
            "by_packet": {path.stem: {k: v for k, v in result.items() if k != "tracks"}
                          for path, result in zip(paths, results)},
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
