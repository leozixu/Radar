"""Diagnose identity changes on the training split. Labels are used only offline."""

import argparse
from collections import Counter, defaultdict, deque
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import DEFAULT_CONFIG, POINT, POINT_KEYS, engine_metadata, iter_circle_payloads

from algorithm import load_runtime
from baseline_cv import FOLLOW_GATE, MAX_MISSED_CIRCLES, residual
from tracker_compat import CompatibilityRuntime, decisions_with_features


def _classify_old_track(track, circle, point, old_batch, raw_points, labels):
    if track is None or circle - track[0] > MAX_MISSED_CIRCLES + 1:
        return "expired"
    last_circle, position, velocity, mean_amp = track
    elapsed = circle - last_circle
    target = np.array([point["Azimuth"], point["Elevation"], point["Distance"]])
    error = residual(target, position + velocity * elapsed) / (FOLLOW_GATE * (1 + 0.35 * (elapsed - 1)))
    distance = float(error @ error)
    if distance > 1:
        return "outside_gate"
    amp = point["Amp"]
    if distance + 0.45 * max(0.35 - amp, 0) / 0.35 + 0.22 * abs(mean_amp - amp) >= 1:
        return "outscored_by_miss_option"
    raw_point = raw_points.get(old_batch)
    if raw_point is None:
        return "available_but_unmatched"
    label = labels[raw_point]
    if label == 0:
        return "preferred_clutter"
    if label < 0:
        return "preferred_unknown"
    return "preferred_same_truth" if label == labels[point["PntID"]] else "preferred_other_truth"


def _relative_pair_order(prediction, positions, axes):
    errors = residual(positions, prediction) / axes
    true_error, other_error = np.sum(errors**2, axis=1)
    return "true_closer" if true_error < other_error else "other_closer_or_tied"


def _pair_diagnostic(history, old_track, circle, correct, wrong, labels):
    # Select a known-correct prior window only for this diagnostic. It estimates
    # whether better motion evidence can rank a current crossing even before an
    # association error has contaminated the track state.
    if len(history) < 4 or any(labels[point["PntID"]] != labels[correct["PntID"]] for point in list(history)[-4:]):
        return {}
    fields = ("Azimuth", "Elevation", "Distance")
    current = np.array([[point[field] for field in fields] for point in (correct, wrong)])
    last_circle, position, velocity, _ = old_track
    predictions = {"cv_kf": position + velocity * (circle - last_circle)}
    for size in (4, 8, 12):
        if len(history) < size:
            continue
        previous = list(history)[-size:]
        positions = np.array([[point[field] for field in fields] for point in previous])
        anchor = positions[-1]
        design = np.column_stack((np.ones(size), [point["CircleID"] - circle for point in previous]))
        coefficients = np.linalg.lstsq(design, residual(positions, anchor), rcond=None)[0]
        predictions[f"linear_{size}"] = anchor + coefficients[0]
    return {method: _relative_pair_order(prediction, current, FOLLOW_GATE) for method, prediction in predictions.items()}


def _switches_and_oracles(path, runtime):
    state, labels, circles, timestamps, emitted, last_output, last_raw_label = {}, {}, {}, {}, {}, {}, {}
    raw_history = {}
    events = []
    for circle, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
        points = [dict(zip(POINT_KEYS, record)) for record in POINT.iter_unpack(payload)]
        by_pid = {point["PntID"]: point for point in points}
        timestamps.update((pid, point["Time"]) for pid, point in by_pid.items())
        old_tracks = {
            track.batch_id: (track.circle, track.position.copy(), track.velocity.copy(), track.mean_amplitude)
            for track in state.get("tracks", ((), (), ()))[0]
        }
        raw, features = decisions_with_features(points, state)
        if not raw:
            continue
        raw_points = {result["BatchID"]: result["PntID"] for result in raw}
        probabilities = runtime.quality_model.predict(features, num_threads=1)
        previous_outputs = last_output.copy()
        for result, probability in sorted(zip(raw, probabilities), key=lambda pair: pair[0]["PntID"]):
            if not probability >= runtime.threshold:
                continue
            pid, batch = result["PntID"], result["BatchID"]
            emitted[pid] = batch
            truth = labels[pid]
            if truth <= 0:
                continue
            previous = last_output.get(truth)
            if previous is not None and previous[1] != batch:
                last_seen, old_batch = previous
                cause = _classify_old_track(old_tracks.get(old_batch), circle, by_pid[pid], old_batch, raw_points, labels)
                competing_pid = raw_points.get(old_batch)
                competing_truth = labels.get(competing_pid, 0)
                possible_swap = previous_outputs.get(competing_truth, (None, None))[1] == batch
                events.append({
                    "pid": pid, "truth": truth, "source": old_batch, "destination": batch,
                    "gap": circle - last_seen,
                    "cause": cause,
                    "old_track_last_raw_label": last_raw_label.get(old_batch),
                    "destination_newly_confirmed": batch not in old_tracks,
                    "reciprocal_swap": cause == "preferred_other_truth" and possible_swap,
                    "pair_position_order": _pair_diagnostic(
                        raw_history.get(old_batch, ()), old_tracks[old_batch], circle,
                        by_pid[pid], by_pid[competing_pid], labels,
                    ) if cause == "preferred_other_truth" else {},
                })
            last_output[truth] = (circle, batch)
        last_raw_label.update((batch, labels[pid]) for batch, pid in raw_points.items())
        for batch, pid in raw_points.items():
            raw_history.setdefault(batch, deque(maxlen=12)).append(by_pid[pid])

    ownership, truth_sequences = defaultdict(Counter), defaultdict(list)
    for pid, batch in emitted.items():
        truth = labels[pid]
        if truth > 0:
            ownership[batch][truth] += 1
            truth_sequences[truth].append((circles[pid], pid, batch))
    owner = {batch: counts.most_common(1)[0][0] for batch, counts in ownership.items()}
    events_by_pid = {event["pid"]: event for event in events}
    for sequence in truth_sequences.values():
        runs = [list(run) for _, run in groupby(sorted(sequence), key=lambda item: item[2])]
        for i in range(1, len(runs)):
            before, incoming = runs[i - 1], runs[i]
            event = events_by_pid[incoming[0][1]]
            event["incoming_run_hits"] = len(incoming)
            event["returns_within_5_scans"] = (
                i + 1 < len(runs) and runs[i + 1][0][2] == before[0][2]
                and runs[i + 1][0][0] - incoming[0][0] <= 5
            )

    def with_aliases(aliases):
        return {pid: aliases.get(batch, batch) for pid, batch in emitted.items()}

    first_issued = max(emitted.values(), default=0) + 1
    pure_aliases = {batch: first_issued + truth for batch, truth in owner.items() if len(ownership[batch]) == 1}
    majority_aliases = {batch: first_issued + truth for batch, truth in owner.items()}
    stages = {
        "replay": emitted,
        "oracle_join_pure_fragments": with_aliases(pure_aliases),
        "oracle_join_majority_fragments": with_aliases(majority_aliases),
        "oracle_drop_cross_identity_outputs": {
            pid: batch for pid, batch in emitted.items() if labels[pid] <= 0 or owner[batch] == labels[pid]
        },
        "oracle_per_point_identity": {
            pid: first_issued + labels[pid] if labels[pid] > 0 else batch for pid, batch in emitted.items()
        },
    }
    metrics = {stage: evaluate_packet(labels, outputs, circles, timestamps) for stage, outputs in stages.items()}
    assert metrics["replay"]["id_switches"] == len(events), path
    return metrics, {
        "switches": len(events),
        "switch_causes": Counter(event["cause"] for event in events),
        "adjacent_switch_causes": Counter(event["cause"] for event in events if event["gap"] == 1),
        "gap_switch_causes": Counter(event["cause"] for event in events if event["gap"] > 1),
        "after_gap": sum(event["gap"] > 1 for event in events),
        "after_gap_more_than_4": sum(event["gap"] > 4 for event in events),
        "newly_confirmed_destination": sum(event["destination_newly_confirmed"] for event in events),
        "one_hit_incoming_run": sum(event["incoming_run_hits"] == 1 for event in events),
        "return_within_5_scans": sum(event["returns_within_5_scans"] for event in events),
        "incoming_id_mainly_other_truth": sum(owner[event["destination"]] != event["truth"] for event in events),
        "reciprocal_swap_observations": sum(event["reciprocal_swap"] for event in events),
        "pair_position_ranking": Counter(
            f"{method}/{rank}" for event in events for method, rank in event["pair_position_order"].items()
        ),
        "old_id_had_other_or_false_internal_assignment": sum(
            event["old_track_last_raw_label"] is not None and event["old_track_last_raw_label"] != event["truth"]
            for event in events
        ),
    }


def compact(stats):
    return {key: stats[key] for key in (
        "score", "idf1", "id_switches", "split_truth_tracks", "merged_output_tracks", "merged_truth_point_ratio",
        "matched_points", "point_coverage", "mean_track_coverage", "false_alarm_ratio",
    )}


def analyze(paths, config):
    runtime = load_runtime(config)
    if not isinstance(runtime, CompatibilityRuntime):
        raise ValueError("Only the compat backend is supported by this diagnostic.")
    by_stage = defaultdict(list)
    combined = {}
    by_packet = []
    for path in paths:
        stages, diagnostics = _switches_and_oracles(path, runtime)
        for key, value in diagnostics.items():
            combined[key] = combined.get(key, Counter() if isinstance(value, Counter) else 0) + value
        for stage, stats in stages.items():
            by_stage[stage].append(stats)
        by_packet.append({"file": path.name, "diagnostics": diagnostics,
                          "stages": {stage: compact(stats) for stage, stats in stages.items()}})
        print(f"{path.name}: {diagnostics['switches']} switches, score {stages['replay']['score']['total']:.4f}", flush=True)
    return {"diagnostics": combined, "stages": {stage: compact(aggregate_packets(stats)) for stage, stats in by_stage.items()},
            "packets": by_packet}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", default=[1, 6, 14, 35, 70])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    except ValueError as error:
        parser.error(str(error))
    result = analyze(paths, args.config)
    result["metadata"] = engine_metadata("candidate", args.config) | {"packet_ids": args.packets}
    print(json.dumps({key: result[key] for key in ("diagnostics", "stages")}, indent=2))
    if args.report:
        with args.report.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
