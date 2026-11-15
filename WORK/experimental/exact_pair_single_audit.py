"""Trace startup and post-birth losses in paired-return replay on training packets."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lightgbm import Booster

from clock_association import wrap, PERIOD_MS
from baseline_cv import residual
from paired_returns import ASSETS, HERE, MAX_GAP, PAIR_WIDTH, BirthOnline, PairReturns, link_features
from causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, SCAN, PathPool
from identity_rejoin import load_quality_reference
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import POINT as WIRE_POINT, POINT_KEYS, file_sha256, iter_circle_payloads
import tracker_compat


PAIR_MODEL = HERE / "models/paired_returns.txt"
BIRTH_FLOOR = .86


class BirthTrace(PathPool):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def propose(self, points, circle):
        self.candidates = points
        self.proposals = super().propose(points, circle)
        self.scores = None
        return self.proposals

    def predict(self, features, num_threads):
        self.scores = self.model.predict(features, num_threads=num_threads)
        return self.scores

    def true_paths(self, labels):
        if self.proposals is None:
            return {}
        first, middle, cols, _ = self.proposals
        paths = {}
        for a, b, j, q in zip(first[:, POINT], middle[:, POINT], cols, self.scores):
            c = int(self.candidates[j, POINT])
            a, b = int(a), int(b)
            target = labels[c]
            if target > 0 and labels[a] == labels[b] == target:
                paths[(a, b, c)] = float(q)
        return paths


class PairTrace(PairReturns):
    def begin(self, tracks, raw, times, circle, state):
        self.observations = raw
        self.previous = [prior for prior in state.get("previous_pairs", ())
                         if circle - prior.observations[0, SCAN] <= (MAX_GAP if prior.confirmed else 1)]
        self.active = {track.batch_id for track in tracks}
        self.frame = super().begin(tracks, raw, times, circle, state)
        return self.frame

    def true_pair_status(self, labels):
        observations = self.observations[self.frame.columns]
        ordered_labels = [tuple(labels[int(pid)] for pid in points[:, POINT]) for points in observations]
        result = {}
        if not self.previous or not len(observations):
            return {int(pid): "measured_without_eligible_pair_history"
                    for points, pair_labels in zip(observations, ordered_labels) if genuine_pair(pair_labels)
                    for pid in points[:, POINT]}
        previous = np.stack([history.observations for history in self.previous])
        past_labels = [tuple(labels[int(pid)] for pid in points[:, POINT]) for points in previous]
        historical = defaultdict(list)
        for row, pair_labels in enumerate(past_labels):
            if not genuine_pair(pair_labels):
                continue
            historical[pair_labels].append(row)
        rows, cols, features = link_features(previous, observations)
        scores = self.model.predict(features, num_threads=1) if len(rows) else ()
        candidates = defaultdict(list)
        for row, col, score in zip(rows, cols, scores):
            if past_labels[row] == ordered_labels[col]:
                candidates[col].append((self.previous[row].confirmed, float(score)))
        decisions = dict((col, prior) for prior, col in self.frame.decisions)
        for col, pair_labels in enumerate(ordered_labels):
            if not genuine_pair(pair_labels):
                continue
            history_rows = historical[pair_labels]
            if col in decisions:
                prior = decisions[col]
                linked = tuple(labels[int(pid)] for pid in prior.observations[:, POINT])
                owner_count = sum(owner in self.active for owner in prior.owners)
                status = f"selected_with_{owner_count}_active_owners"
                if linked != pair_labels:
                    status = "selected_from_other_truth_history"
            elif not history_rows:
                status = "measured_without_eligible_pair_history"
            elif not candidates[col]:
                status = "true_history_outside_signature_or_motion_gate"
            elif not any(score >= (self.link_floor if confirmed else self.start_floor)
                         for confirmed, score in candidates[col]):
                status = "true_candidate_below_pair_model_floor"
            else:
                status = "true_candidate_lost_pair_allocation"
            for pid in observations[col, :, POINT]:
                result[int(pid)] = status
        return result


def genuine_pair(labels):
    return labels[0] > 0 and labels[1] > 0 and labels[0] != labels[1]


class AssociationTrace:
    def __init__(self, matcher, pairing):
        self.matcher = matcher
        self.pairing = pairing

    def associate(self, tracks, positions, amplitudes, *args):
        matches, held = self.matcher.associate(tracks, positions, amplitudes, *args)
        frame = self.pairing.frame
        raw = self.pairing.observations
        locked = np.zeros(len(raw), dtype=bool)
        for _, col, _ in frame.locked:
            locked[col] = True
        self.held = set(map(int, raw[~locked][held, POINT]))
        return matches, held

    def observe(self, *args):
        self.matcher.observe(*args)


def category(ids, status):
    if any(status[pid][0] == "quality_rejected" for pid in ids):
        return "some_measurement_quality_rejected"
    if any(status[pid][0] == "held" for pid in ids):
        return "some_measurement_held"
    if any(not status[pid][1] for pid in ids):
        return "some_measurement_not_birth_eligible"
    if any(not status[pid][2] for pid in ids):
        return "some_measurement_unavailable_to_path_pool"
    return "all_available_but_no_true_path"


def diagnostics(path):
    weights = {key: Booster(model_file=str(asset)) for key, asset in ASSETS.items()}
    pairing = PairTrace(Booster(model_file=str(PAIR_MODEL)), .985, .95, early=True)
    tracker = BirthOnline(*weights.values(), BIRTH_FLOOR, "supplement", paired_birth=True, joint=True,
                          two_hit_pairs=True, pairing=pairing)
    trace = BirthTrace(weights["birth"])
    with patch.object(tracker, "birth", trace), patch.object(tracker, "matcher", AssociationTrace(tracker.matcher, pairing)):
        observed = replay(path, tracker, trace, pairing)
    labels, circles, timestamps, emitted, status, mates, path_scores, scans, pair_status = observed
    truth_points = defaultdict(list)
    truth_by_scan = defaultdict(list)
    output_labels = defaultdict(Counter)
    for pid, label in labels.items():
        if label > 0:
            truth_points[label].append(pid)
            truth_by_scan[circles[pid], label].append(pid)
            if pid in emitted:
                output_labels[emitted[pid]][label] += 1

    results = defaultdict(Counter)
    oracles = defaultdict(dict)
    probe = defaultdict(list)
    for label, ids in truth_points.items():
        ids.sort(key=lambda p: (circles[p], p))
        mate, overlap = mates[label].most_common(1)[0] if mates[label] else (None, 0)
        group = "stable_mate" if overlap >= 3 and overlap >= .25 * len(ids) else "singleton_or_intermittent"
        result = results[group]
        result["targets"] += 1
        result["raw_true_points"] += len(ids)
        result["true_points_in_a_measured_pair"] += sum(status[pid][3] for pid in ids)
        first = next((index for index, pid in enumerate(ids) if pid in emitted), len(ids))
        result["ever_emitted"] += first != len(ids)
        if len(ids) >= 3:
            result["targets_with_three_returns"] += 1
            result["on_time"] += first <= 2
            result["late_but_eventually_emitted"] += 2 < first < len(ids)
            result["never_emitted"] += first == len(ids)
        prior, previous_emitted = 0, -1
        for idx, pid in enumerate(ids):
            reason, _, _, _ = status[pid]
            if pid in emitted:
                owner = emitted[pid]
                result["emitted_true_points"] += 1
                if prior and prior != owner:
                    result["switches"] += 1
                    if previous_emitted == idx - 1:
                        result["switches_no_true_miss"] += 1
                    if group == "stable_mate":
                        event = pair_status.get(pid, "no_measured_pair")
                        result[f"pair_switch/{event}"] += 1
                        if previous_emitted == idx - 1:
                            result[f"pair_switch_without_missing/{event}"] += 1
                        if event == "no_measured_pair":
                            circle = circles[pid]
                            reason = measure_pair_drop(pid, truth_by_scan[circle, mate], scans[circle], timestamps)
                            result[f"pair_detection_miss/{reason}"] += 1
                prior = owner
                previous_emitted = idx
                result["emitted_to_other_targets_dominant_id"] += output_labels[owner].most_common(1)[0][0] != label
            else:
                phase = "first_two" if idx < 2 else "startup" if idx < first else "mature"
                result[f"missing/{phase}/{reason}"] += 1
                if phase != "first_two":
                    oracles[group + "/" + phase][pid] = label
        if len(ids) < 3:
            continue

        start_ids = tuple(ids[:3])
        if first > 2:
            result["late_third/" + status[start_ids[-1]][0]] += 1
            gaps = [circles[b] - circles[a] for a, b in zip(start_ids, start_ids[1:])]
            result[f"late_gap/{gaps[0]}+{gaps[1]}"] += 1
            if start_ids in path_scores:
                result["late_chain/pool_passed_floor" if path_scores[start_ids] >= BIRTH_FLOOR
                       else "late_chain/pool_below_floor"] += 1
            elif sum(gaps) > 4 or max(gaps) > 2 or min(gaps) < 1:
                result["late_chain/outside_pool_history"] += 1
            else:
                result["late_chain/" + category(start_ids, status)] += 1
        if group != "stable_mate":
            measure_signatures(ids, scans, circles, timestamps, labels, probe)

    baseline = evaluate_packet(labels, emitted, circles, timestamps)
    assert sum(group["switches"] for group in results.values()) == baseline["id_switches"], path.name
    assert sum(group["emitted_true_points"] for group in results.values()) == baseline["matched_points"], path.name
    offset = max(emitted.values(), default=0) + 1
    ideal_id = {pid: offset + labels[pid] if labels[pid] > 0 else bid for pid, bid in emitted.items()}
    comparisons = {
        "perfect_identity_existing_output": evaluate_packet(labels, ideal_id, circles, timestamps)
    } | {
        "perfect_identity_plus_" + group: evaluate_packet(labels, ideal_id | {pid: offset + label
                                                                             for pid, label in extra.items()},
                                                         circles, timestamps)
        for group, extra in oracles.items()
    }
    return baseline, results, comparisons, probe


def replay(path, tracker, trace, pairing):
    labels, circles, timestamps, emitted, status, path_scores = {}, {}, {}, {}, {}, {}
    scans, pair_status, mates = {}, {}, defaultdict(Counter)
    reference = load_quality_reference()
    state = {"birth_path_pool": trace}
    with patch.object(tracker_compat, "baseline_circle", tracker.track):
        for circle, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in WIRE_POINT.iter_unpack(payload)]
            time = np.fromiter((p["Time"] for p in points), dtype=np.int64)
            timestamps.update((p["PntID"], p["Time"]) for p in points)
            output, features = tracker_compat.decisions_with_features(points, state)
            confidence = reference.quality_model.predict(features, num_threads=1) if output else ()
            raw_outputs = {p["PntID"]: p["BatchID"] for p in output}
            public = {p["PntID"]: p["BatchID"] for p, q in zip(output, confidence) if q >= reference.threshold}
            emitted.update(public)
            raw = pairing.observations
            pair_status.update(pairing.true_pair_status(labels))
            paired = set()
            for a, b in raw[pairing.frame.columns, POINT].astype(np.int64):
                la, lb = labels[a], labels[b]
                if genuine_pair((la, lb)):
                    mates[la][lb] += 1
                    mates[lb][la] += 1
                    paired.update((a, b))
            candidates = set(trace.candidates[:, POINT].astype(np.int64))
            for row in raw:
                pid = int(row[POINT])
                if labels[pid] <= 0:
                    continue
                if pid in public:
                    reason = "public"
                elif pid in raw_outputs:
                    reason = "quality_rejected"
                elif pid in tracker.matcher.held:
                    reason = "held"
                else:
                    reason = "unclaimed"
                status[pid] = reason, row[AMPLITUDE] >= .12 or row[CLOCK] >= .42, pid in candidates, pid in paired
            phase = wrap(raw[:, 0] - time / PERIOD_MS)
            scans[circle] = np.column_stack((raw[:, POSITION], raw[:, AMPLITUDE], raw[:, POINT], phase,
                                            raw[:, CLOCK], time))
            path_scores.update(trace.true_paths(labels))
    return labels, circles, timestamps, emitted, status, mates, path_scores, scans, pair_status


def measure_pair_drop(pid, mates, scan, timestamps):
    if not mates:
        return "companion_absent_from_raw_scan"
    current = scan[scan[:, 4] == pid][0]
    other = scan[np.isin(scan[:, 4], mates)]
    adjacent = np.abs(other[:, 7] - timestamps[pid]) == 1
    if not adjacent.any():
        return "not_neighboring_timestamp_ticks"
    other = other[adjacent]
    if np.count_nonzero(scan[:, 7] == timestamps[pid]) != 1:
        return "a_timestamp_has_multiple_measurements"
    unique = np.asarray([np.count_nonzero(scan[:, 7] == stamp) == 1 for stamp in other[:, 7]])
    if not unique.any():
        return "a_timestamp_has_multiple_measurements"
    other = other[unique]
    close = (np.abs(residual(other[:, :3], current[:3])) <= PAIR_WIDTH).all(axis=1)
    if not close.any():
        return "outside_spatial_pair_width"
    other = other[close]
    difference = np.abs(other[:, 3] - current[3])
    allowed = (difference >= .000004) & (difference <= .032)
    if not allowed.any():
        return "outside_pair_amplitude_difference_bounds"
    other = other[allowed]
    plausible = (np.minimum(other[:, 6], current[6]) >= .20) | (np.minimum(other[:, 3], current[3]) >= .20)
    if not plausible.any():
        return "below_pair_measurement_quality"
    return "pair_limit_or_alternative_partner"


def measure_signatures(ids, scans, circles, timestamps, labels, destination):
    # Compare the real next return with the hardest known rival in the startup spatial gate.
    from causal_birth_paths import LINK_WIDTH, MOTION_SCALE
    source, target = ids[:2]
    a, b = circles[source], circles[target]
    gap = b - a
    if gap not in (1, 2):
        return
    earlier, current = scans[a], scans[b]
    previous = earlier[earlier[:, 4] == source][0]
    shift = residual(current[:, :3], previous[:3])
    scales = LINK_WIDTH * (1. + .45 * (gap - 1))
    distance = np.square(shift / scales).sum(axis=1)
    gate = distance <= 1.
    pids = current[:, 4].astype(np.int64)
    truths = np.asarray([labels[pid] for pid in pids])
    true = pids == target
    if not np.any(gate & true):
        return
    phase_change = np.abs(wrap(current[:, 5] - previous[5]))
    amplitude_change = np.abs(current[:, 3] - previous[3])
    elapsed = np.asarray([timestamps[pid] for pid in pids]) - timestamps[source]
    # Phase aliasing means multiple returns can have low wrapped residual. Also compare unwrapped scan travel.
    scan_change = np.abs(elapsed - 3. * PERIOD_MS * gap - shift[:, 0] * PERIOD_MS)
    metrics = {"geometry": np.sqrt(distance), "amplitude_change": amplitude_change,
               "phase_change": phase_change, "clock_scan_change": scan_change,
               "distance_change": np.abs(shift[:, 2]) / MOTION_SCALE[2]}
    for pool, mask in (("other_truth", gate & (truths > 0) & (truths != labels[source])),
                       ("clutter", gate & (truths == 0))):
        if not np.any(mask):
            continue
        for name, values in metrics.items():
            destination[f"{pool}/{name}"].append((float(values[true][0]), float(values[mask].min())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    if args.jobs == 1:
        results = list(map(diagnostics, paths))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(diagnostics, paths))
    baseline, grouped, alternatives, samples = zip(*results)
    totals = defaultdict(Counter)
    signatures = defaultdict(list)
    for groups, observations in zip(grouped, samples):
        for group, counts in groups.items():
            totals[group].update(counts)
        for metric, rows in observations.items():
            signatures[metric].extend(rows)
    report = {
        "score_version": SCORE_VERSION,
        "packet_ids": args.packets,
        "baseline": aggregate_packets(baseline),
        "by_group": totals,
        "oracle_total_scores": {key: aggregate_packets([row.get(key, row["perfect_identity_existing_output"])
                                                        for row in alternatives])["score"]["total"]
                                for key in sorted(set().union(*alternatives))},
        "startup_signature_candidates": {
            metric: {"target_choices": len(rows), "truth_beats_best_rival_pct": float(100 * np.mean(a < b)),
                     "truth_median": float(np.median(a)), "best_rival_median": float(np.median(b))}
            for metric, rows in signatures.items() for a, b in [np.asarray(rows).T]
        },
        "model_sha256": {key: file_sha256(value) for key, value in
                         (ASSETS | {"paired_returns": PAIR_MODEL,
                                    "quality": HERE.parent / "participant_demo/models/quality.txt"}).items()},
        "source_sha256": {name: file_sha256(HERE / name)
                          for name in ("paired_returns.py", "causal_birth_paths.py", Path(__file__).name)},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
