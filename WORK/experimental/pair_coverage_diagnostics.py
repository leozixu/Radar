"""Trace where the paired tracker loses true observations on the training split."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pair_swap_probe as pairs
import teacher_assoc as teacher
from identity_rejoin import load_quality_reference
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, iter_circle_payloads
from tracker_compat import decisions_with_features


FIELDS = ("Azimuth", "Elevation", "Distance", "Amp")
STAGES = ("mature", "tentative", "seed")


class Recorder:
    def __init__(self, predictor):
        self.predictor = predictor
        self.probabilities = None

    def predict(self, x, num_threads):
        result = self.predictor.predict(x, num_threads=num_threads)
        self.probabilities = result
        return result


def diagnostic(spec):
    from lightgbm import Booster

    path, weights, pair_weights = spec
    predictor = Recorder(Booster(model_file=str(weights)))
    corrector = pairs.PairCorrector(Booster(model_file=str(pair_weights)), .85)
    tracker = teacher.Online(predictor, floor=.20, mix=0., coast=7, ambiguity=.30, robust=4.)
    quality = load_quality_reference()
    state, labels, circles, timestamps, emitted = {}, {}, {}, {}, {}
    lineage, by_truth, by_batch, raw_latest, public_latest = {}, defaultdict(list), defaultdict(Counter), {}, {}
    raw_truth = {}
    diagnostics = Counter()
    misses, gaps, switches, startup = Counter(), Counter(), Counter(), Counter()
    late_delay, late_gap_pattern, late_third_reason, late_plan = Counter(), Counter(), Counter(), Counter()
    first_raw_delay = Counter()
    held_oracle = {}
    reference_features = pairs.candidate_features
    features, contenders, held, observed, stages, lookup = None, None, None, None, None, None

    def capture_features(*args):
        nonlocal features
        features = reference_features(*args)
        return features

    def capture_association(tracks, positions, amplitudes, circle, *args):
        nonlocal contenders, held
        contenders = [t.batch_id for t in tracks]
        matched, held = corrector.associate(tracks, positions, amplitudes, circle, *args)
        return matched, held

    def capture_tracker(points, state, observe=None):
        def observe_all(track, position, amplitude, circle):
            pid = lookup.get((*position, amplitude))
            if pid is not None:
                key = id(track)
                prior = lineage.setdefault(key, deque(maxlen=8))
                observed[pid] = (stages.get(key, "unknown"), tuple(prior), track.batch_id)
                prior.append(pid)
            if observe is not None:
                observe(track, position, amplitude, circle)

        return tracker.track(points, state, observe_all)

    with (patch.object(pairs, "candidate_features", capture_features),
          patch.object(teacher, "assign_confirmed", capture_association),
          patch("tracker_compat.baseline_circle", capture_tracker)):
        for circle, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            timestamps.update((p["PntID"], p["Time"]) for p in points)
            lookup = {tuple(point[f] for f in FIELDS): point["PntID"] for point in points}
            diagnostics["duplicate_measurement_keys"] += len(points) - len(lookup)
            previous = state.get("tracks", ([], [], []))
            stages = {id(t): name for name, collection in zip(STAGES, previous) for t in collection}
            prior_active = {t.batch_id: t.circle for t in previous[0]}
            features, contenders, held, observed = None, [], np.empty(0, dtype=int), {}
            predictor.probabilities = None
            raw, quality_features = decisions_with_features(points, state)
            probabilities = quality.quality_model.predict(quality_features, num_threads=1) if raw else ()
            raw_by_point = {item["PntID"]: (item["BatchID"], q) for item, q in zip(raw, probabilities)}
            raw_by_track = {item["BatchID"]: item["PntID"] for item in raw}
            reported = {pid: bid for pid, (bid, q) in raw_by_point.items() if q >= quality.threshold}
            emitted.update(reported)
            for pid, bid in reported.items():
                if labels[pid] > 0:
                    by_batch[bid][labels[pid]] += 1
            held_ids = {points[j]["PntID"] for j in held}

            seeded = set()
            for seed in state.get("tracks", ([], [], []))[2]:
                pid = lookup.get((*seed.first_position, seed.mean_amplitude))
                if pid is not None:
                    lineage[id(seed)] = deque([pid], maxlen=8)
                    seeded.add(pid)
            still_live = {id(t) for collection in state.get("tracks", ([], [], [])) for t in collection}
            lineage = {key: value for key, value in lineage.items() if key in still_live}
            candidates = {}
            if features is not None and predictor.probabilities is not None:
                rows, cols, _ = features
                candidates = {(contenders[r], points[c]["PntID"]): float(q)
                              for r, c, q in zip(rows, cols, predictor.probabilities)}

            def incumbent_reason(old_batch, pid, truth):
                if old_batch not in prior_active or circle - prior_active[old_batch] > tracker.coast:
                    return "expired"
                q = candidates.get((old_batch, pid))
                if q is None:
                    return "outside_gate"
                if q < tracker.floor:
                    return "below_link_floor"
                rival = raw_by_track.get(old_batch)
                if rival is None:
                    return "available_but_unassigned"
                rival_truth = labels[rival]
                if rival_truth <= 0:
                    return "incumbent_assigned_clutter" if rival_truth == 0 else "incumbent_assigned_unknown"
                return "incumbent_assigned_same_truth" if rival_truth == truth else "incumbent_assigned_other_truth"

            for point in points:
                pid = point["PntID"]
                truth = labels[pid]
                if truth <= 0:
                    continue
                stage, prior, batch = observed.get(pid, (None, (), None))
                if pid in reported:
                    result = "reported"
                elif pid in raw_by_point:
                    result = "quality_rejected"
                elif pid in held_ids:
                    result = "held_for_ambiguity"
                elif stage == "tentative":
                    result = "tentative_failed_confirmation"
                elif stage == "seed":
                    result = "seed_advanced_to_tentative"
                elif stage == "mature":
                    result = "mature_observed_but_no_raw_output"
                elif pid in seeded:
                    result = "idle_became_seed"
                else:
                    result = "idle_no_seed"
                by_truth[truth].append((circle, pid, result, stage, prior, batch))
                previous_public = public_latest.get(truth)
                previous_raw = raw_latest.get(truth)
                if pid not in reported:
                    misses[result] += 1
                    if previous_public is None:
                        gaps[f"startup/{result}"] += 1
                    else:
                        cause = incumbent_reason(previous_public[1], pid, truth)
                        gaps[f"after_first_report/{result}/{cause}"] += 1
                        if result == "held_for_ambiguity" and cause == "available_but_unassigned":
                            held_oracle[pid] = previous_public[1]
                if pid in reported:
                    bid = reported[pid]
                    if previous_public is not None and previous_public[1] != bid:
                        gap = circle - previous_public[0]
                        cause = incumbent_reason(previous_public[1], pid, truth)
                        switches[f"{'gap' if gap > 1 else 'adjacent'}/{cause}"] += 1
                        diagnostics["switch_dest_new"] += bid not in prior_active
                        diagnostics["switch_gap_gt_coast"] += gap > tracker.coast
                        diagnostics["switch_previous_raw_id_same_as_public_target"] += (
                            previous_raw is not None and previous_raw[1] == previous_public[1])
                    public_latest[truth] = (circle, bid)
                if pid in raw_by_point:
                    raw_latest[truth] = (circle, raw_by_point[pid][0])
                    raw_truth[pid] = raw_by_point[pid][0]

    first_output_batch = defaultdict(lambda: float("inf"))
    for pid, bid in emitted.items():
        first_output_batch[bid] = min(first_output_batch[bid], timestamps[pid])
    prior_to_first, after_first = set(), set()
    for truth, sequence in by_truth.items():
        sequence.sort(key=lambda row: (row[0], row[1]))
        if len(sequence) < 3:
            startup["short_truth_track"] += 1
            continue
        pred = min((j for j, row in enumerate(sequence) if row[1] in emitted),
                   key=lambda j: timestamps[sequence[j][1]], default=None)
        raw_pred = min((j for j, row in enumerate(sequence) if row[1] in raw_truth),
                       key=lambda j: timestamps[sequence[j][1]], default=None)
        first_index = next((j for j, row in enumerate(sequence) if row[1] in emitted), len(sequence))
        prior_to_first.update(row[1] for row in sequence[2:first_index])
        after_first.update(row[1] for row in sequence[first_index:] if row[1] not in emitted)
        ref_circle, ref_pid, third_reason, third_stage, third_prior, _ = sequence[2]
        ref_time = timestamps[ref_pid]
        if raw_pred is not None:
            first_raw_delay[str(max(0, sequence[raw_pred][0] - ref_circle))] += 1
        if pred is None:
            startup["never_emitted"] += 1
        elif timestamps[sequence[pred][1]] <= ref_time:
            startup["on_time_or_early"] += 1
        else:
            startup["late_true_hit"] += 1
            late_delay[str(sequence[pred][0] - ref_circle)] += 1
        if pred is None or timestamps[sequence[pred][1]] > ref_time:
            late_third_reason[third_reason] += 1
            gaps01 = sequence[1][0] - sequence[0][0]
            gaps12 = ref_circle - sequence[1][0]
            late_gap_pattern[f"{gaps01}+{gaps12}"] += 1
            first_pid, second_pid = sequence[0][1], sequence[1][1]
            if raw_pred is not None and timestamps[sequence[raw_pred][1]] <= ref_time:
                late_plan["quality_blocked_timely_raw_hit"] += 1
            elif gaps01 != 1 or gaps12 != 1:
                late_plan["truth_sample_gap_breaks_three_consecutive_scans"] += 1
            elif sequence[0][2] != "idle_became_seed":
                late_plan[f"first_truth_unavailable_as_seed/{sequence[0][2]}"] += 1
            elif first_pid not in sequence[1][4][-1:] or sequence[1][3] != "seed":
                late_plan[f"first_seed_not_matched_back_on_second/{sequence[1][2]}"] += 1
            elif (first_pid, second_pid) != third_prior[-2:] or third_stage != "tentative":
                late_plan[f"true_tentative_not_matched_on_third/{third_reason}"] += 1
            else:
                late_plan[f"correct_three_hit_chain_not_reported/{third_reason}"] += 1
        if pred is not None:
            associated_batches = {emitted[row[1]] for row in sequence if row[1] in emitted}
            if min(first_output_batch[bid] for bid in associated_batches) < timestamps[sequence[pred][1]]:
                startup["proxy_batch_time_precedes_first_truth_hit"] += 1

    score = evaluate_packet(labels, emitted, circles, timestamps)
    dominant = {bid: counts.most_common(1)[0][0] for bid, counts in by_batch.items()}
    diagnostics["reported_true_pts_on_other_truths_dominant_batch"] = sum(
        labels[pid] > 0 and labels[pid] != dominant[bid] for pid, bid in emitted.items())
    diagnostics["pair_swaps"] = corrector.swaps
    assert score["missed_truth_points"] == sum(misses.values()), path.name
    assert score["id_switches"] == sum(switches.values()), path.name
    ideal_offset = max(emitted.values(), default=0) + 1
    ideal_identity = {pid: ideal_offset + labels[pid] if labels[pid] > 0 else bid for pid, bid in emitted.items()}
    startup_add = {pid: ideal_offset + labels[pid] for pid in prior_to_first}
    mature_add = {pid: ideal_offset + labels[pid] for pid in after_first}
    counterfactual = {
        "oracle_reinstate_held_points_under_previous_public_id": emitted | held_oracle,
        "oracle_identity_only": ideal_identity,
        "oracle_identity_plus_earliest_start": ideal_identity | startup_add,
        "oracle_identity_plus_post_start_misses": ideal_identity | mature_add,
        "oracle_identity_plus_earliest_start_and_post_start_misses": ideal_identity | startup_add | mature_add,
    }
    ceilings = {name: evaluate_packet(labels, outputs, circles, timestamps) for name, outputs in counterfactual.items()}
    return score, {"causes_of_missed_points": misses, "incumbent_causes": gaps, "switches": switches,
                   "startup": startup, "startup_delay_scans": late_delay, "startup_first_raw_delay_scans": first_raw_delay,
                   "late_third_point": late_third_reason, "late_truth_gap_pattern": late_gap_pattern,
                   "late_chain_failure": late_plan, "diagnostics": diagnostics}, ceilings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.suffix.lower() != ".json":
        parser.error("--report must be a JSON file")
    files = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    jobs = [(path, args.teacher, args.pair) for path in files]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(diagnostic, jobs))
    statistics, contributions, counterfactuals = zip(*results)
    totals = defaultdict(Counter)
    for _, detail, _ in results:
        for key, subsection in detail.items():
            totals[key].update(subsection)
    report = {
        "packet_ids": args.packets,
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        "pair_sha256": hashlib.sha256(args.pair.read_bytes()).hexdigest(),
        "summary": aggregate_packets(statistics), "diagnostics": totals,
        "oracle_counterfactuals": {variant: aggregate_packets(case[variant] for case in counterfactuals)
                                   for variant in counterfactuals[0]},
        "by_packet": dict(zip(args.packets, contributions)),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
