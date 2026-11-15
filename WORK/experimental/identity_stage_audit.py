"""Attribute public BatchID failures in the joint and early-birth profile."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import json
from pathlib import Path
from unittest.mock import patch

from causal_birth_paths import BirthOnline
from identity_rejoin import load_quality_reference
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, file_sha256, iter_circle_payloads
from tracker_compat import decisions_with_features


ASSETS = {
    "teacher": Path("experimental/teacher_assoc_wide.txt"),
    "pair": Path("experimental/models/pair_swap_wide.txt"),
    "birth": Path("experimental/models/causal_birth_paths.txt"),
    "clock": Path("experimental/models/scan_clock.txt"),
}


def reference_tracker(assets=ASSETS):
    from lightgbm import Booster

    weights = [Booster(model_file=str(assets[key])) for key in ("teacher", "pair", "birth", "clock")]
    return BirthOnline(*weights, floor=.86, mode="supplement", joint=True, paired_birth=True, two_hit_pairs=True)


def first_index(sequence, emitted):
    return next((i for i, pid in enumerate(sequence) if pid in emitted), len(sequence))


def new_id_oracle(emitted, circles, labels, gap_limits):
    """Use true current labels only to test one-way aliases at their first publication."""
    retagged = {}
    public, last_truth = {}, {}
    counters = Counter()
    earliest_used = max(emitted.values(), default=0) + 1
    ordered = sorted(emitted, key=lambda pid: (circles[pid], pid))
    for scan, live in groupby(ordered, key=circles.__getitem__):
        live = list(live)
        known = [pid for pid in live if emitted[pid] in public]
        inactive = [pid for pid in live if emitted[pid] not in public]
        occupied = set()
        for pid in known:
            old = emitted[pid]
            ident = public[old]
            if ident in occupied:
                ident = earliest_used
                earliest_used += 1
                public[old] = ident
                counters["return_collision"] += 1
            occupied.add(ident)
            retagged[pid] = ident
        for pid in inactive:
            old = emitted[pid]
            ident = old
            history = last_truth.get(labels[pid])
            if history is not None and gap_limits[0] <= scan - history[0] <= gap_limits[1]:
                if history[1] in occupied:
                    counters["occupied_old_public_id"] += 1
                else:
                    ident = history[1]
                    counters["reattached_new_raw_id"] += 1
            occupied.add(ident)
            public[old] = ident
            retagged[pid] = ident
        for pid in live:
            if labels[pid] > 0:
                last_truth[labels[pid]] = scan, retagged[pid]
    return retagged, counters


def audit(path):
    quality = load_quality_reference()
    tracker = reference_tracker()
    state, labels, circles, timestamps, emitted = {}, {}, {}, {}, {}
    truth = defaultdict(list)
    raw_first, public_first = {}, {}
    actual_raw_by_scan = {}
    actual_emitted_by_scan = {}
    truth_first_raw = {}
    prior_output_confidence = {}
    with patch("tracker_compat.baseline_circle", tracker.track):
        for scan, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            for point in points:
                pid = point["PntID"]
                timestamps[pid] = point["Time"]
                target = labels[pid]
                if target > 0:
                    truth[target].append(pid)
            rows, features = decisions_with_features(points, state)
            confidence = quality.quality_model.predict(features, num_threads=1) if rows else ()
            all_raw = {}
            visible = {}
            for record, prob in zip(rows, confidence):
                pid, batch = record["PntID"], record["BatchID"]
                raw_first.setdefault(batch, scan)
                all_raw[batch] = pid
                if labels[pid] > 0:
                    truth_first_raw.setdefault(labels[pid], pid)
                if prob >= quality.threshold:
                    public_first.setdefault(batch, scan)
                    emitted[pid] = batch
                    prior_output_confidence[pid] = state.get("association_confidence", {}).get(pid)
                    visible[batch] = pid
            actual_raw_by_scan[scan] = all_raw
            actual_emitted_by_scan[scan] = visible

    prefix, suffix, switches, identity_runs, misassignment_runs = (Counter() for _ in range(5))
    switch_events, brief_alias = [], {}
    early_add, post_add = set(), set()
    ownership = defaultdict(Counter)
    for pid, batch in emitted.items():
        if labels[pid] > 0:
            ownership[batch][labels[pid]] += 1
    owner = {batch: histogram.most_common(1)[0][0] for batch, histogram in ownership.items()}
    for target, pids in truth.items():
        pids.sort(key=lambda pid: (circles[pid], pid))
        i = first_index(pids, emitted)
        early_add.update(pids[2:i])
        post_add.update(pid for pid in pids[i:] if pid not in emitted)
        prefix["truth_tracks"] += 1
        if i >= len(pids):
            prefix["never_visible"] += 1
        elif i > 2:
            prefix["first_true_late"] += 1
            prefix["raw_true_on_time_but_hidden"] += (
                target in truth_first_raw and (circles[truth_first_raw[target]], truth_first_raw[target])
                <= (circles[pids[2]], pids[2]))
            prefix["late_missing_true_points_after_third"] += i - 2
        elif i == 2:
            prefix["first_true_at_third"] += 1
        else:
            prefix["first_true_before_third"] += 1
        if i < len(pids):
            suffix["omitted_after_first_true"] += len(pids[i:]) - sum(pid in emitted for pid in pids[i:])
        matched = [pid for pid in pids if pid in emitted]
        runs = [(batch, list(group)) for batch, group in groupby(matched, emitted.__getitem__)]
        for n, (batch, sequence) in enumerate(runs):
            length = len(sequence)
            identity_runs["total"] += 1
            if owner[batch] != target:
                misassignment_runs[f"{'1' if length == 1 else '2_to_3' if length <= 3 else '4_plus'}_runs"] += 1
                misassignment_runs["truth_points"] += length
                misassignment_runs[f"{'1' if length == 1 else '2_to_3' if length <= 3 else '4_plus'}_points"] += length
            if n == 0:
                continue
            previous, older = runs[n - 1]
            pid = sequence[0]
            scan, gap = circles[pid], circles[pid] - circles[older[-1]]
            bucket = "adjacent" if gap == 1 else "gap_2_4" if gap <= 4 else "gap_5_7" if gap <= 7 else "gap_8_plus"
            switch_events.append((scan, previous, batch, target, length))
            switches[f"{bucket}/total"] += 1
            switches[f"{bucket}/dest_newly_confirmed"] += raw_first[batch] == scan
            switches[f"{bucket}/dest_first_visible"] += public_first[batch] == scan
            switches[f"{bucket}/dest_owner_other_truth"] += owner[batch] != target
            switches[f"{bucket}/incoming_single_point"] += length == 1
            switches[f"{bucket}/incoming_four_plus"] += length >= 4
            old_assignment = actual_raw_by_scan[scan].get(previous)
            if old_assignment is None:
                switches[f"{bucket}/previous_no_raw_match"] += 1
            elif labels[old_assignment] <= 0:
                switches[f"{bucket}/previous_assigned_clutter"] += 1
            elif labels[old_assignment] != target:
                switches[f"{bucket}/previous_assigned_other_truth"] += 1
            is_brief = (n + 1 < len(runs) and length <= 3 and runs[n + 1][0] == previous
                        and circles[runs[n + 1][1][0]] - scan <= 5)
            if is_brief:
                switches[f"{bucket}/return_to_previous_within_5_scans"] += 1
                for k in sequence:
                    if previous not in actual_emitted_by_scan[circles[k]]:
                        brief_alias[k] = previous
            raw_link = prior_output_confidence[pid]
            if raw_link is not None:
                switches[f"{bucket}/dest_link_confidence_ge_75"] += raw_link >= .75

    event_key = {(scan, a, b): length for scan, a, b, _, length in switch_events}
    switches["reciprocal"] = sum((scan, b, a) in event_key for scan, a, b, _, _ in switch_events)
    switches["reciprocal_four_plus"] = sum(length >= 4 and event_key.get((scan, b, a), 0) >= 4
                                           for scan, a, b, _, length in switch_events)
    baseline = evaluate_packet(labels, emitted, circles, timestamps)
    assert baseline["id_switches"] == len(switch_events), path.name

    ideal_offset = max(emitted.values(), default=0) + 1
    ideal = {pid: (ideal_offset + labels[pid] if labels[pid] > 0 else bid) for pid, bid in emitted.items()}
    oracle_any, improvement_any = new_id_oracle(emitted, circles, labels, (1, 16))
    oracle_dormant, improvement_dormant = new_id_oracle(emitted, circles, labels, (4, 16))
    pure_ids = {pid: (ideal_offset + labels[pid] if labels[pid] > 0 and len(ownership[bid]) == 1 else bid)
                for pid, bid in emitted.items()}
    variations = {
        "baseline": baseline,
        "oracle_identity_only": evaluate_packet(labels, ideal, circles, timestamps),
        "oracle_identity_and_timely_first": evaluate_packet(
            labels, ideal | {pid: ideal_offset + labels[pid] for pid in early_add}, circles, timestamps),
        "oracle_identity_and_fill_after_first": evaluate_packet(
            labels, ideal | {pid: ideal_offset + labels[pid] for pid in post_add}, circles, timestamps),
        "oracle_new_id_reatach_1_to_16": evaluate_packet(labels, oracle_any, circles, timestamps),
        "oracle_dormant_reatach_4_to_16": evaluate_packet(labels, oracle_dormant, circles, timestamps),
        "oracle_pure_aliases": evaluate_packet(labels, pure_ids, circles, timestamps),
        "oracle_retag_brief_only_if_old_id_physically_unused": evaluate_packet(
            labels, emitted | brief_alias, circles, timestamps),
    }
    data = {"startup": prefix, "coverage_after_start": suffix, "switches": switches,
            "identity_runs": identity_runs, "wrong_owner_runs": misassignment_runs,
            "oracle_new_id": improvement_any, "oracle_dormant": improvement_dormant,
            "counts": {"brief_candidates_unused_old_id": len(brief_alias),
                       "two_hit_confirmations": tracker.early_confirmations, "path_confirmations": tracker.confirmations}}
    print(path.name, round(baseline["score"]["total"], 4), baseline["id_switches"], dict(prefix), flush=True)
    return variations, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.suffix.lower() != ".json":
        parser.error("--report must be JSON")
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(audit, paths))
    names = results[0][0]
    statistics = {name: aggregate_packets(variants[name] for variants, _ in results) for name in names}
    counters = defaultdict(Counter)
    for _, data in results:
        for key, items in data.items():
            counters[key].update(items)
    report = {"packet_ids": args.packets, "asset_sha256": {name: file_sha256(path) for name, path in ASSETS.items()},
              "scores": statistics, "diagnostics": dict(counters),
              "by_packet": dict(zip(args.packets, (detail for _, detail in results)))}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, scoring in statistics.items():
        print(name, scoring["score"]["total"], scoring["id_switches"], scoring["split_truth_tracks"], flush=True)


if __name__ == "__main__":
    main()
