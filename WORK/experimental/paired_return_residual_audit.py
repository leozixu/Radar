"""Attribute remaining ID changes after exact paired-return association."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from identity_rejoin import SCALE, load_quality_reference, residual, straight_line
from paired_public_identity import Sample, nearby, reliable_signature
from paired_returns import (
    AMPLITUDE, AMPLITUDE_TOLERANCE, ASSETS, CLOCK, HERE, PAIR_WIDTH, POINT as MEASUREMENT_POINT,
    BirthOnline, PairReturns,
)
from lightgbm import Booster
from emission_identity import (
    FEATURES as EMISSION_FEATURES, RAW_RESCUE, EmissionFeatures, accepted_emissions, current_raw_probabilities,
    decision_policies,
)
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, file_sha256, iter_circle_payloads
from raw_timing_detection import RawReturnGate
from tracker_compat import decisions_with_features


class TracePairs(PairReturns):
    def begin(self, tracks, raw, times, circle, state):
        self.raw = raw
        self.fallback_held = {}
        return super().begin(tracks, raw, times, circle, state)

    def protect_fallback(self, frame, tracks, positions, circle, matches, held):
        kept, deferred = super().protect_fallback(frame, tracks, positions, circle, matches, held)
        rejected = set(deferred).difference(held)
        self.fallback_held = {int(self.raw[index, MEASUREMENT_POINT]): (tracks[row].batch_id, probability)
                              for row, index, probability in matches if index in rejected}
        return kept, deferred

    def remember(self, frame, raw, output, circle, state):
        super().remember(frame, raw, output, circle, state)
        locked = {point for _, point, _ in frame.locked}
        provenance = {}
        paired = raw[frame.columns, MEASUREMENT_POINT].astype(np.int64)
        for prior, current in frame.decisions:
            members = prior.observations[:, MEASUREMENT_POINT].astype(np.int64)
            for index, new in enumerate(frame.columns[current]):
                if new in locked:
                    provenance[paired[current, index]] = (
                        members[index], prior.confirmed, members[1 - index], paired[current, 1 - index]
                    )
        state["pair_trace"] = paired, provenance, {tuple(paired[index]) for _, index in frame.decisions}


def raw_pair_failure(source, mates, paired, selected, observations):
    if not mates:
        return "mate_absent"
    reference, reference_amp, reference_clock, reference_duplicates = observations[source]
    failures = set()
    for mate in mates:
        sample, amp, clock, duplicates = observations[mate]
        key = tuple(sorted((source, mate), key=lambda pid: observations[pid][0].time))
        if key in selected:
            return "pair_selected_but_this_owner_not_locked"
        if key in paired:
            failures.add("pair_measured_but_link_not_selected")
            continue
        delay = abs(sample.time - reference.time)
        if delay != 1:
            failures.add("same_tick" if delay == 0 else "tick_distance_gt_1")
        elif reference_duplicates > 1 or duplicates > 1:
            failures.add("nonunique_tick")
        else:
            amp_gap = abs(reference_amp - amp)
            geometry = np.array((abs((reference.azimuth - sample.azimuth + .5) % 1. - .5),
                                 abs(reference.elevation - sample.elevation),
                                 abs(reference.distance - sample.distance)))
            if np.any(geometry > PAIR_WIDTH):
                failures.add("geometry")
            elif not .000004 <= amp_gap <= .032:
                failures.add("amplitude")
            elif min(reference_clock, clock) < .20 and min(reference_amp, amp) < .20:
                failures.add("clock_or_amplitude_eligibility")
            else:
                failures.add("pruned_at_pair_count_cap")
    priorities = ("pair_measured_but_link_not_selected", "pruned_at_pair_count_cap",
                  "clock_or_amplitude_eligibility", "amplitude", "geometry", "nonunique_tick",
                  "tick_distance_gt_1", "same_tick")
    return next(reason for reason in priorities if reason in failures)


def physical_alias_eligibility(old, new_pid, circle, histories, observations, emitted_labels, hits):
    preceding = sorted((t, sample) for t, sample in histories[old].items() if 0 < circle - t)
    if len(preceding) < 2:
        return "one_prior_public_output"
    scans, positions = zip(*preceding[-9:])
    gap = circle - scans[-1]
    if gap < 2 or gap > 12:
        return "below_two_scan_gap" if gap < 2 else "above_twelve_scan_gap"
    if hits[new_pid] > 10:
        return "destination_already_mature_privately"
    if sum(t < circle for t in emitted_labels[old]) < 2:
        return "fewer_than_two_known_past_true_outputs"
    values = np.array([(row.azimuth, row.elevation, row.distance) for row in positions])
    values[:, 0] = np.unwrap(values[:, 0], period=1.)
    x = np.asarray(scans) - scans[-1]
    p, v = straight_line(values, x)
    short_p, short_v = straight_line(values[-4:], x[-4:])
    point = observations[new_pid][0]
    observed = np.array([point.azimuth, point.elevation, point.distance])
    predicted = np.array((p + gap * v, short_p + gap * short_v, values[-1] + gap * v))
    d = np.linalg.norm(residual(observed, predicted) / SCALE, axis=1).min()
    return "above_physical_motion_gate" if d > 4.8 + .75 * (gap - 1) else "eligible"


def identity_switches(emitted, labels, circles, contexts, raw_first, histories, scans, observations, raw_groups,
                      emitted_hits):
    by_target = defaultdict(list)
    ownership = defaultdict(Counter)
    by_scan = defaultdict(dict)
    public_first = {}
    emitted_labels = defaultdict(list)
    for pid, output in emitted.items():
        target = labels[pid]
        by_scan[circles[pid]][output] = pid
        public_first[output] = min(circles[pid], public_first.get(output, circles[pid]))
        if target > 0:
            by_target[target].append(pid)
            ownership[output][target] += 1
            emitted_labels[output].append(circles[pid])
    owner = {batch: histogram.most_common(1)[0][0] for batch, histogram in ownership.items()}
    summary, wrong_run, switch_scans = Counter(), Counter(), {}
    detector = Counter()
    single_role = Counter()
    cohorts = defaultdict(set)
    active_pure = []
    events = []
    for target, points in by_target.items():
        points.sort(key=lambda pid: (circles[pid], pid))
        runs = [(batch, list(group)) for batch, group in groupby(points, emitted.__getitem__)]
        for index, ((old, before), (new, after)) in enumerate(zip(runs, runs[1:])):
            returning = index > 0 and runs[index - 1][0] == new
            first_run = index == 0
            events.append((old, new, before, after, returning, first_run))
            switch_scans[circles[after[0]], old, new] = len(after)
        for batch, fragment in runs:
            if owner[batch] != target:
                wrong_run[len(fragment) >= 4] += len(fragment)

    for old, new, before, after, returning, first_run in events:
        prior, pid = before[-1], after[0]
        length = len(after)
        circle = circles[pid]
        gap = circle - circles[prior]
        paired, locked, active, selected = contexts[circle]
        reciprocal = switch_scans.get((circle, new, old), 0)
        target = labels[pid]
        raw_source = raw_groups[circle].get(old)
        if old not in active:
            cohort = "inactive_source_new_destination" if raw_first[new] >= circle - 1 else "inactive_source_existing"
        elif old in by_scan[circle]:
            cohort = "active_source_public_elsewhere"
        elif raw_source is not None:
            cohort = "active_source_raw_elsewhere_hidden"
        else:
            cohort = "active_source_no_raw_match"
        summary[f"cohort/{cohort}"] += 1
        cohorts[cohort].add(target)
        if first_run and len(before) <= 3 and len(after) >= 4:
            summary["first_1_to_3_returns_then_persistent_destination"] += 1
            cohorts["first_1_to_3_returns_then_persistent_destination"].add(target)
        if first_run and owner[old] != target and owner[new] == target:
            summary["first_run_wrong_owner_to_own_majority_destination"] += 1
            cohorts["first_run_wrong_owner_to_own_majority_destination"].add(target)
        if owner[old] == target and owner[new] == target:
            summary["both_public_ids_are_same_truth_majority"] += 1
            cohorts["both_public_ids_are_same_truth_majority"].add(target)
        if (cohort == "active_source_no_raw_match" and len(ownership[old]) == len(ownership[new]) == 1
                and owner[old] == target == owner[new]):
            summary["pure_active_coasting"] += 1
            cohorts["pure_active_coasting"].add(target)
            if circle == public_first[new]:
                summary["pure_active_coasting_first_public_destination"] += 1
                cohorts["pure_active_coasting_first_public_destination"].add(target)
                active_pure.append((circle, old, new))
                eligibility = physical_alias_eligibility(
                    old, pid, circle, histories, observations, emitted_labels, emitted_hits)
                summary[f"pure_first_public_eligibility/{eligibility}"] += 1
        if gap > 1:
            skipped_true = sum(target in scans[scan] for scan in range(circles[prior] + 1, circle))
            summary["switch_after_missing_a_raw_true_return"] += skipped_true > 0
            if skipped_true:
                cohorts["switch_after_missing_a_raw_true_return"].add(target)
        cohorts["total"].add(target)
        summary["total"] += 1
        summary["adjacent"] += gap == 1
        summary["reciprocal"] += reciprocal > 0
        summary["persistent_reciprocal"] += reciprocal >= 4 and length >= 4
        summary["destination_persistent"] += length >= 4
        summary["source_visible_now"] += old in by_scan[circle]
        summary["source_active_now"] += old in active
        summary["destination_newly_confirmed"] += raw_first[new] == circle
        summary["destination_recently_confirmed"] += raw_first[new] >= circle - 1
        summary["destination_owned_by_other_truth"] += owner[new] != labels[pid]
        summary["source_owned_by_other_truth"] += owner[old] != labels[pid]
        summary["returning_after_interruption"] += returning
        summary["returning_after_1_to_3_wrong_returns"] += returning and len(before) <= 3
        summary["source_origin_within_3_scans"] += raw_first[old] >= circle - 3
        summary["current_in_exact_geometry_pair"] += any(pid in pair for pair in paired)
        summary["current_locked_by_pair"] += pid in locked
        if pid in locked:
            linked_pid, prior_confirmed, older_mate, current_mate = locked[pid]
            linked_true = labels[linked_pid] == labels[pid]
            summary["lock_kept_same_physical_target"] += linked_true
            summary["lock_prior_was_confirmed_pair"] += prior_confirmed
            summary["lock_prior_kept_same_target_and_confirmed"] += linked_true and prior_confirmed
            summary["lock_reverted_from_unlocked_prior_output"] += linked_true and prior not in contexts[circles[prior]][1]
            summary["lock_prior_real_return_was_within_3_scans"] += linked_true and circle - circles[linked_pid] <= 3
            summary["lock_prior_return_was_linked_to_destination"] += emitted.get(linked_pid) == new
            summary["lock_resumed_after_1_to_3_wrong_returns"] += linked_true and returning and len(before) <= 3
            summary["lock_overrode_source_owned_by_other_truth"] += linked_true and owner[old] != labels[pid]
            unsafe_circle = circles[prior]
            unsafe_pairs, unsafe_locks, _, unsafe_selected = contexts[unsafe_circle]
            if linked_true and prior not in unsafe_locks:
                mate = labels[older_mate]
                if mate <= 0 or mate != labels[current_mate] or mate == labels[pid]:
                    detector["no_stable_truth_mate"] += 1
                else:
                    mates = scans[unsafe_circle].get(mate, ())
                    issue = raw_pair_failure(prior, mates, unsafe_pairs, unsafe_selected, observations)
                    detector[issue] += 1
                    if issue == "mate_absent":
                        return_time = observations[linked_pid][0].time
                        mate_time = observations[older_mate][0].time
                        role = "earlier" if return_time < mate_time else "later"
                        single_role[role] += 1
        if old in active or raw_first[new] < circle - 1:
            continue
        summary["inactive_source_recent_destination"] += 1
        summary["inactive_source_gap_8_plus"] += gap >= 8
        incoming = histories[new][circle]
        for anchor, anchor_pid in by_scan[circle].items():
            if anchor == new or labels[anchor_pid] <= 0 or labels[anchor_pid] == labels[pid]:
                continue
            sample = histories[anchor][circle]
            if not nearby(incoming, sample):
                continue
            signature = reliable_signature(histories[old], histories[anchor], circle, max_gap=24)
            if signature is None:
                continue
            if (abs(incoming.amp - sample.amp - signature.amplitude) <= AMPLITUDE_TOLERANCE and
                    abs(incoming.elevation - sample.elevation - signature.elevation) <= .001):
                summary["anchored_inactive_source_recent_destination"] += 1
                break
    wrong = {"points_in_short_wrong_run": wrong_run[False], "points_in_persistent_wrong_run": wrong_run[True]}
    return summary, wrong, detector, single_role, {key: len(values) for key, values in cohorts.items()}, active_pure


def merge_nonoverlapping(emitted, circles, proposed):
    """Diagnostic only: merge coherent output IDs whose emitted scans do not overlap."""
    by_output = defaultdict(set)
    for point, batch in emitted.items():
        by_output[batch].add(circles[point])
    aliases = {}

    def canonical(batch):
        while batch in aliases:
            batch = aliases[batch]
        return batch

    for _, old, new in sorted(proposed):
        old, new = canonical(old), canonical(new)
        if old != new and by_output[old].isdisjoint(by_output[new]):
            aliases[new] = old
            by_output[old].update(by_output[new])
    return {point: canonical(batch) for point, batch in emitted.items()}, len(aliases)


def audit(spec):
    path, wide, floor, emission_mode, raw_floor, association_floor = spec
    weights = [Booster(model_file=str(asset)) for asset in ASSETS.values()]
    pairing = TracePairs(Booster(model_file=str(HERE / "models/paired_returns.txt")), .985, .95, early=True,
                         wide=wide, fallback_floor=floor)
    gate = (RawReturnGate(Booster(model_file=str(HERE / "models/raw_timing_detection.txt")), raw_floor)
            if raw_floor is not None else None)
    tracker = BirthOnline(*weights, .86, "supplement", paired_birth=True, joint=True, two_hit_pairs=True, pairing=pairing,
                          raw_gate=gate, association_floor=association_floor)
    runtime = load_quality_reference()
    if emission_mode:
        emission = {kind: Booster(model_file=str(HERE / "models" / f"emission_{kind}.txt"))
                    for kind in ("detection", "ownership")}
        emission_history = EmissionFeatures()
        thresholds = decision_policies(gate)[emission_mode]
        rescue_raw_floor = RAW_RESCUE[emission_mode][1] if emission_mode in RAW_RESCUE else None
    raw_published, raw_gated, raw_q, fallback_held, raw_groups = set(), set(), {}, set(), {}
    fallback_proven, fallback_labeled = {}, Counter()
    past_public = defaultdict(Counter)
    labels, circles, stamps, emitted, raw_first, state, emitted_hits = {}, {}, {}, {}, {}, {}, {}
    truth_points, histories = defaultdict(list), defaultdict(dict)
    contexts, scans, observations = {}, {}, {}
    with patch("tracker_compat.baseline_circle", tracker.track):
        for circle, frame in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(frame)]
            source = {row["PntID"]: row for row in points}
            for pid, row in source.items():
                stamps[pid] = row["Time"]
                if labels[pid] > 0:
                    truth_points[labels[pid]].append(pid)
            proposals, quality = decisions_with_features(points, state)
            probabilities = runtime.quality_model.predict(quality, num_threads=1) if proposals else np.empty(0)
            keep = probabilities >= runtime.threshold
            if emission_mode:
                features = emission_history.build(points, proposals, quality, probabilities, state, weights[-1])
                if proposals:
                    detection = emission["detection"].predict(features, num_threads=1)
                    ownership = emission["ownership"].predict(features, num_threads=1)
                    trusted = features[:, EMISSION_FEATURES.index("trusted_proposals")]
                    raw_probability = current_raw_probabilities(points, proposals, state) if rescue_raw_floor else None
                    keep = accepted_emissions(keep, detection, ownership, trusted < 3, thresholds,
                                              raw_probability, rescue_raw_floor)
            rows = [row for row, accepted in zip(proposals, keep) if accepted]
            raw_published.update(row["PntID"] for row in proposals)
            fallback_held.update(pairing.fallback_held)
            occupied = {row["BatchID"] for row in rows}
            for held_pid, (owner, probability) in pairing.fallback_held.items():
                if labels[held_pid] <= 0:
                    continue
                prior_votes = past_public[owner]
                known = prior_votes.most_common(1)
                if not known or known[0][1] < 3 or known[0][1] < .80 * prior_votes.total():
                    fallback_labeled["not_enough_past_public_identity"] += 1
                elif known[0][0] != labels[held_pid]:
                    fallback_labeled["different_past_public_identity"] += 1
                    fallback_labeled[f"different_owner_probability_{int(probability * 10) / 10:g}"] += 1
                else:
                    fallback_labeled["same_past_public_identity"] += 1
                    fallback_labeled[f"same_owner_probability_{int(probability * 10) / 10:g}"] += 1
                    if owner not in occupied:
                        fallback_proven[held_pid] = owner
                        occupied.add(owner)
            if gate is not None:
                for point, probability in zip(points, state["raw_return_probability"]):
                    pid = point["PntID"]
                    if labels[pid] > 0:
                        raw_q[pid] = probability
                        if np.isfinite(probability) and probability < raw_floor:
                            raw_gated.add(pid)
            raw_groups[circle] = {row["BatchID"]: row["PntID"] for row in proposals}
            for track in state["tracks"][0]:
                raw_first.setdefault(track.batch_id, circle)
            hits = {track.batch_id: track.hits for track in state["tracks"][0]}
            paired, locked, selected = state["pair_trace"]
            contexts[circle] = ({tuple(pair) for pair in paired}, locked,
                                {track.batch_id for track in state["tracks"][0]}, selected)
            nearby_truth = defaultdict(list)
            duplicate_times = Counter(row["Time"] for row in points)
            for row, measure in zip(points, pairing.raw):
                pid = row["PntID"]
                label = labels[pid]
                if label <= 0:
                    continue
                raw = Sample(row["Time"], row["Amp"], row["Elevation"], row["Azimuth"], row["Distance"])
                observations[pid] = (raw, measure[AMPLITUDE], measure[CLOCK], duplicate_times[row["Time"]])
                nearby_truth[label].append(pid)
            scans[circle] = nearby_truth
            for row in rows:
                pid, owner = row["PntID"], row["BatchID"]
                emitted[pid] = owner
                emitted_hits[pid] = hits[owner]
                if labels[pid] > 0:
                    past_public[owner][labels[pid]] += 1
                raw = source[pid]
                histories[owner][circle] = Sample(raw["Time"], raw["Amp"], raw["Elevation"], raw["Azimuth"], raw["Distance"])
    summary, wrong, detector, single_role, affected, active_pure = identity_switches(
        emitted, labels, circles, contexts, raw_first, histories, scans, observations, raw_groups, emitted_hits)
    score = evaluate_packet(labels, emitted, circles, stamps)
    assert score["id_switches"] == summary["total"], (score["id_switches"], summary["total"])
    new_id = 1 + max(emitted.values(), default=0)
    perfect = {pid: new_id + labels[pid] if labels[pid] > 0 else batch for pid, batch in emitted.items()}
    pure_rejoin, pure_alias_count = merge_nonoverlapping(emitted, circles, active_pure)
    summary["pure_active_coasting_nonoverlapping_aliases"] = pure_alias_count
    late, mature, startup_same_id = {}, {}, {}
    occupied = {(circles[pid], batch) for pid, batch in emitted.items()}
    startup = Counter()
    for target, sequence in truth_points.items():
        ordinal = next((i for i, pid in enumerate(sequence) if pid in emitted), len(sequence))
        startup["truth_tracks"] += 1
        startup["late_truth_tracks"] += ordinal > 2
        startup["never_visible"] += ordinal == len(sequence)
        if len(sequence) >= 3 and ordinal > 2:
            third = sequence[2]
            initial_reason = ("raw_proposal_filtered" if third in raw_published else
                              "fallback_deferred" if third in fallback_held else
                              "raw_below_gate" if third in raw_gated else "no_raw_proposal")
            startup[f"late_third/{initial_reason}"] += 1
        late.update((pid, new_id + target) for pid in sequence[2:ordinal])
        mature.update((pid, new_id + target) for pid in sequence[ordinal:] if pid not in emitted)
        if ordinal < len(sequence):
            first_id = emitted[sequence[ordinal]]
            if len(past_public[first_id]) == 1:
                for pid in sequence[2:ordinal]:
                    key = circles[pid], first_id
                    if key not in occupied:
                        startup_same_id[pid] = first_id
                        occupied.add(key)
    missing = Counter()
    for phase, group in (("late", late), ("mature", mature)):
        missing[f"{phase}/proposed_but_filtered"] = sum(pid in raw_published for pid in group)
        missing[f"{phase}/fallback_held"] = sum(pid in fallback_held for pid in group)
        missing[f"{phase}/below_raw_gate_without_proposal"] = sum(
            pid in raw_gated and pid not in raw_published and pid not in fallback_held for pid in group)
        missing[f"{phase}/unmatched_raw_q_ge_90"] = sum(
            bool(raw_q.get(pid, 0.) >= .90) and pid not in raw_published and pid not in fallback_held for pid in group)
        missing[f"{phase}/not_proposed_or_fallback_held"] = sum(
            pid not in raw_published and pid not in fallback_held for pid in group)
    oracles = {
        "exact_pairs": score,
        "pure_active_coasting_aliases_identical_points": evaluate_packet(labels, pure_rejoin, circles, stamps),
        "startup_fill_own_first_pure_id": evaluate_packet(labels, emitted | startup_same_id, circles, stamps),
        "fallback_fill_proven_past_owner": evaluate_packet(labels, emitted | fallback_proven, circles, stamps),
        "perfect_identity_on_observed_points": evaluate_packet(labels, perfect, circles, stamps),
        "perfect_identity_and_fill_late_birth": evaluate_packet(labels, perfect | late, circles, stamps),
        "perfect_identity_and_fill_mature": evaluate_packet(labels, perfect | mature, circles, stamps),
    }
    print(path.stem, {name: round(result["score"]["total"], 5) for name, result in oracles.items()},
          score["id_switches"], dict(affected), dict(missing), dict(fallback_labeled), flush=True)
    return oracles, summary, wrong, startup, detector, single_role, affected, missing, dict(fallback_labeled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--fallback-floor", type=float, default=0.)
    parser.add_argument("--emission", choices=tuple(decision_policies(True)))
    parser.add_argument("--raw-floor", type=float)
    parser.add_argument("--association-floor", type=float, default=.20)
    args = parser.parse_args()
    if not 0. <= args.fallback_floor <= 1.:
        parser.error("--fallback-floor must be between 0 and 1")
    if args.raw_floor is not None and not 0 <= args.raw_floor <= 1:
        parser.error("--raw-floor must be between 0 and 1")
    if not 0 < args.association_floor < 1 or (args.emission in RAW_RESCUE and args.raw_floor is None):
        parser.error("expected valid association floor and raw gate for a raw rescue policy")
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        result = list(pool.map(audit, [(path, args.wide, args.fallback_floor, args.emission, args.raw_floor,
                                       args.association_floor) for path in paths]))
    assets = ASSETS | {"paired": HERE / "models/paired_returns.txt"}
    if args.emission:
        assets.update({f"emission_{kind}": HERE / "models" / f"emission_{kind}.txt"
                       for kind in ("detection", "ownership")})
    if args.raw_floor is not None:
        assets["raw_timing_detection"] = HERE / "models/raw_timing_detection.txt"
    report = {
        "packets": args.packets, "score_version": SCORE_VERSION,
        "wide": args.wide, "fallback_floor": args.fallback_floor, "emission": args.emission,
        "raw_floor": args.raw_floor, "association_floor": args.association_floor,
        "model_sha256": {key: file_sha256(path) for key, path in assets.items()},
        "source_sha256": {name: file_sha256(HERE / name) for name in (
            "paired_returns.py", "causal_birth_paths.py", "emission_identity.py", "paired_return_residual_audit.py")},
        "variants": {variant: aggregate_packets(row[0][variant] for row in result) for variant in result[0][0]},
    }
    detail_keys = ("switches", "wrong_owner", "startup", "detector_at_prior_unsafe_output", "sole_return_timing_role",
                   "split_truth_by_nonexclusive_cohort", "missing_points", "fallback_prior_owner_truth")
    for i, key in enumerate(detail_keys, 1):
        report[key] = dict(sum((Counter(row[i]) for row in result), Counter()))
    report["by_packet"] = {packet: dict(zip(detail_keys, row[1:])) for packet, row in zip(args.packets, result)}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
