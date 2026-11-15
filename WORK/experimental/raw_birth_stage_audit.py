"""Trace the missing first three true returns in the wide-pair/raw-gated profile."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lightgbm import Booster

from exact_pair_single_audit import BirthTrace, PAIR_MODEL
from paired_returns import ASSETS, HERE, PairReturns
from causal_birth_paths import (AMPLITUDE, CLOCK, LINK_WIDTH, POINT, POSITION, SCAN,
                                BirthOnline, eligible)
from emission_identity import (EmissionFeatures, FEATURES, RAW_RESCUE, accepted_emissions,
                               current_raw_probabilities)
from identity_rejoin import load_quality_reference
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from raw_timing_detection import RawReturnGate
from replay_validation import POINT as WIRE_POINT, POINT_KEYS, file_sha256, iter_circle_payloads
from baseline_cv import residual
import tracker_compat


RAW_MODEL = HERE / "models/raw_timing_detection.txt"
EMISSION_MODELS = {model: HERE / f"models/emission_{model}.txt" for model in ("detection", "ownership")}
PROFILE = dict(association_floor=.035, raw_floor=.025, fallback_floor=.96, birth_floor=.86,
               emission="rescue25_raw80")


class PoolAudit(BirthTrace):
    def __init__(self, model, labels):
        super().__init__(model)
        self.labels = labels

    def propose(self, points, circle):
        result = super().propose(points, circle)
        self.historic_points = {int(pid) for _, rows in self.observations for pid in rows[:, POINT]
                                if self.labels[int(pid)] > 0}
        self.historic_arcs = {
            (int(a), int(b)) for _, arcs in self.arcs for a, b in zip(arcs.first[:, POINT], arcs.middle[:, POINT])
            if self.labels[int(a)] == self.labels[int(b)] > 0
        }
        return result


class PairAudit(PairReturns):
    def begin(self, tracks, raw, times, circle, state):
        self.raw = raw
        return super().begin(tracks, raw, times, circle, state)

    def protect_fallback(self, frame, tracks, positions, circle, matches, held):
        resolved, withheld = super().protect_fallback(frame, tracks, positions, circle, matches, held)
        self.holds = set(map(int, self.raw[held, POINT]))
        self.protected = set(map(int, self.raw[withheld, POINT])) - self.holds
        return resolved, withheld


class LegacyAudit(BirthOnline):
    def promote_legacy(self, points, state, active, position, amp, circle, remaining, output, next_id, observe,
                       reserved_keys):
        self.available = set(map(int, self.pairing.raw[remaining, POINT]))
        self.mature_or_pair = set(output)
        self.previous_young = frozenset(state.get("young_keys", {}).values())
        tentative, pending, taken, counter = super().promote_legacy(
            points, state, active, position, amp, circle, remaining, output, next_id, observe, reserved_keys)
        self.legacy_confirmation = taken
        self.legacy_used = self.available.difference(self.pairing.raw[remaining, POINT])
        self.legacy_advance = {chain[-1] for chain in pending.values()}
        return tentative, pending, taken, counter


def models():
    return {key: Booster(model_file=str(path)) for key, path in ASSETS.items()}


def make_tracker(weights, analyzer):
    gate = RawReturnGate(Booster(model_file=str(RAW_MODEL)), PROFILE["raw_floor"])
    pair_type = PairAudit if analyzer else PairReturns
    tracker_type = LegacyAudit if analyzer else BirthOnline
    pairs = pair_type(Booster(model_file=str(PAIR_MODEL)), .985, .95, wide=True,
                      fallback_floor=PROFILE["fallback_floor"])
    return tracker_type(*weights.values(), PROFILE["birth_floor"], "supplement", paired_birth=True, joint=True,
                        two_hit_pairs=True, pairing=pairs, raw_gate=gate,
                        association_floor=PROFILE["association_floor"])


def emissions(points, tracker_state, proposals, quality, emission_features, clock, weights, quality_reference):
    probabilities = quality_reference.quality_model.predict(quality, num_threads=1) if proposals else np.empty(0)
    features = emission_features.build(points, proposals, quality, probabilities, tracker_state, clock)
    if not proposals:
        return {}
    reveal, raw_floor = RAW_RESCUE[PROFILE["emission"]]
    detection = weights["detection"].predict(features, num_threads=1)
    ownership = weights["ownership"].predict(features, num_threads=1)
    young = features[:, FEATURES.index("trusted_proposals")] < 3
    source = current_raw_probabilities(points, proposals, tracker_state)
    keep = accepted_emissions(probabilities >= quality_reference.threshold, detection, ownership, young,
                              (0., reveal, 0.), source, raw_floor)
    return {p["PntID"]: p["BatchID"] for p, decision in zip(proposals, keep) if decision}


def audit(path):
    labels, circles, times = {}, {}, {}
    weights = models()
    output_weights = {name: Booster(model_file=str(model)) for name, model in EMISSION_MODELS.items()}
    history, reference = EmissionFeatures(), load_quality_reference()
    tracker = make_tracker(weights, analyzer=True)
    pool = PoolAudit(weights["birth"], labels)
    state = {"birth_path_pool": pool}
    tracker.birth = pool
    status, emitted, measurement, snapshots = {}, {}, {}, {}
    with patch.object(tracker_compat, "baseline_circle", tracker.track):
        for circle, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in WIRE_POINT.iter_unpack(payload)]
            times.update((p["PntID"], p["Time"]) for p in points)
            raw, features = tracker_compat.decisions_with_features(points, state)
            public = emissions(points, state, raw, features, history, weights["clock"], output_weights, reference)
            emitted.update(public)
            pointids = set(map(int, tracker.pairing.raw[:, POINT]))
            current_candidates = set(map(int, pool.candidates[:, POINT]))
            born = set(map(int, tracker.legacy_confirmation))
            raw_published = {p["PntID"] for p in raw}
            pooled = {int(pid) for t, rows in pool.observations if t == circle for pid in rows[:, POINT]}
            raw_scores = state["raw_return_probability"]
            current_past = pool.historic_points
            snapshots[circle] = (current_past, pool.historic_arcs, pool.true_paths(labels), tracker.previous_young)
            for obs, q in zip(tracker.pairing.raw, raw_scores):
                pid = int(obs[POINT])
                if labels[pid] <= 0:
                    continue
                measurement[pid] = obs
                if pid in raw_published:
                    ownership_stage = ("mature_or_pair" if pid in tracker.mature_or_pair else
                                       "legacy_birth" if pid in born else "path_or_two_hit_birth")
                    emission_stage = "public" if pid in public else "rejected_emission"
                elif pid in tracker.pairing.holds:
                    ownership_stage, emission_stage = "mature_ambiguity_held", "no_output"
                elif pid in tracker.pairing.protected:
                    ownership_stage, emission_stage = "pair_fallback_held", "no_output"
                elif q < PROFILE["raw_floor"] and pid not in tracker.available:
                    ownership_stage, emission_stage = "below_raw_gate", "no_output"
                elif pid in tracker.legacy_used:
                    ownership_stage = "legacy_advanced" if pid in tracker.legacy_advance else "legacy_rejected"
                    emission_stage = "no_output"
                elif pid in current_candidates:
                    ownership_stage, emission_stage = "birth_available", "no_output"
                elif pid in pointids and not len(eligible(obs[None, :])):
                    ownership_stage, emission_stage = "birth_eligibility_failed", "no_output"
                else:
                    ownership_stage, emission_stage = "other", "no_output"
                status[pid] = (ownership_stage, emission_stage, pid in current_candidates, pid in pooled, float(q))

    return analyze(labels, circles, times, emitted, status, measurement, snapshots)


def analyze(labels, circles, times, emitted, status, measurement, snapshots):
    truths, diagnostic, misses = defaultdict(list), Counter(), {}
    failed_motion = []
    for pid, label in labels.items():
        if label > 0:
            truths[label].append(pid)
    for label, ids in truths.items():
        ids.sort(key=lambda pid: (circles[pid], pid))
        if len(ids) < 3:
            continue
        first = next((i for i, pid in enumerate(ids) if pid in emitted), len(ids))
        if first <= 2:
            continue
        diagnostic["late_targets"] += 1
        a, b, c = ids[:3]
        for pid in ids[2:first]:
            misses[pid] = label
            diagnostic["startup_misses"] += 1
            diagnostic["startup_misses_q90"] += status[pid][4] >= .90
        for phase, pid in enumerate((a, b, c), start=1):
            reason, result, candidate, pooled, q = status[pid]
            diagnostic[f"return_{phase}/{reason}/{result}"] += 1
            diagnostic[f"return_{phase}/path_candidate"] += candidate
            diagnostic[f"return_{phase}/retained_in_pool"] += pooled
            diagnostic[f"return_{phase}/q90"] += q >= .90

        present, arcs, paths, pending = snapshots[circles[c]]
        gap1, gap2 = circles[b] - circles[a], circles[c] - circles[b]
        diagnostic[f"truth_gap/{gap1}+{gap2}"] += 1
        reason = explain((a, b, c), (gap1, gap2), snapshots[circles[c]], status, measurement)
        diagnostic[f"explanation/{reason}"] += 1
        if reason in {"first_second_link_outside_gate", "true_return_circle_gap_exceeds_path_pool"}:
            old, middle, latest = (measurement[pid][POSITION] for pid in (a, b, c))
            span = residual(middle, old) / max(gap1, 1)
            tail = residual(latest, middle) / max(gap2, 1)
            failed_motion.append({
                "reason": reason, "gaps": [gap1, gap2], "q_min": min(status[pid][4] for pid in (a, b, c)),
                "first_velocity_Az_El_Range": span.tolist(), "second_velocity_Az_El_Range": tail.tolist(),
                "first_input_displacement_to_width": float(np.linalg.norm(span * gap1 /
                    (LINK_WIDTH * (1 + .45 * (gap1 - 1))))),
                "third_forecast_error_to_width": float(np.linalg.norm((tail - span) * gap2 /
                    (np.array([.028, .120, .012]) * (1 + .4 * (gap2 - 1))))),
            })
        if (a, b) in pending:
            diagnostic["legacy_true_tentative_before_third"] += 1
            diagnostic[f"legacy_true_tentative/{status[c][0]}"] += 1
        if (a,) in pending or (b,) in pending:
            diagnostic["legacy_true_seed_before_third"] += 1
        diagnostic["all_three_raw_q90"] += all(status[pid][4] >= .90 for pid in (a, b, c))
    baseline = evaluate_packet(labels, emitted, circles, times)
    offset = max(emitted.values(), default=0) + 1
    # Historical IDs are chosen only in this offline upper bound.
    output_groups = defaultdict(Counter)
    for pid, owner in emitted.items():
        if labels[pid] > 0:
            output_groups[labels[pid]][owner] += 1
    complete = emitted | {pid: output_groups[label].most_common(1)[0][0] if output_groups[label] else offset + label
                          for pid, label in misses.items()}
    return baseline, evaluate_packet(labels, complete, circles, times), diagnostic, failed_motion


def explain(path, gaps, snapshot, status, measurement):
    a, b, c = path
    first, second = gaps
    present, arcs, proposals, _ = snapshot
    if status[c][1] == "rejected_emission":
        return "third_emitted_internally_but_publicly_rejected"
    if path in proposals:
        return "true_path_passed_floor_but_lost" if proposals[path] >= PROFILE["birth_floor"] else "true_path_below_floor"
    if status[c][0] in {"mature_ambiguity_held", "pair_fallback_held", "below_raw_gate", "birth_eligibility_failed"}:
        return f"third/{status[c][0]}"
    if first not in (1, 2) or second not in (1, 2):
        return "true_return_circle_gap_exceeds_path_pool"
    for name, pid in (("first", a), ("second", b)):
        if pid not in present:
            if status[pid][3]:
                return f"{name}/entered_pool_then_removed"
            return f"{name}/never_entered_pool/{status[pid][0]}"
    if not status[c][2]:
        return f"third/never_entered_candidates/{status[c][0]}"
    A, B, C = (measurement[pid][POSITION] for pid in path)
    if (a, b) not in arcs:
        width = LINK_WIDTH * (1. + .45 * (first - 1))
        outside = np.square(residual(B, A) / width).sum() > 1.
        return "first_second_link_outside_gate" if outside else "first_second_link_pruned_from_beam"
    prediction = B + residual(B, A) / first * second
    width = np.array([.028, .120, .012]) * (1. + .4 * (second - 1))
    if np.square(residual(C, prediction) / width).sum() > 1.:
        return "third_link_outside_cv_gate"
    return "other_unlisted_candidate_path"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    if args.jobs == 1:
        results = list(map(audit, paths))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(audit, paths))
    baseline, ideal, diagnostics, motions = zip(*results)
    counts = sum(diagnostics, Counter())
    report = {
        "score_version": SCORE_VERSION, "packets": args.packets, "profile": PROFILE,
        "base": aggregate_packets(baseline), "early_start_label_oracle": aggregate_packets(ideal), "diagnostics": counts,
        "by_packet": dict(zip(args.packets, diagnostics)),
        "late_start_motion": dict(zip(args.packets, motions)),
        "model_sha256": {name: file_sha256(path) for name, path in
                         (ASSETS | {"paired_returns": PAIR_MODEL, "raw_timing": RAW_MODEL}
                          | {f"emission_{kind}": path for kind, path in EMISSION_MODELS.items()}).items()},
        "source_sha256": {name: file_sha256(HERE / name) for name in (
            "paired_returns.py", "causal_birth_paths.py", "emission_identity.py", Path(__file__).name)},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
