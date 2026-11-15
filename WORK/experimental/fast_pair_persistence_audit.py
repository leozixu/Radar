"""Find the first loss of new fast-pair IDs during training-only sequential replay."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from lightgbm import Booster

from fast_pair_center import FastPairCenter, ORDINARY_WIDTH
from paired_returns import ASSETS, HERE, pair_offset, AMPLITUDE_TOLERANCE
from causal_birth_paths import POINT, POSITION, SCAN, BirthOnline, parallel
from raw_birth_stage_audit import EMISSION_MODELS, PAIR_MODEL, PROFILE, RAW_MODEL, emissions, models
from raw_timing_detection import RawReturnGate
from emission_identity import EmissionFeatures
from identity_rejoin import load_quality_reference
from baseline_cv import residual
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
import replay_validation as replay
import tracker_compat


class TraceFastCenter(FastPairCenter):
    def begin(self, tracks, raw, times, circle, state):
        self.prior = state.get("previous_pairs", ())
        self.frame = super().begin(tracks, raw, times, circle, state)
        return self.frame

    def protect_fallback(self, frame, tracks, positions, circle, matches, held):
        accepted, withheld = super().protect_fallback(frame, tracks, positions, circle, matches, held)
        self.held = set(map(int, self.raw[held, POINT]))
        self.protected = set(map(int, self.raw[withheld, POINT])) - self.held
        return accepted, withheld


def loss_detail(group, pairing, labels, proposals, public, circle):
    identities = group["owners"]
    labels_by_pid = {int(pid): labels[int(pid)] for pid in pairing.raw[:, POINT]}
    true_points = defaultdict(list)
    for pid, label in labels_by_pid.items():
        if label in group["truth"]:
            true_points[label].append(pid)
    if not true_points:
        return None
    expected = {pid: owner for label, owner in zip(group["truth"], identities) for pid in true_points[label]}
    missed = {pid for pid, owner in expected.items() if public.get(pid) != owner}
    if not missed:
        return "intact"
    if len(true_points) < 2:
        return "companion_missing_in_raw_circle"
    same_pair = [observations for observations in pairing.raw[pairing.frame.columns]
                 if tuple(labels_by_pid[int(pid)] for pid in observations[:, POINT]) == group["truth"]]
    if not same_pair:
        return "no_detector_pair_in_raw_circle"
    prior = [observed for observed in pairing.prior if observed.confirmed and observed.owners == identities]
    if not prior:
        return "registered_pair_no_longer_available"
    before = np.stack([observed.observations for observed in prior])
    after = np.stack(same_pair)
    gap = circle - before[:, 0, SCAN]
    distance = np.abs(residual(after[None, :, :, POSITION], before[:, None, :, POSITION]).mean(axis=2))
    signature = np.abs(pair_offset(before)[:, None] - pair_offset(after)[None, :]) <= AMPLITUDE_TOLERANCE
    if not signature.any():
        return "signed_amplitude_change_exceeding_gate"
    widths = ORDINARY_WIDTH * (1 + .60 * np.maximum(0, gap - 1))[:, None]
    if not ((distance <= widths[:, None, :]).all(axis=2) & signature).any():
        return "registered_exact_match_outside_center_gate"
    if any(pid in pairing.protected for pid in missed):
        return "normal_proposal_held_by_pair_protection"
    if any(pid in pairing.held for pid in missed):
        return "held_by_mature_ambiguity"
    if any(proposals.get(pid) == expected[pid] for pid in missed):
        return "rejected_public_emission"
    if any(pid in proposals for pid in missed):
        return "published_in_other_raw_owner"
    return "available_without_proposal"


def analyze(path):
    weights = models()
    emission_weights = {name: Booster(model_file=str(path)) for name, path in EMISSION_MODELS.items()}
    pairing = TraceFastCenter(Booster(model_file=str(PAIR_MODEL)), .985, .95, wide=True,
                              fallback_floor=PROFILE["fallback_floor"])
    tracker = BirthOnline(*weights.values(), PROFILE["birth_floor"], "supplement", paired_birth=True, joint=True,
                          two_hit_pairs=True, pairing=pairing,
                          raw_gate=RawReturnGate(Booster(model_file=str(RAW_MODEL)), PROFILE["raw_floor"]),
                          association_floor=PROFILE["association_floor"])
    history, reference = EmissionFeatures(), load_quality_reference()
    labels, circles, state, active, result = {}, {}, {}, [], Counter()
    with patch.object(tracker_compat, "baseline_circle", tracker.track):
        for circle, payload in replay.iter_circle_payloads(path, labels, circles, TRAIN_PACKETS):
            points = [dict(zip(replay.POINT_KEYS, row)) for row in replay.POINT.iter_unpack(payload)]
            old_count = len(pairing.started)
            proposals, quality = tracker_compat.decisions_with_features(points, state)
            public = emissions(points, state, proposals, quality, history, weights["clock"], emission_weights, reference)
            owners = {p["PntID"]: p["BatchID"] for p in proposals}
            forward = []
            for group in active:
                delta = circle - group["started"]
                if delta > 5:
                    if group["seen"] > 0:
                        result["seen_then_intact_for_five_circles"] += 1
                    continue
                reason = loss_detail(group, pairing, labels, owners, public, circle)
                if reason is None or reason == "intact":
                    group["seen"] += reason == "intact"
                    forward.append(group)
                else:
                    result[f"first_loss/{reason}"] += 1
                    result[f"first_loss_after_{delta}_circles"] += 1
            active = forward
            for p, q, left, right in pairing.started[old_count:]:
                if labels[p] == labels[left] > 0 and labels[q] == labels[right] > 0:
                    result["correct_pair_births"] += 1
                    active.append({"started": circle, "truth": (labels[left], labels[right]),
                                   "owners": (owners[left], owners[right]), "seen": 0})
                else:
                    result["wrong_pair_births"] += 1
    result["pending_near_packet_end"] = len(active)
    print(path.name, dict(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    results = parallel(analyze, paths, args.jobs)
    args.report.write_text(json.dumps({
        "packets": args.packets, "profile": PROFILE,
        "diagnostics": sum(results, Counter()), "by_packet": dict(zip(args.packets, results)),
        "source_sha256": {name: replay.file_sha256(HERE / name) for name in (
            Path(__file__).name, "fast_pair_center.py", "paired_returns.py", "causal_birth_paths.py", "emission_identity.py")},
        "model_sha256": {key: replay.file_sha256(path) for key, path in (
            ASSETS | {"paired_returns": PAIR_MODEL, "raw_timing": RAW_MODEL} | EMISSION_MODELS).items()},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
