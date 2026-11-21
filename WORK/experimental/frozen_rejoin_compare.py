"""Evaluate causal public-ID reconnect variants with an isolated precision runtime."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, encode_tracks, file_sha256, iter_circle_payloads

from active_lineage_rejoin import CorroboratedActiveLineage, PHYSICAL_FEATURES
import identity_rejoin


MODEL = Path(__file__).with_name("models") / "active_lineage_rejoin_assoc007.txt"


class BridgeEvidence(CorroboratedActiveLineage):
    """Score reciprocal extrapolation across old and new track histories."""

    minimum_history = 3
    maximum_new_hits = 5

    def __init__(self, floor, width):
        super().__init__(floor=floor)
        self.width = width

    def probabilities(self, features):
        names = PHYSICAL_FEATURES
        gate = features[:, names.index("shared_scans")] == 0
        gate &= features[:, names.index("new_history_count")] >= 2
        gate &= features[:, names.index("raw_probability")] >= .80
        current = features[:, names.index("current_kf_azimuth"):names.index("current_kf_range") + 1]
        forward = features[:, names.index("first_old_forward_azimuth"):names.index("first_old_forward_range") + 1]
        backward = features[:, names.index("new_backwards_azimuth"):names.index("new_backwards_range") + 1]
        # Strong old-to-new fit also requires a reasonably consistent new-to-old fit.
        normalized_error = np.maximum(np.maximum(current.max(axis=1), forward.max(axis=1)),
                                      .65 * backward.max(axis=1))
        return np.where(gate, np.exp(-.5 * np.square(normalized_error / self.width)), 0.)


class SignatureLineage(CorroboratedActiveLineage):
    """Reconnect an unassigned ID on a unique, repeated raw pair signature."""

    def __init__(self, amplitude_width, elevation_width):
        super().__init__(floor=.90)
        self.amplitude_width = amplitude_width
        self.elevation_width = elevation_width

    def candidates(self, points, proposals, rows, qualities, raw_state):
        if not rows:
            return None
        lineage = self.state
        active = {track.batch_id: track for track in raw_state['tracks'][0]}
        births = [i for i, row in enumerate(rows)
                  if lineage.public.get(row['BatchID'], row['BatchID']) not in lineage.seen
                  and active[row['BatchID']].hits <= self.maximum_new_hits]
        if not births:
            return None
        scan = raw_state['circle']
        signatures = self.pairs(points, rows, scan)
        assigned = {row['BatchID'] for row in proposals}
        old = [(raw, lineage.public.get(raw, raw)) for raw in active if raw not in assigned]
        found = []
        for i, (_, public) in enumerate(old):
            observations = self.paired_history.get(public, ())
            history = lineage.history.get(public)
            if len(observations) < 2 or history is None:
                continue
            previous_scan, previous_position, _, _, _ = history.measurements[-1]
            gap = scan - previous_scan
            if not 1 <= gap <= 12:
                continue
            for j, index in enumerate(births):
                row = rows[index]
                signature = signatures.get(row['PntID'])
                if signature is None or abs(previous_position[1] - row['Elevation']) > self.elevation_width:
                    continue
                role, amp, elevation, distance = signature
                support = sum(prior_role == role and abs(prior_amp - amp) <= self.amplitude_width
                              and abs(prior_el - elevation) <= .0025 and abs(prior_range - distance) <= .0025
                              for _, prior_role, prior_amp, prior_el, prior_range in observations)
                if support >= 2:
                    found.append((i, j, .99))
        if not found:
            return None
        old_ids = [raw for raw, _ in old]
        r, c, p = np.asarray(found).T
        return old_ids, births, r.astype(int), c.astype(int), p[:, None]

    def probabilities(self, features):
        return features[:, 0]


class BroadAzimuthLineage(CorroboratedActiveLineage):
    """Relax the azimuth forecast when elevation and range still corroborate."""

    def __init__(self, floor, width):
        super().__init__(floor=floor)
        self.width = width

    def candidates(self, points, proposals, rows, qualities, raw_state):
        # Broaden only the preliminary gate. Reciprocal bridge features still use
        # their original normalization in active_lineage_rejoin.
        with patch.object(identity_rejoin, 'SCALE', np.array((.045, .025, .0035))):
            return super().candidates(points, proposals, rows, qualities, raw_state)

    def probabilities(self, features):
        names = PHYSICAL_FEATURES
        gate = features[:, names.index('shared_scans')] == 0
        gate &= features[:, names.index('new_history_count')] >= 2
        gate &= features[:, names.index('raw_probability')] >= .80
        current = features[:, names.index('current_kf_azimuth'):names.index('current_kf_range') + 1]
        forward = features[:, names.index('first_old_forward_azimuth'):names.index('first_old_forward_range') + 1]
        backward = features[:, names.index('new_backwards_azimuth'):names.index('new_backwards_range') + 1]
        motion = np.maximum(np.maximum(current, forward), .55 * backward)
        # The bridge components above use 0.004 / 0.025 / 0.0016 as their scale.
        error = np.linalg.norm(motion / np.array((35., 1.0, 3.0)), axis=1)
        return np.where(gate, np.exp(-.5 * np.square(error / self.width)), 0.)


class CapturedRuntime:
    """Observe the production proposal and quality outputs without changing the runtime."""

    def __init__(self, runtime):
        self.tracker, history = runtime.new_packet()
        self.state = {"_precision_packet": (self.tracker, history)}
        self.quality = runtime.models["quality"]
        self.original_track = self.tracker.track
        self.original_predict = self.quality.predict

    def track(self, *args, **kwargs):
        self.proposals = self.original_track(*args, **kwargs)
        return self.proposals

    def predict(self, *args, **kwargs):
        self.probabilities = self.original_predict(*args, **kwargs)
        return self.probabilities

    def public_qualities(self, public):
        if not public:
            return np.empty(0)
        by_pid = {row["PntID"]: value for row, value in zip(self.proposals, self.probabilities, strict=True)}
        return np.array([by_pid[row["PntID"]] for row in public])


def evaluated_packet(task):
    path, participant, allowed = task
    sys.path.insert(0, str(participant / "src"))
    from algorithm import load_runtime
    from lightgbm import Booster

    runtime = load_runtime(participant / "models" / "frozen.json")
    trace = CapturedRuntime(runtime)
    model = Booster(model_file=str(MODEL))
    if tuple(model.feature_name()) != PHYSICAL_FEATURES:
        raise ValueError("Archived lineage model feature schema mismatch")
    variants = {f"archive_{floor}": CorroboratedActiveLineage(model, floor)
                for floor in (.45, .65, .80)} | {
        "bridge_strict": BridgeEvidence(.82, 2.0),
        "bridge_moderate": BridgeEvidence(.72, 3.0),
        "signature_strict": SignatureLineage(.0000026, .010),
        "signature_wide": SignatureLineage(.0000040, .020),
        "wide_azimuth_strict": BroadAzimuthLineage(.80, 1.0),
        "wide_azimuth_moderate": BroadAzimuthLineage(.66, 1.25),
    }
    outputs = {"baseline": {}} | {name: {} for name in variants}
    votes = {name: defaultdict(Counter) for name in variants}
    labels, circles, times = {}, {}, {}
    with patch.object(trace.tracker, "track", trace.track), patch.object(trace.quality, "predict", trace.predict):
        for _, payload in iter_circle_payloads(path, labels, circles, allowed=allowed):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            source = {(p["Time"], p["CircleID"], p["PntID"]): (p["Azimuth"], p["Elevation"], p["Distance"])
                      for p in points}
            times.update((pid, stamp) for stamp, _, pid in source)
            public = runtime.process_circle(points, trace.state)
            encode_tracks(public, source)
            outputs["baseline"].update((row["PntID"], row["BatchID"]) for row in public)
            qualities = trace.public_qualities(public)
            for name, guard in variants.items():
                changed = guard.correct(points, trace.proposals, public, qualities, trace.state)
                if any(old["BatchID"] != new["BatchID"] for old, new in zip(public, changed, strict=True)):
                    encode_tracks(changed, source)
                votes_for_variant = votes[name]
                for previous, pid in guard.last_links:
                    owners = votes_for_variant[previous].most_common(1)
                    confidence = owners[0][0] if owners else None
                    outcome = "correct_alias" if confidence == labels[pid] else "incorrect_alias"
                    if confidence is None or labels[pid] == -1:
                        outcome = "unknown_alias"
                    guard.evidence[outcome] += 1
                for row in changed:
                    pid, raw_id = row["PntID"], row["BatchID"]
                    outputs[name][pid] = raw_id
                    if labels[pid] > 0:
                        votes_for_variant[raw_id][labels[pid]] += 1

    if any(s.keys() != outputs["baseline"].keys() for s in outputs.values()):
        raise ValueError("Identity-only evaluation changed the returned source points")
    metrics = {name: evaluate_packet(labels, series, circles, times) for name, series in outputs.items()}
    evidence = {name: variant.evidence for name, variant in variants.items()}
    print(path.name, {name: round(score['score']['total'], 4) for name, score in metrics.items()}, evidence, flush=True)
    return metrics, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--participant", type=Path, required=True)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--split", choices=("train", "tune"), default="tune")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    allowed = TRAIN_PACKETS if args.split == "train" else TUNING_PACKETS
    paths = packet_paths(args.data_dir, args.packets, allowed)
    tasks = [(path, args.participant.resolve(), allowed) for path in paths]
    if args.jobs == 1:
        results = list(map(evaluated_packet, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(evaluated_packet, tasks))
    statistics = {name: aggregate_packets(metrics[name] for metrics, _ in results) for name in results[0][0]}
    evidence = {name: sum((observed[name] for _, observed in results), Counter()) for name in results[0][1]}
    report = {
        "metadata": {"packets": args.packets, "score_version": SCORE_VERSION,
                     "config_sha256": file_sha256(args.participant / "models" / "frozen.json"),
                     "runtime_sha256": file_sha256(args.participant / "src" / "precision_tracking" / "runtime.py"),
                     "lineage_model_sha256": file_sha256(MODEL)},
        "summary": statistics, "evidence": evidence,
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print({name: summary['score']['total'] for name, summary in statistics.items()}, evidence, flush=True)


if __name__ == "__main__":
    main()
