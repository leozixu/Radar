"""Protect established public IDs using causal local amplitude/elevation signatures."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from identity_rejoin import load_quality_reference
from identity_stage_audit import ASSETS, reference_tracker
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay


HISTORY_SCANS = 32
MAX_LAST_PAIR_GAP = 6
MIN_COOCCURRENCES = 5
TIME_RADIUS = 2
AMP_TOLERANCE = .00025
EL_TOLERANCE = .0009


@dataclass(slots=True)
class Sample:
    time: int
    amp: float
    elevation: float
    azimuth: float
    distance: float


@dataclass(slots=True)
class Signature:
    amplitude: float
    elevation: float
    support: int


@dataclass(slots=True)
class PacketIdentity:
    alias: dict = field(default_factory=dict)
    archive: dict = field(default_factory=dict)


def nearby(a, b):
    az_err = (a.azimuth - b.azimuth + .5) % 1. - .5
    return abs(a.time - b.time) <= TIME_RADIUS and abs(a.distance - b.distance) < .024 and abs(az_err) < .10


def reliable_signature(first, second, circle, max_gap=MAX_LAST_PAIR_GAP):
    """Fit stable, signed differences; every sample comes from an earlier cycle."""
    common = sorted(first.keys() & second.keys())
    supported = [(scan, first[scan].amp - second[scan].amp, first[scan].elevation - second[scan].elevation)
                 for scan in common if circle - HISTORY_SCANS <= scan < circle and nearby(first[scan], second[scan])]
    if len(supported) < MIN_COOCCURRENCES:
        return None
    scans, amp, el = np.asarray(supported).T
    center = np.median(amp)
    if abs(center) < .0035:
        return None
    stable = abs(amp - center) <= AMP_TOLERANCE
    count = int(stable.sum())
    if count < MIN_COOCCURRENCES or count < .8 * len(amp) or circle - scans[stable][-1] > max_gap:
        return None
    scans, values = scans[stable], el[stable]
    offsets = scans - scans.mean()
    slope = float(offsets @ values / (offsets @ offsets))
    intercept = values.mean()
    residuals = values - (intercept + slope * offsets)
    if np.sqrt(np.mean(residuals**2)) > .0005 or abs(slope) > .003:
        return None
    estimate = intercept + slope * (circle - scans.mean())
    return Signature(float(center), float(estimate), count)


def supports_swap(reference, first, second):
    amp = first.amp - second.amp
    el = first.elevation - second.elevation
    return (abs(amp + reference.amplitude) <= AMP_TOLERANCE
            and abs(amp - reference.amplitude) >= 5 * AMP_TOLERANCE
            and abs(el + reference.elevation) <= EL_TOLERANCE)


class PublicPairGuard:
    """A bijection of raw IDs into public IDs, changed only under strong pair evidence."""

    def __init__(self):
        self.statistics = Counter()

    def correct(self, points, emitted, state, circle):
        if not emitted:
            return emitted
        public = state.setdefault("public_pair_identity", PacketIdentity())
        by_point = {p["PntID"]: p for p in points}
        records = {item["BatchID"]: item for item in emitted}
        if len(records) != len(emitted):
            raise ValueError("public pair identity requires one point per underlying ID per circle")
        samples = {
            raw: Sample(item["Time"], by_point[item["PntID"]]["Amp"], item["Elevation"],
                        item["Azimuth"], item["Distance"])
            for raw, item in records.items()
        }
        view = {raw: public.alias.get(raw, raw) for raw in samples}
        others = sorted(samples, key=lambda raw: (samples[raw].time, raw))
        proposals = []
        for i, left in enumerate(others):
            a = samples[left]
            for right in others[i + 1:]:
                b = samples[right]
                if b.time - a.time > TIME_RADIUS:
                    break
                if not nearby(a, b):
                    continue
                self.statistics["current_near_simultaneous_pairs"] += 1
                if view[left] not in public.archive or view[right] not in public.archive:
                    continue
                reference = reliable_signature(public.archive[view[left]], public.archive[view[right]], circle)
                if reference is None:
                    continue
                self.statistics["supported_current_pairs"] += 1
                if supports_swap(reference, a, b):
                    proposals.append((left, right))
        consistency = Counter(raw for pair in proposals for raw in pair)
        for left, right in proposals:
            if consistency[left] != 1 or consistency[right] != 1:
                self.statistics["refused_conflicting_swaps"] += 1
                continue
            pair = (view[right], view[left])
            view[left], view[right] = pair
            public.alias[left], public.alias[right] = pair
            self.statistics["public_pair_swaps"] += 1
        for raw, measurement in samples.items():
            identity = view[raw]
            timeline = public.archive.setdefault(identity, {})
            timeline[circle] = measurement
            for expired in [scan for scan in timeline if scan < circle - HISTORY_SCANS]:
                del timeline[expired]
        for identity in [key for key, timeline in public.archive.items() if circle - max(timeline) > HISTORY_SCANS]:
            del public.archive[identity]
        return [item | {"BatchID": view[item["BatchID"]]} for item in emitted]


def run(spec):
    path, with_control = spec
    return compare(path, reference_tracker().track, load_quality_reference().process_circle, with_control)


def compare(path, tracking, filtering, with_control=True):
    guard = PublicPairGuard()
    raw_outputs = {}
    raw_metrics = {}
    overhead = []

    def process(points, state):
        rows = filtering(points, state)
        if with_control:
            raw_outputs.update((row["PntID"], row["BatchID"]) for row in rows)
        started = perf_counter()
        result = guard.correct(points, rows, state, state["circle"])
        overhead.append((perf_counter() - started) * 1000)
        return result

    def score(labels, predictions, circles, stamps):
        if with_control:
            raw_metrics.update(evaluate_packet(labels, raw_outputs, circles, stamps))
        return evaluate_packet(labels, predictions, circles, stamps)

    with (patch.object(replay, "evaluate_packet", side_effect=score),
          patch("tracker_compat.baseline_circle", tracking),
          patch.object(replay, "load_handler", return_value=process)):
        guarded = replay.replay_packet(path, allowed=TUNING_PACKETS)
    control = None
    if with_control:
        times = [full - added for full, added in zip(guarded["_circle_times"], overhead, strict=True)]
        slowest = max(range(len(times)), key=times.__getitem__)
        control = guarded | raw_metrics | {
            "processing_ms": sum(times), "p99_circle_ms": replay._p99(times),
            "max_circle_ms": times[slowest], "slowest_circle": slowest, "_circle_times": times,
        }
    print(path.name, "same-point control", raw_metrics.get("score"),
          "guarded", round(guarded["score"]["total"], 5), guarded["id_switches"],
          dict(guard.statistics), flush=True)
    return control, guarded, guard.statistics, overhead


def summarize(output, packet_ids, assets, with_control=True):
    overhead = [value for _, _, _, timings in output for value in timings]
    report = {
        "packet_ids": packet_ids, "score_version": replay.SCORE_VERSION,
        "assets_sha256": {key: replay.file_sha256(path) for key, path in assets.items()},
        "guard_p99_ms": replay._p99(overhead), "guard_max_ms": max(overhead), "variants": {},
    }
    variants = ("control", "paired_public_identity") if with_control else ("paired_public_identity",)
    for variant in variants:
        index = 0 if variant == "control" else 1
        selected = [row[index] for row in output]
        diagnostics = Counter() if variant == "control" else sum((row[2] for row in output), Counter())
        stats = aggregate_packets(selected)
        print(variant, stats["score"]["total"], stats["id_switches"], stats["split_truth_tracks"],
              stats["merged_truth_points"], flush=True)
        report["variants"][variant] = {
            "summary": stats, "diagnostics": diagnostics,
            "by_packet": [{k: v for k, v in s.items() if k not in ("tracks", "_circle_times")} for s in selected],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--with-control", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.suffix.lower() != ".json":
        parser.error("--report must be JSON")
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        output = list(pool.map(run, [(path, args.with_control) for path in paths]))
    report = summarize(output, args.packets, ASSETS, args.with_control)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
