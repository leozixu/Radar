"""Estimate each scan's common elevation residual from azimuth/range anchors."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from teacher_assoc import Online, fit_predict, history_arrays
from baseline_cv import residual
from identity_rejoin import load_quality_reference
from metrics import aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, iter_circle_payloads, encode_tracks


def estimate_offset(points, state, max_gap, min_anchor, previous_offset):
    if not points:
        return previous_offset, 0
    circle = points[0]["CircleID"]
    tracks = [t for t in state.get("tracks", ([],))[0] if t.hits >= 5 and 0 < circle - t.circle <= max_gap]
    if len(tracks) < min_anchor or len(points) < min_anchor:
        return previous_offset, 0
    observed = np.array([(p["Azimuth"], p["Elevation"], p["Distance"]) for p in points])
    amplitude = np.array([p["Amp"] for p in points])
    history, times, _, present = history_arrays(tracks)
    velocity = np.array([track.velocity for track in tracks])
    fit, _, _ = fit_predict(history[:, -5:], times[:, -5:], present[:, -5:], circle, velocity)
    last = np.asarray([t.position for t in tracks]) + np.asarray([circle - t.circle for t in tracks])[:, None] * velocity
    expected = fit + .7 * residual(last, fit)
    expected[:, 1] = fit[:, 1]
    dr = residual(observed[None], expected[:, None])
    cost = np.sum((dr[:, :, [0, 2]] / np.array([.006, .0019])) ** 2, axis=2)
    mean_amp = np.array([track.mean_amplitude for track in tracks])
    cost += 3 * np.abs(amplitude[None] - mean_amp[:, None])
    cost[:, amplitude < .20] += 2
    columns = np.argmin(cost, axis=1)
    rows = np.argmin(cost, axis=0)
    candidates = np.flatnonzero(rows[columns] == np.arange(len(tracks)))
    if not len(candidates):
        return previous_offset, 0
    nearest = cost[candidates, columns[candidates]]
    next_best_by_row = np.partition(cost, kth=1, axis=1)[candidates, 1]
    next_best_by_col = np.partition(cost, kth=1, axis=0)[1, columns[candidates]]
    sure = (nearest < 4.5) & (next_best_by_row - nearest > .35) & (next_best_by_col - nearest > .12)
    anchor = candidates[sure]
    offsets = dr[anchor, columns[anchor], 1]
    if len(offsets) < min_anchor:
        return previous_offset, len(offsets)
    center = np.median(offsets)
    clean = np.abs(offsets - center) < .015
    if np.count_nonzero(clean) >= min_anchor:
        center = np.median(offsets[clean])
    return float(center), len(offsets)


class ElevationRuntime:
    def __init__(self, upstream, max_gap, min_anchor):
        self.upstream = upstream
        self.max_gap = max_gap
        self.min_anchor = min_anchor
        self.anchors = []

    def process_circle(self, points, state):
        previous = state.get("common_el_offset", 0.)
        offset, anchors = estimate_offset(points, state, self.max_gap, self.min_anchor, previous)
        state["common_el_offset"] = offset
        self.anchors.append(anchors)
        by_id = {p["PntID"]: p["Elevation"] for p in points}
        corrected = [p | {"Elevation": p["Elevation"] - offset} for p in points] if offset else points
        results = self.upstream.process_circle(corrected, state)
        return [item | {"Elevation": by_id[item["PntID"]]} for item in results]


def evaluate(path, handler):
    state, labels, circles, times, outputs = {}, {}, {}, {}, {}
    for _, payload in iter_circle_payloads(path, labels, circles, allowed=TUNING_PACKETS):
        points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
        result = handler(points, state)
        encode_tracks(result, {(p["Time"], p["CircleID"], p["PntID"]): (p["Azimuth"], p["Elevation"], p["Distance"])
                               for p in points})
        times.update((p["PntID"], p["Time"]) for p in points)
        outputs.update((item["PntID"], item["BatchID"]) for item in result)
    return evaluate_packet(labels, outputs, circles, times)


def run(args):
    path, weights, mode = args
    from lightgbm import Booster

    tracker = Online(Booster(model_file=str(weights)), floor=.23, mix=.20, coast=5, ambiguity=0.)
    quality = load_quality_reference()
    with patch("tracker_compat.baseline_circle", tracker.track):
        if mode == "teacher":
            result = evaluate(path, quality.process_circle)
        else:
            wrapper = ElevationRuntime(quality, max_gap=1, min_anchor=4)
            result = evaluate(path, wrapper.process_circle)
            result["median_anchors"] = float(np.median(wrapper.anchors))
    result["file"] = path.name
    print(mode, path.name, result["score"]["total"], "IDF1", result["idf1"],
          "switch", result["id_switches"], "merge", result["merged_truth_points"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    combinations = [(path, args.weights, mode) for mode in ("teacher", "corrected") for path in paths]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, combinations))
    grouped = {}
    for mode, group in (("teacher", results[:len(paths)]), ("corrected", results[len(paths):])):
        grouped[mode] = dict(summary=aggregate_packets(group), by_packet=[{k:v for k,v in item.items() if k != "tracks"} for item in group])
    args.report.write_text(json.dumps(dict(packet_ids=args.packets, variants=grouped), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
