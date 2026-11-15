"""Recover two-track ambiguity holds only with decisive paired motion evidence."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import pair_swap_probe as pairs
import teacher_assoc as teacher
from identity_rejoin import load_quality_reference
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay


def held_components(choices, columns, available):
    """Find connected components of the actually withheld rows and columns."""
    row_links, col_links = {}, defaultdict(set)
    for row, edges in choices.items():
        cols = {int(columns[index]) for index in edges if int(columns[index]) in available}
        if cols:
            row_links[row] = cols
            for col in cols:
                col_links[col].add(row)
    unused = set(row_links)
    while unused:
        queue = [min(unused)]
        rows, cols = set(), set()
        while queue:
            r = queue.pop()
            if r not in unused:
                continue
            unused.remove(r)
            rows.add(r)
            fresh = row_links[r] - cols
            cols.update(fresh)
            for c in fresh:
                queue.extend(col_links[c] & unused)
        yield sorted(rows), sorted(cols)


class HeldPairRecovery:
    def __init__(self, pair_model, confidence, unary_floor):
        self.reference = pairs.PairCorrector(pair_model, .85)
        self.model = pair_model
        self.confidence = confidence
        self.unary_floor = unary_floor
        self.statistics = Counter()

    def associate(self, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity):
        if self.confidence is None:
            return self.reference.associate(tracks, positions, amplitudes, circle, model, floor, mix, ambiguity)

        basis, scores, choices = None, None, None
        actual_features, actual_assign = pairs.candidate_features, pairs.assign

        def capture_features(*args):
            nonlocal basis
            basis = actual_features(*args)
            return basis

        def capture_probability(x, num_threads):
            nonlocal scores
            scores = model.predict(x, num_threads=num_threads)
            return scores

        def capture_assignment(*args):
            nonlocal choices
            selected, choices = actual_assign(*args)
            return selected, choices

        predictor = SimpleNamespace(predict=capture_probability)
        with (patch.object(pairs, "candidate_features", capture_features),
              patch.object(pairs, "assign", capture_assignment)):
            matched, held = self.reference.associate(tracks, positions, amplitudes, circle, predictor,
                                                     floor, mix, ambiguity)
        if not len(held) or basis is None or scores is None or not choices:
            return matched, held

        rows, cols, individual = basis
        reserved_rows = {row for row, _, _ in matched}
        held_set = set(map(int, held)) - {col for _, col, _ in matched}
        components = held_components({r: edges for r, edges in choices.items() if r not in reserved_rows},
                                     cols, held_set)
        index = np.full((len(tracks), len(positions)), -1, dtype=int)
        index[rows, cols] = np.arange(len(rows))
        log_p = np.log(np.clip(scores, 1.e-6, 1.))
        candidates, features = [], []
        for group_rows, group_cols in components:
            self.statistics[f"held_{len(group_rows)}x{len(group_cols)}"] += 1
            if len(group_rows) != 2 or len(group_cols) != 2:
                continue
            left, right = group_rows
            a, b = group_cols
            if min(len(tracks[left].history), len(tracks[right].history)) < 4:
                continue
            edges = index[np.ix_(group_rows, group_cols)]
            if (edges < 0).any() or (scores[edges] < floor / 2).any():
                continue
            forward, reverse, _ = pairs.make_features(tracks, positions, individual, index, log_p,
                                                       left, right, a, b, circle)
            if forward[4 * len(pairs.FEATURES) + 2] < 3:
                continue
            candidates.append(((left, right), (a, b), edges))
            features.extend((forward, reverse))
        if not candidates:
            return matched, held

        evidence = self.model.predict(np.asarray(features), num_threads=1).reshape(-1, 2)
        for (group_rows, group_cols, edges), (normal, flipped) in zip(candidates, evidence):
            confidence_normal = min(normal, 1. - flipped)
            confidence_flipped = min(flipped, 1. - normal)
            if max(confidence_normal, confidence_flipped) < self.confidence:
                continue
            orientation = int(confidence_flipped > confidence_normal)
            path = (orientation, 1 - orientation)
            support = scores[edges[[0, 1], path]]
            if np.min(support) < self.unary_floor:
                continue
            for j, r in enumerate(group_rows):
                col = group_cols[path[j]]
                matched.append((r, col, float(max(support[j], .60))))
                held_set.remove(col)
            self.statistics["released_pair_matches"] += 2
        return matched, np.asarray(sorted(held_set), dtype=int)


MODES = {"baseline": (None, 0.), "selective": (.94, .30), "strict": (.98, .45)}


def run(spec):
    from lightgbm import Booster

    path, teacher_model, pair_model, mode = spec
    quality = load_quality_reference()
    tracker = teacher.Online(Booster(model_file=str(teacher_model)), .20, mix=0., coast=7, ambiguity=.30, robust=4.)
    assignment = HeldPairRecovery(Booster(model_file=str(pair_model)), *MODES[mode])
    with (patch.object(teacher, "assign_confirmed", assignment.associate),
          patch("tracker_compat.baseline_circle", tracker.track),
          patch.object(replay, "load_handler", return_value=quality.process_circle)):
        result = replay.replay_packet(path, allowed=TUNING_PACKETS)
    print(mode, path.name, round(result["score"]["total"], 4), result["id_switches"],
          assignment.statistics["released_pair_matches"], flush=True)
    return result, assignment.statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=tuple(MODES))
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.suffix.lower() != ".json":
        parser.error("--report must be a JSON file")
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    tasks = [(path, args.teacher, args.pair, mode) for mode in args.modes for path in paths]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, tasks))
    report = {
        "packet_ids": args.packets,
        "teacher_sha256": hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        "pair_sha256": hashlib.sha256(args.pair.read_bytes()).hexdigest(),
        "variants": {},
    }
    for j, mode in enumerate(args.modes):
        cases, counters = zip(*results[j * len(paths):(j + 1) * len(paths)])
        stats = aggregate_packets(cases)
        diagnostics = sum(counters, Counter())
        report["variants"][mode] = {
            "config": MODES[mode], "summary": stats, "held": diagnostics,
            "by_packet": [{k: v for k, v in packet.items() if k not in ('tracks', '_circle_times')} for packet in cases],
        }
        print(mode, round(stats["score"]["total"], 4), stats["id_switches"], stats["split_truth_tracks"],
              stats["merged_truth_points"], dict(diagnostics), flush=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
