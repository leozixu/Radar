"""Rescue isolated matches using stable elevation differences to reliable anchors."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__:
    from . import teacher_assoc as teacher
else:
    import teacher_assoc as teacher
from algorithm import load_runtime
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths, validate_packet_path
from replay_validation import file_sha256, replay_packet
import replay_validation as replay
from tracker_assoc import assign
from train_tracker import read_packet


WINDOW = 10
HISTORY = np.arange(-WINDOW, 0, dtype=float)


def fit_relative_elevation(history, anchors):
    """Fit prior elevation differences on common scans, then extrapolate one step."""
    differences = history[:, None, :] - anchors[None, :, :]
    mask = np.isfinite(differences)
    delta = np.where(mask, differences, 0.)
    count = mask.sum(axis=2)
    x_sum = (mask * HISTORY).sum(axis=2)
    x2_sum = (mask * HISTORY**2).sum(axis=2)
    y_sum = delta.sum(axis=2)
    xy_sum = (delta * HISTORY).sum(axis=2)
    denominator = np.maximum(count * x2_sum - x_sum**2, 1.)
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    projected = (y_sum - slope * x_sum) / np.maximum(count, 1)
    residual = np.where(mask, delta - projected[..., None] - slope[..., None] * HISTORY, 0.)
    spread = np.sqrt((residual**2).sum(axis=2) / np.maximum(count - 2, 1))
    recent = mask[..., -3:].any(axis=2)
    return projected, spread, count, recent


def recent_elevation(tracks, circle):
    history = np.full((len(tracks), WINDOW), np.nan)
    for row, track in enumerate(tracks):
        for scan, _, observation in track.history:
            slot = WINDOW + scan - circle
            if 0 <= slot < WINDOW:
                history[row, slot] = observation[1]
    return history


def reliable_anchors(rows, cols, probabilities, selected, min_probability, margin, n_tracks, n_points):
    """A current match must beat other points for its row and other rows for its point."""
    assigned_row = np.full(n_tracks, -1)
    assigned_col = np.full(n_points, -1)
    assigned_row[rows[selected]] = selected
    assigned_col[cols[selected]] = selected
    edge = np.arange(len(rows))
    row_alternatives = assigned_row[rows] != edge
    col_alternatives = assigned_col[cols] != edge
    row_next = np.zeros(n_tracks)
    col_next = np.zeros(n_points)
    np.maximum.at(row_next, rows[row_alternatives], probabilities[row_alternatives])
    np.maximum.at(col_next, cols[col_alternatives], probabilities[col_alternatives])
    confidence = probabilities[selected]
    alternatives = np.maximum(row_next[rows[selected]], col_next[cols[selected]])
    return selected[(confidence >= min_probability) & (confidence - alternatives >= margin)]


class RelativeElevationRepair:
    """Preserve anchors and occupied columns while rescuing unmatched tracks or replacing isolated outliers."""

    def __init__(self, anchor_probability=.80, anchor_margin=.12, spread=.0005, tolerance=.0009,
                 edge_floor=.08, single_strict=True):
        self.anchor_probability = anchor_probability
        self.anchor_margin = anchor_margin
        self.spread = spread
        self.tolerance = tolerance
        self.edge_floor = edge_floor
        self.single_strict = single_strict
        self.diagnostics = Counter()

    def repair(self, tracks, positions, circle, rows, cols, probabilities, selected):
        selected = np.asarray(selected, dtype=int)
        if not len(selected):
            return selected
        n_tracks, n_points = len(tracks), len(positions)
        anchors = reliable_anchors(rows, cols, probabilities, selected, self.anchor_probability,
                                   self.anchor_margin, n_tracks, n_points)
        if not len(anchors):
            return selected
        locked = np.zeros(n_tracks, dtype=bool)
        locked[rows[anchors]] = True
        occupied = np.zeros(n_points, dtype=bool)
        occupied[cols[selected]] = True
        free_edges = np.flatnonzero(~occupied[cols] & ~locked[rows] & (probabilities >= self.edge_floor))
        if not len(free_edges):
            return selected

        available_rows, inverse = np.unique(rows[free_edges], return_inverse=True)
        history = recent_elevation(tracks, circle)
        offset, spread, count, recent = fit_relative_elevation(history[available_rows], history[rows[anchors]])
        predicted = positions[cols[anchors], 1][None, :] + offset
        valid = (count >= 5) & recent & (spread <= self.spread)
        order = np.argsort(np.where(valid, spread, np.inf), axis=1, kind="stable")[:, :3]
        valid = np.take_along_axis(valid, order, axis=1)
        predictions = np.take_along_axis(predicted, order, axis=1)
        support = valid.sum(axis=1)
        nearest_spread = np.take_along_axis(spread, order[:, :1], axis=1)[:, 0]
        nearest_count = np.take_along_axis(count, order[:, :1], axis=1)[:, 0]
        coherent = (np.where(valid, predictions, -np.inf).max(axis=1)
                    - np.where(valid, predictions, np.inf).min(axis=1)) <= self.tolerance
        supported = (support >= 1) & coherent
        if self.single_strict:
            supported &= (support > 1) | ((nearest_spread <= self.spread * .6) & (nearest_count >= 6))
        if not supported.any():
            return selected
        forecast = np.where(support >= 2, np.where(valid, predictions, 0.).sum(axis=1) / np.maximum(1, support),
                            predictions[:, 0])
        self.diagnostics["supported_rows"] += int(supported.sum())

        old_edge = np.full(n_tracks, -1)
        old_edge[rows[selected]] = selected
        previous = old_edge[available_rows]
        previous_el = positions[cols[np.maximum(previous, 0)], 1]
        remove_outliers = (previous < 0) | (np.abs(previous_el - forecast) >= 3. * self.tolerance)
        elevation_error = np.abs(positions[cols[free_edges], 1] - forecast[inverse])
        viable = supported[inverse] & remove_outliers[inverse] & (elevation_error <= self.tolerance)
        candidates = free_edges[viable]
        if not len(candidates):
            return selected

        # A narrow relative-elevation band can span multiple objects with almost
        # identical elevation. Leave such a row to the original assignment.
        degree = np.bincount(rows[candidates], minlength=n_tracks)
        error = elevation_error[viable][degree[rows[candidates]] == 1]
        candidates = candidates[degree[rows[candidates]] == 1]
        if not len(candidates):
            return selected
        order = np.lexsort((candidates, error / self.tolerance - .25 * probabilities[candidates]))
        replaced = np.zeros(n_tracks, dtype=bool)
        recovered = []
        for edge in candidates[order]:
            row, col = rows[edge], cols[edge]
            if occupied[col]:
                continue
            occupied[col] = True
            replaced[row] = True
            recovered.append(edge)
            self.diagnostics["replaced" if old_edge[row] >= 0 else "recovered_misses"] += 1
        return np.r_[selected[~replaced[rows[selected]]], recovered].astype(int)

    def associate(self, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity):
        if not tracks or not len(positions):
            return [], np.empty(0, dtype=int)
        rows, cols, features = teacher.candidate_features(tracks, positions, amplitudes, circle)
        if not len(rows):
            return [], np.empty(0, dtype=int)
        probability = model.predict(features, num_threads=1)
        utility = np.log(np.clip(probability, 1e-6, 1.)) - np.log(floor) - mix * features[:, 10]
        if ambiguity > 0:
            chosen, competing = assign(rows, cols, utility, np.ones(len(tracks), dtype=bool),
                                        len(tracks), len(positions), ambiguity)
            held = {int(cols[edge]) for candidates in competing.values() for edge in candidates}
        else:
            cost = np.full((len(tracks), len(positions) + len(tracks)), np.inf)
            cost[rows, cols] = -utility
            cost[np.arange(len(tracks)), len(positions) + np.arange(len(tracks))] = 0.
            rr, cc = linear_sum_assignment(cost)
            indexed = np.full((len(tracks), len(positions)), -1, dtype=int)
            indexed[rows, cols] = np.arange(len(rows))
            fits = cc < len(positions)
            chosen, held = indexed[rr[fits], cc[fits]], set()

        repaired = self.repair(tracks, positions, circle, rows, cols, probability, chosen)
        original = set(chosen)
        matches = []
        for edge in repaired:
            column = int(cols[edge])
            held.discard(column)
            confidence = probability[edge] if edge in original else max(probability[edge], .60)
            matches.append((int(rows[edge]), column, float(confidence)))
        return matches, np.fromiter(sorted(held), dtype=int)

    def after_pairs(self, pair_module, pair_solver, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity):
        """Study composition with the existing reciprocal solver without re-scoring the same edges."""
        edges = probabilities = None
        original_features = pair_module.candidate_features

        def remember_edges(*args):
            nonlocal edges
            edges = original_features(*args)
            return edges

        def remember_probabilities(*args, **kwargs):
            nonlocal probabilities
            probabilities = model.predict(*args, **kwargs)
            return probabilities

        capture = SimpleNamespace(predict=remember_probabilities)
        with patch.object(pair_module, "candidate_features", remember_edges):
            matched, held = pair_solver.associate(tracks, positions, amplitudes, circle, capture, floor, mix, ambiguity)
        if not matched:
            return matched, held
        rows, cols, _ = edges
        edge_index = {(row, col): i for i, (row, col) in enumerate(zip(rows, cols))}
        confidence = {edge_index[row, col]: q for row, col, q in matched}
        revised = self.repair(tracks, positions, circle, rows, cols, probabilities, list(confidence))
        out = [(int(rows[edge]), int(cols[edge]), float(confidence.get(edge, max(probabilities[edge], .60))))
               for edge in revised]
        used = {col for _, col, _ in out}
        return out, np.fromiter((col for col in held if col not in used), dtype=int)


def oracle_packet(path, max_spread, tolerance):
    """Measure one-step elevation extrapolation on known observations only."""
    data = read_packet(validate_packet_path(path, TRAIN_PACKETS))
    good = data[:, 8] > 0
    tracks, circles = data[good, 8].astype(int), data[good, 6].astype(int)
    history = np.full((tracks.max() + 1, circles.max() + 1), np.nan)
    history[tracks, circles] = data[good, 2].astype(np.float32)
    all_errors, selected_errors = [], []
    targets = supported = 0
    for circle in range(WINDOW, history.shape[1]):
        present = np.flatnonzero(np.isfinite(history[:, circle]))
        if len(present) < 2:
            continue
        past = history[present, circle-WINDOW:circle]
        current = history[present, circle]
        prediction, spread, count, recency = fit_relative_elevation(past, past)
        valid = (count >= 5) & recency & ~np.eye(len(present), dtype=bool)
        absolute_error = np.abs(current[:, None] - current[None, :] - prediction)
        selected = valid & (spread <= max_spread)
        all_errors.append(absolute_error[valid])
        selected_errors.append(absolute_error[selected])
        targets += int(valid.any(axis=1).sum())
        supported += int(selected.any(axis=1).sum())

    def describe(errors):
        all_values = np.concatenate(errors) if errors else np.empty(0)
        return {"pairs": len(all_values), "within_tolerance": int((all_values <= tolerance).sum()),
                "absolute_error_quantiles": dict(zip(("p50", "p90", "p95", "p99"),
                                                    np.quantile(all_values, (.5, .9, .95, .99))))
                if len(all_values) else {}}

    return {"all": describe(all_errors), "stable": describe(selected_errors),
            "track_observations": targets, "supported_observations": supported}


def _test_packet(config, path):
    from lightgbm import Booster

    mature = teacher.Online(Booster(model_file=str(config.teacher)), .20, 0., 7, .30, 4.)
    policy = RelativeElevationRepair(config.anchor_probability, config.anchor_margin, config.spread,
                                     config.tolerance, config.edge_floor, not config.relax_single)
    context = load_runtime()
    associate = policy.associate
    if config.pair:
        import pair_swap_probe as pairs

        pair_solver = pairs.PairCorrector(Booster(model_file=str(config.pair)), .85)
        associate = partial(policy.after_pairs, pairs, pair_solver)
    with (patch.object(teacher, "assign_confirmed", associate),
          patch.object(teacher.compat, "baseline_circle", mature.track),
          patch.object(replay, "load_handler", return_value=context.process_circle)):
        stats = replay_packet(path, allowed=TUNING_PACKETS)
    stats.pop("_circle_times", None)
    print(path.name, round(stats["score"]["total"], 4), stats["id_switches"], dict(policy.diagnostics), flush=True)
    return stats, policy.diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "tune"))
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--pair", type=Path, help="Apply one-sided repair after the existing pair solver")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--anchor-probability", type=float, default=.80)
    parser.add_argument("--anchor-margin", type=float, default=.12)
    parser.add_argument("--spread", type=float, default=.0005)
    parser.add_argument("--tolerance", type=float, default=.0009)
    parser.add_argument("--edge-floor", type=float, default=.08)
    parser.add_argument("--relax-single", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    allowed = TRAIN_PACKETS if args.phase == "probe" else TUNING_PACKETS
    try:
        paths = packet_paths(DEFAULT_DATA_DIR, args.packets, allowed)
    except ValueError as error:
        parser.error(str(error))
    metadata = {"packet_ids": args.packets, "source_sha256": file_sha256(Path(__file__)),
                "settings": {name: getattr(args, name) for name in (
                    "anchor_probability", "anchor_margin", "spread", "tolerance", "edge_floor", "relax_single",
                )}}
    report = {"metadata": metadata}
    if args.phase == "probe":
        report["packets"] = {path.name: oracle_packet(path, args.spread, args.tolerance) for path in paths}
        print(json.dumps(report["packets"], indent=2), flush=True)
    else:
        if args.teacher is None:
            parser.error("--teacher is required for tune")
        metadata["teacher_sha256"] = file_sha256(args.teacher)
        if args.pair:
            metadata["pair_sha256"] = file_sha256(args.pair)
            metadata["pair_cutoff"] = .85
        runner = partial(_test_packet, args)
        if args.jobs == 1:
            results = list(map(runner, paths))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                results = list(pool.map(runner, paths))
        stats, diagnostics = zip(*results)
        report["summary"] = summary = aggregate_packets(stats)
        report["diagnostics"] = sum(diagnostics, Counter())
        report["packets"] = [{k: v for k, v in item.items() if k != "tracks"} for item in stats]
        print(json.dumps({k: summary[k] for k in ("score", "idf1", "false_alarm_ratio", "id_switches",
                                                  "split_truth_tracks", "merged_output_tracks")}))
        print(dict(report["diagnostics"]))
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
