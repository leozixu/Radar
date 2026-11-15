"""Learn which current points belong to a stable elevation pair, including missing points."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import causal_birth_paths as birth
import joint_elevation_pairs as joint
import pair_swap_probe as pairs
import teacher_assoc as teacher
from algorithm import load_runtime
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths, validate_packet_path
import replay_validation as replay
import tracker_compat
from train_tracker import read_packet


FEATURES = tuple(f"{tag}_{name}" for tag in ("low", "high") for name in teacher.FEATURES) + (
    "present_count", "low_probability", "high_probability", "small_edge_count", "large_edge_count",
    "weak_best_probability", "strong_best_probability", "relative_elevation_error",
    "best_compatible_alternative", "best_unobserved_role_edge", "current_azimuth_separation",
    "current_range_separation", "common_history_count", "relative_elevation_spread",
    "relative_amplitude_error", "relative_amplitude_spread", "best_joint_compatible_alternative",
)
DEFAULT_TEACHER = Path("experimental/teacher_assoc_wide.txt")
DEFAULT_PAIR = Path("experimental/models/pair_swap_wide.txt")
DEFAULT_BIRTH = Path("experimental/models/causal_birth_paths.txt")
DEFAULT_CLOCK = Path("experimental/models/scan_clock.txt")
POINT_FLOOR = .02
EL_ERROR = FEATURES.index("relative_elevation_error")
AMP_ERROR = FEATURES.index("relative_amplitude_error")
MISSING_Q = FEATURES.index("best_unobserved_role_edge")


@dataclass(slots=True)
class Choices:
    row_a: int
    row_b: int
    combinations: np.ndarray
    features: np.ndarray
    own_edges_a: np.ndarray
    own_edges_b: np.ndarray


def features_for_group(a, b, first, second, offset, common, spread, amp_offset, amp_spread, cols,
                       probabilities, features, positions, amplitudes):
    present = (a >= 0, b >= 0)
    aa, bb = np.maximum(a, 0), np.maximum(b, 0)
    fa = np.where(present[0][:, None], features[aa], np.nan)
    fb = np.where(present[1][:, None], features[bb], np.nan)
    qa = np.where(present[0], probabilities[aa], 0.)
    qb = np.where(present[1], probabilities[bb], 0.)
    best_a = float(probabilities[first].max(initial=0.))
    best_b = float(probabilities[second].max(initial=0.))
    error = abs(positions[cols[aa], 1] - positions[cols[bb], 1] - offset)
    amplitude_error = abs(amplitudes[cols[aa]] - amplitudes[cols[bb]] - amp_offset)
    separation = abs(teacher.cv.residual(positions[cols[aa]], positions[cols[bb]]))
    both = present[0] & present[1]

    # In a one-point option, report whether the missing role actually has an
    # alternative that agrees with the observed member's relative elevation.
    alternative = np.full(len(a), np.nan)
    joint_alternative = np.full(len(a), np.nan)
    for observed, masked, others, sign in ((a, ~present[1], second, 1), (b, ~present[0], first, -1)):
        if not masked.any():
            continue
        estimate = positions[cols[observed[masked]], 1] - sign * offset
        candidate_columns = cols[others]
        compatible = (abs(estimate[:, None] - positions[candidate_columns, 1]) <= .0018)
        compatible &= cols[observed[masked], None] != candidate_columns
        alternative[masked] = np.max(np.where(compatible, probabilities[others], 0.), axis=1, initial=0.)
        amp_estimate = amplitudes[cols[observed[masked]]] - sign * amp_offset
        amp_compatible = abs(amp_estimate[:, None] - amplitudes[candidate_columns]) <= .0003
        joint_alternative[masked] = np.max(np.where(compatible & amp_compatible, probabilities[others], 0.),
                                           axis=1, initial=0.)
    unobserved = np.where(~present[0], best_a, np.where(~present[1], best_b, np.nan))
    return np.column_stack((
        np.fmin(fa, fb), np.fmax(fa, fb), present[0].astype(int) + present[1],
        np.minimum(qa, qb), np.maximum(qa, qb), np.full(len(a), min(len(first), len(second))),
        np.full(len(a), max(len(first), len(second))), np.full(len(a), min(best_a, best_b)),
        np.full(len(a), max(best_a, best_b)), np.where(both, error, np.nan), alternative, unobserved,
        np.where(both, separation[:, 0], np.nan), np.where(both, separation[:, 2], np.nan),
        np.full(len(a), common), np.full(len(a), spread),
        np.where(both, amplitude_error, np.nan), np.full(len(a), amp_spread), joint_alternative,
    )).astype(np.float32)


def enumerate_choices(tracks, positions, amplitudes, circle, rows, cols, x, probabilities, stable, owners,
                      *, informative=None):
    if not len(rows) or not stable:
        return []
    by_row = np.lexsort((-probabilities, rows))
    end = np.cumsum(np.bincount(rows, minlength=len(tracks)))
    start = np.r_[0, end[:-1]]
    past = joint.recent_elevation(tracks, circle)
    amps_past = np.full_like(past, np.nan)
    for row, track in enumerate(tracks):
        for scan, amplitude, _ in track.history:
            slot = past.shape[1] + scan - circle
            if 0 <= slot < past.shape[1]:
                amps_past[row, slot] = amplitude
    options = []
    for ta, tb, offset in stable:
        if informative is not None and (ta, tb) not in informative:
            continue
        edge_lists = []
        for row in (ta, tb):
            candidates = by_row[start[row]:end[row]]
            owner = owners[cols[candidates]]
            admissible = (owner < 0) | (owner == ta) | (owner == tb)
            edge_lists.append(candidates[admissible & (probabilities[candidates] >= POINT_FLOOR)][:7])
        first, second = edge_lists
        if not len(first) and not len(second):
            continue
        aa = np.repeat(first, len(second))
        bb = np.tile(second, len(first))
        # Include current wrong-target/clutter alternatives near the predicted
        # relationship and exclude only physically remote options.
        error = abs(positions[cols[aa], 1] - positions[cols[bb], 1] - offset)
        possible = (cols[aa] != cols[bb]) & (error <= .018)
        a = np.r_[aa[possible], first, np.full(len(second), -1, dtype=int)]
        b = np.r_[bb[possible], np.full(len(first), -1, dtype=int), second]

        history_a, history_b = past[ta:ta+1], past[tb:tb+1]
        _, deviation, count, _ = joint.fit_relative_elevation(history_a, history_b)
        amp_forecast, amp_deviation, _, _ = joint.fit_relative_elevation(amps_past[ta:ta+1], amps_past[tb:tb+1])
        feature = features_for_group(a, b, first, second, offset, count[0, 0], deviation[0, 0],
                                     amp_forecast[0, 0], amp_deviation[0, 0], cols, probabilities, x, positions,
                                     amplitudes)
        options.append(Choices(ta, tb, np.column_stack((a, b)), feature, first, second))
    return options


def label_group(choices, owner, truth, cols, probabilities, rng):
    a, b = choices.combinations.T
    truth_a, truth_b = owner[choices.row_a], owner[choices.row_b]
    left, right = truth[cols[np.maximum(0, a)]], truth[cols[np.maximum(0, b)]]
    seen_a = (truth[cols[choices.own_edges_a]] == truth_a).any()
    seen_b = (truth[cols[choices.own_edges_b]] == truth_b).any()
    correct_a = np.where(a >= 0, left == truth_a, not seen_a)
    correct_b = np.where(b >= 0, right == truth_b, not seen_b)
    known = ((a < 0) | (left >= 0)) & ((b < 0) | (right >= 0))
    positive = correct_a & correct_b & known
    negative = known & ~positive
    both = (a >= 0) & (b >= 0)
    single = ~both
    clutter = ((a >= 0) & (left == 0)) | ((b >= 0) & (right == 0))
    current_probability = (np.where(a >= 0, probabilities[np.maximum(a, 0)], 1.)
                           * np.where(b >= 0, probabilities[np.maximum(b, 0)], 1.))
    selected = positive.copy()
    near = choices.features[:, EL_ERROR] + np.minimum(choices.features[:, AMP_ERROR], .015) * .5
    for subset in (both & ~clutter, both & clutter, single & ~clutter, single & clutter):
        indices = np.flatnonzero(negative & subset)
        if not len(indices):
            continue
        risk = current_probability[indices] + rng.random(len(indices)) * .025
        selected[indices[np.argsort(-risk, kind="stable")[:2]]] = True
        if both[indices[0]]:
            selected[indices[np.argmin(near[indices])]] = True
    chosen = np.flatnonzero(selected)
    return choices.features[chosen], positive[chosen], {
        "positive_both": int((positive & both).sum()), "positive_single": int((positive & single).sum()),
        "negative_with_clutter": int((selected & negative & clutter).sum()),
        "negative_without_clutter": int((selected & negative & ~clutter).sum()),
    }


def collect(teacher_path, path):
    from lightgbm import Booster

    data = read_packet(validate_packet_path(path, TRAIN_PACKETS))
    model = Booster(model_file=str(teacher_path))
    previous = {}
    blocks, labels = [], []
    diagnostics = Counter()
    rng = np.random.default_rng(int(path.stem[-3:]) + 7201)
    edges = np.r_[0, np.flatnonzero(np.diff(data[:, 6])) + 1, len(data)]
    for begin, end in zip(edges, edges[1:]):
        scan = data[begin:end]
        circle = int(scan[0, 6])
        positions = scan[:, 1:4].astype(np.float32).astype(float)
        amplitudes = scan[:, 4].astype(np.float32).astype(float)
        truth = scan[:, 8].astype(np.int64)
        active = [(key, track) for key, track in previous.items() if 0 < circle - track.circle <= 7]
        if len(active) >= 2:
            identities, tracks = zip(*active)
            stable = joint.stable_pairs(tracks, circle, .0005, np.ones(len(tracks), dtype=bool))
            if stable:
                rows, cols, x = teacher.candidate_features(tracks, positions, amplitudes, circle)
                if len(rows):
                    q = model.predict(x, num_threads=1)
                    lock, _ = joint.strong_columns(rows, cols, q, len(tracks), len(positions))
                    options = enumerate_choices(tracks, positions, amplitudes, circle, rows, cols, x, q, stable, lock)
                    for group in options:
                        feature, y, counts = label_group(group, identities, truth, cols, q, rng)
                        blocks.append(feature)
                        labels.append(y)
                        diagnostics.update(counts)
        for col in np.flatnonzero(truth > 0):
            key = truth[col]
            track = previous.get(key)
            if track is None or not 0 < circle - track.circle <= 7:
                previous[key] = teacher.spawn(positions[col], amplitudes[col], circle)
            else:
                teacher.update(track, positions[col], amplitudes[col], circle)
    print(path.name, dict(diagnostics), flush=True)
    if not blocks:
        raise ValueError(f"{path.name} produced no stable group examples")
    return np.concatenate(blocks), np.concatenate(labels), diagnostics


class MembershipMatcher:
    def __init__(self, original, membership, confidence=.90, margin=.12):
        self.original = original
        self.membership = membership
        self.confidence = confidence
        self.margin = margin
        self.stats = Counter()

    def associate(self, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity):
        if not tracks or not len(positions):
            return [], np.empty(0, dtype=int)
        original = self.original
        feature_call = original.base_module.candidate_features
        rows, cols, features = feature_call(tracks, positions, amplitudes, circle)
        if not len(rows):
            return [], np.empty(0, dtype=int)
        q = model.predict(features, num_threads=1)
        scores = SimpleNamespace(predict=lambda *_a, **_k: q)
        utility = np.log(np.clip(q, 1e-6, 1.)) - np.log(floor) - mix * features[:, 10]
        available = np.bincount(rows[q >= original.min_probability], minlength=len(tracks)) > 0
        original_stable = joint.stable_pairs(tracks, circle, original.spread, available, original.minimum_history)
        stable = joint.stable_pairs(tracks, circle, original.spread, np.ones(len(tracks), dtype=bool),
                                    original.minimum_history)
        old_groups = {(group.first_row, group.second_row): group for group in joint.candidate_groups(
            original_stable, rows, cols, q, utility, positions, len(tracks), tolerance=original.tolerance,
            bonus=original.bonus, min_probability=original.min_probability,
        )}
        old_owner = {row: group for group in old_groups for row in group}
        owners, reliable = joint.strong_columns(rows, cols, q, len(tracks), len(positions))
        established = np.zeros(len(tracks), dtype=bool)
        for edge in reliable:
            established[rows[edge]] = True
        informative = {(a, b) for a, b, _ in stable if (not established[a] or not established[b])
                       and old_owner.get(a, (a, b)) == (a, b) and old_owner.get(b, (a, b)) == (a, b)}
        proposed = enumerate_choices(tracks, positions, amplitudes, circle, rows, cols, features, q, stable, owners,
                                     informative=informative)
        if proposed:
            prediction = self.membership.predict(np.concatenate([group.features for group in proposed]), num_threads=1)
        replacements = {}
        start = 0
        for group in proposed:
            stop = start + len(group.features)
            p = prediction[start:stop]
            start = stop
            ranking = np.argsort(-p, kind="stable")
            best = ranking[0]
            runner_up = p[ranking[1]] if len(p) > 1 else 0.
            if p[best] < self.confidence or p[best] - runner_up < self.margin:
                continue
            pair = group.row_a, group.row_b
            a, b = (int(value) for value in group.combinations[best])
            if (a >= 0 and b >= 0):
                if group.features[best, EL_ERROR] > .0018:
                    continue
                if pair in old_groups:
                    old = old_groups[pair].options[0]
                    if (a, b) == (old.first, old.second):
                        continue
            else:
                # Keep a counterpart that already has a strong individual edge.
                if p[best] < max(.94, self.confidence) or group.features[best, MISSING_Q] >= .70:
                    continue
            replacements[pair] = a, b, float(p[best] - runner_up)

        cached = rows, cols, features
        if not replacements:
            with patch.object(original.base_module, "candidate_features", return_value=cached):
                return original.associate(tracks, positions, amplitudes, circle, scores, floor, mix, ambiguity)

        real_edges = len(rows)
        augmented_rows = np.r_[rows, np.arange(len(tracks))]
        augmented_cols = np.r_[cols, len(positions) + np.arange(len(tracks))]
        augmented_utility = np.r_[utility, np.full(len(tracks), -np.inf)]
        new_options = []
        for pair, (a, b, margin) in replacements.items():
            taken = [edge for edge in (a, b) if edge >= 0]
            indices = (a if a >= 0 else real_edges + pair[0], b if b >= 0 else real_edges + pair[1])
            option = joint.PairOption(*indices, float(sum(utility[taken]) + original.bonus + .65 * margin))
            previous = tuple(old for old in old_groups[pair].options if (old.first, old.second) != indices
                             )[:2] if pair in old_groups else ()
            all_options = tuple(sorted((option, *previous), key=lambda choice: -choice.utility))
            old_groups[pair] = joint.PairGroup(*pair, all_options)
            new_options.append(indices)

        chosen = joint.select_joint_options(
            list(old_groups.values()), augmented_rows, augmented_cols, augmented_utility, len(tracks),
            len(positions) + len(tracks), original.minimum_gain, original.max_groups,
        )
        live = chosen[chosen < real_edges]
        locked_rows, locked_cols = np.zeros(len(tracks), dtype=bool), np.zeros(len(positions), dtype=bool)
        locked_rows[augmented_rows[chosen]] = True
        locked_cols[cols[live]] = True
        keep = ~locked_rows[rows] & ~locked_cols[cols]
        remaining = rows[keep], cols[keep], features[keep]
        filtered = SimpleNamespace(predict=lambda *_a, **_k: q[keep])
        with patch.object(original.base_module, "candidate_features", return_value=remaining):
            output, held = original.base.associate(tracks, positions, amplitudes, circle, filtered, floor, mix,
                                                   ambiguity)
        self.stats["proposed_membership_changes"] += len(replacements)
        self.stats["selected_missing_members"] += len(chosen) - len(live)
        chosen_edges = set(chosen.tolist())
        self.stats["selected_membership_groups"] += sum(a in chosen_edges and b in chosen_edges for a, b in new_options)
        output.extend((int(rows[edge]), int(cols[edge]), float(max(.60, q[edge]))) for edge in live)
        return output, held


def _evaluate(args, path):
    from lightgbm import Booster

    load = lambda path: Booster(model_file=str(path))
    tracker = birth.BirthOnline(load(args.teacher), load(args.pair), load(args.birth), load(args.clock), .86,
                                "supplement", paired_birth=True, joint=True, two_hit_pairs=True)
    membership = load(args.model)
    if tuple(membership.feature_name()) != FEATURES:
        raise ValueError("Membership weight vector does not match the feature schema")
    matcher = MembershipMatcher(tracker.matcher, membership, args.confidence, args.margin)
    tracker.matcher = matcher
    with (patch.object(tracker_compat, "baseline_circle", tracker.track),
          patch.object(replay, "load_handler", return_value=load_runtime().process_circle)):
        stats = replay.replay_packet(path, allowed=TUNING_PACKETS)
    stats.pop("_circle_times", None)
    print(path.name, stats["score"]["total"], stats["id_switches"], dict(matcher.stats), flush=True)
    return stats, matcher.stats


def main():
    from lightgbm import Dataset, train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "tune"))
    parser.add_argument("--packets", nargs="+", type=int, required=True)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--pair", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--birth", type=Path, default=DEFAULT_BIRTH)
    parser.add_argument("--clock", type=Path, default=DEFAULT_CLOCK)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--confidence", type=float, default=.90)
    parser.add_argument("--margin", type=float, default=.12)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    allowed = TRAIN_PACKETS if args.phase == "train" else TUNING_PACKETS
    try:
        paths = packet_paths(DEFAULT_DATA_DIR, args.packets, allowed)
    except ValueError as error:
        parser.error(str(error))
    metadata = {"packet_ids": args.packets, "source_sha256": replay.file_sha256(Path(__file__)),
                "models": {"teacher": replay.file_sha256(args.teacher)},
                "runtime_sources": {module.__name__: replay.file_sha256(Path(module.__file__))
                                    for module in (teacher, joint, pairs, birth)}}
    runner = partial(collect, args.teacher) if args.phase == "train" else partial(_evaluate, args)
    if args.jobs == 1:
        results = list(map(runner, paths))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(runner, paths))
    if args.phase == "train":
        features, labels, diagnostics = zip(*results)
        x, y = np.concatenate(features), np.concatenate(labels)
        parameters = {"objective": "binary", "learning_rate": .085, "num_leaves": 31, "min_child_samples": 45,
                      "verbosity": -1, "num_threads": args.jobs, "seed": 2175, "force_col_wise": True}
        model = train(parameters, Dataset(x, label=y, feature_name=list(FEATURES)), num_boost_round=125)
        model.save_model(str(args.model))
        report = {"metadata": metadata, "training_rows": len(y), "training_positives": int(y.sum()),
                  "diagnostics": sum(diagnostics, Counter()), "parameters": parameters}
    else:
        stats, diagnostics = zip(*results)
        summary = aggregate_packets(stats)
        metadata["models"].update({name: replay.file_sha256(getattr(args, name)) for name in ("birth", "clock", "pair")})
        metadata["settings"] = {"confidence": args.confidence, "margin": args.margin, "two_hit_pairs": True,
                                "paired_birth": True, "joint": True, "birth_floor": .86}
        report = {"metadata": metadata, "diagnostics": sum(diagnostics, Counter()), "summary": summary,
                  "packets": [{k: v for k, v in item.items() if k != "tracks"} for item in stats]}
        print(summary["score"]["total"], summary["id_switches"], summary["split_truth_tracks"], summary["idf1"],
              report["diagnostics"], flush=True)
    metadata["models"]["membership"] = replay.file_sha256(args.model)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
