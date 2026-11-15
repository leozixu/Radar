"""Track simultaneous paired returns using repeatable signed amplitude offsets."""

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from causal_birth_paths import AMPLITUDE, CLOCK, POINT, POSITION, SCAN, BirthOnline, measurements, parallel
from algorithm import load_runtime
from baseline_cv import residual
from metrics import SCORE_VERSION, aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
import replay_validation as replay
from teacher_assoc import NOISE
import tracker_compat


HERE = Path(__file__).resolve().parent
ASSETS = {
    "teacher": HERE / "teacher_assoc_wide.txt",
    "pair": HERE / "models/pair_swap_wide.txt",
    "birth": HERE / "models/causal_birth_paths.txt",
    "clock": HERE / "models/scan_clock.txt",
}
MAX_PAIRS = 800
AMPLITUDE_TOLERANCE = .0000031
MAX_GAP = 12
PAIR_WIDTH = np.array((.018, .012, .008))
FEATURES = (
    "gap", "signature_difference", "signature_magnitude", "prior_clock_min", "current_clock_min",
    "prior_amplitude_mean", "amplitude_mean", "amplitude_change", "amplitude_min",
) + tuple(f"{part}_{axis}" for part in ("prior_width", "current_width", "width_change", "movement")
          for axis in ("azimuth", "elevation", "range"))


def measured_pairs(raw, times, wide=False):
    """Return spatially close measurements with distinct neighbouring radar ticks."""
    order = np.argsort(times, kind="stable")
    stamps = times[order]
    neighbours = np.diff(stamps)
    unique = np.r_[True, neighbours != 0] & np.r_[neighbours != 0, True] if len(stamps) else np.empty(0, bool)
    if wide:
        adjacent = []
        for step in range(1, min(3, len(stamps) - 1) + 1):
            elapsed = stamps[step:] - stamps[:-step]
            left = np.flatnonzero((elapsed >= 1) & (elapsed <= 3) & unique[:-step] & unique[step:])
            adjacent.append(np.column_stack((order[left], order[left + step])))
        pair = np.concatenate(adjacent) if adjacent else np.empty((0, 2), dtype=int)
    else:
        left = np.flatnonzero((neighbours == 1) & unique[:-1] & unique[1:])
        pair = np.column_stack((order[left], order[left + 1]))
    before, after = raw[pair[:, 0]], raw[pair[:, 1]]
    width = np.abs(residual(after[:, POSITION], before[:, POSITION]))
    amp_difference = np.abs(after[:, AMPLITUDE] - before[:, AMPLITUDE])
    clock = np.minimum(before[:, CLOCK], after[:, CLOCK])
    widths = np.array((.046, .065, .020)) if wide else PAIR_WIDTH
    valid = (width <= widths).all(axis=1) & (amp_difference >= .000004) & (amp_difference <= .032)
    if not wide:
        valid &= (clock >= .20) | (np.minimum(before[:, AMPLITUDE], after[:, AMPLITUDE]) >= .20)
    eligible = np.flatnonzero(valid)
    if len(eligible) > MAX_PAIRS:
        eligible = eligible[np.argsort(-clock[eligible], kind="stable")[:MAX_PAIRS]]
    return pair[eligible]


def pair_offset(observations):
    return observations[:, 1, AMPLITUDE] - observations[:, 0, AMPLITUDE]


def link_features(prior, current, tolerance=AMPLITUDE_TOLERANCE):
    """Gate by the signed amplitude difference and local motion before model inference."""
    past_offsets = pair_offset(prior)
    offsets = pair_offset(current)
    order = np.argsort(offsets, kind="stable")
    ranges = np.searchsorted(offsets[order], past_offsets[:, None] + np.array([-tolerance, tolerance]))
    counts = ranges[:, 1] - ranges[:, 0]
    rows = np.repeat(np.arange(len(prior)), counts)
    if not len(rows):
        return rows, rows.copy(), np.empty((0, len(FEATURES)), np.float32)
    starts = np.repeat(ranges[:, 0] - np.r_[0, counts.cumsum()[:-1]], counts)
    cols = order[starts + np.arange(len(rows))]
    former, latter = prior[rows], current[cols]
    gap = latter[:, 0, SCAN] - former[:, 0, SCAN]
    azimuth_range = residual(latter[:, :, POSITION], former[:, :, POSITION])
    motion = np.abs(azimuth_range.mean(axis=1))
    limit = np.array([.038, .14, .020]) * (1 + .6 * np.maximum(0, gap - 1))[:, None]
    valid = (gap > 0) & (gap <= MAX_GAP) & (motion <= limit).all(axis=1)
    rows, cols, former, latter, motion, gap = (array[valid] for array in (rows, cols, former, latter, motion, gap))
    former_delta = residual(former[:, 1, POSITION], former[:, 0, POSITION])
    latter_delta = residual(latter[:, 1, POSITION], latter[:, 0, POSITION])
    prior_amp = former[:, :, AMPLITUDE].mean(axis=1)
    amp = latter[:, :, AMPLITUDE].mean(axis=1)
    features = np.column_stack((
        gap, np.abs(past_offsets[rows] - offsets[cols]), np.abs(offsets[cols]),
        former[:, :, CLOCK].min(axis=1), latter[:, :, CLOCK].min(axis=1), prior_amp, amp,
        np.abs(prior_amp - amp), latter[:, :, AMPLITUDE].min(axis=1), np.abs(former_delta), np.abs(latter_delta),
        np.abs(residual(latter_delta, former_delta)), motion / gap[:, None],
    ))
    return rows, cols, features.astype(np.float32)


def collect(spec):
    from lightgbm import Booster

    path, clock_path = spec
    clock = Booster(model_file=str(clock_path))
    labels, circles = {}, {}
    previous = deque(maxlen=5)
    trained, targets, evidence = [], [], Counter()
    for circle, payload in replay.iter_circle_payloads(path, labels, circles, TRAIN_PACKETS):
        scan = np.asarray(list(replay.POINT.iter_unpack(payload)), dtype=float).reshape(-1, len(replay.POINT_KEYS))
        raw = measurements(scan[:, 1:4], scan[:, 4], scan[:, 0], scan[:, 7], circle, clock)
        candidates = measured_pairs(raw, scan[:, 0])
        now = raw[candidates]
        truth_now = np.fromiter((labels[int(point)] for point in now[:, :, POINT].ravel()), dtype=int).reshape(-1, 2)
        if len(now) and previous:
            earlier = np.concatenate([rows for rows, _ in previous])
            truth_earlier = np.concatenate([truth for _, truth in previous])
            rows, cols, features = link_features(earlier, now, tolerance=.000007)
            past_ids, current_ids = truth_earlier[rows], truth_now[cols]
            known = (past_ids >= 0).all(axis=1) & (current_ids >= 0).all(axis=1)
            positive = (past_ids > 0).all(axis=1) & (past_ids == current_ids).all(axis=1)
            evidence["candidates"] += len(rows)
            evidence["known"] += int(known.sum())
            evidence["positive"] += int((known & positive).sum())
            for gap in (1., 3., 5.):
                evidence[f"positive_gap_{int(gap)}"] += int((known & positive & (features[:, 0] == gap)).sum())
            trained.append(features[known])
            targets.append(positive[known])
        previous.append((now, truth_now))
    print(path.name, dict(evidence), flush=True)
    return np.concatenate(trained), np.concatenate(targets), evidence


@dataclass(slots=True)
class ObservedPair:
    observations: np.ndarray
    owners: tuple[int, int]
    confirmed: bool


@dataclass(slots=True)
class PairScan:
    columns: np.ndarray
    decisions: list
    locked: list
    strict_count: int = 0
    protected: set = field(default_factory=set)
    mates: dict = field(default_factory=dict)


class PairReturns:
    def __init__(self, model, start_floor, link_floor, early=True, *, wide=False, fallback_floor=0.,
                 fallback_margin=None):
        self.model = model
        self.start_floor = start_floor
        self.link_floor = link_floor
        self.early = early
        self.wide = wide
        self.fallback_floor = fallback_floor
        self.fallback_margin = fallback_margin
        self.stats = Counter()

    @staticmethod
    def retained(state):
        return {owner for prior in state.get("previous_pairs", ()) if prior.confirmed for owner in prior.owners if owner}

    def begin(self, tracks, raw, times, circle, state):
        columns = measured_pairs(raw, times)
        strict_count = len(columns)
        previous = [prior for prior in state.get("previous_pairs", ())
                    if circle - prior.observations[0, SCAN] <= (MAX_GAP if prior.confirmed else 1)]
        registered_owners = {owner for prior in previous if prior.confirmed for owner in prior.owners}
        if self.wide and registered_owners:
            strict = set(map(tuple, columns))
            extended = [pair for pair in measured_pairs(raw, times, wide=True) if tuple(pair) not in strict]
            if extended:
                columns = np.vstack((columns, extended))
        frame = PairScan(columns, [], [], strict_count)
        active = {track.batch_id: index for index, track in enumerate(tracks)}
        if self.fallback_floor:
            frame.protected = {active[owner] for owner in registered_owners if owner in active}
        if self.fallback_margin is not None:
            for prior in previous:
                if prior.confirmed and all(owner in active for owner in prior.owners):
                    left, right = (active[owner] for owner in prior.owners)
                    frame.mates.setdefault(left, right)
                    frame.mates.setdefault(right, left)
        if not len(columns) or not previous:
            return frame
        rows, cols, features = link_features(np.stack([r.observations for r in previous]), raw[columns])
        if not len(rows):
            return frame
        scores = self.model.predict(features, num_threads=1)
        registered = np.asarray([prior.confirmed for prior in previous])[rows]
        ordinary = cols < strict_count
        accepted = (scores >= np.where(registered, self.link_floor, self.start_floor)) & (ordinary | registered)
        # A proven pair keeps both identities when a faint duplicate has the same offset.
        priority = scores - .045 * np.minimum(1., np.linalg.norm(features[:, -3:] / [.014, .05, .012], axis=1))
        priority += .05 * registered - .04 * ~ordinary
        used_rows, used_columns, used_owners = set(), set(), set()
        ranked = np.flatnonzero(accepted)
        for index in ranked[np.argsort(-priority[ranked], kind="stable")]:
            row, col = int(rows[index]), int(cols[index])
            prior, dest = previous[row], columns[col]
            owners = tuple((active[owner], int(point)) for owner, point in zip(prior.owners, dest) if owner in active)
            if row in used_rows or any(p in used_columns for p in dest) or any(k in used_owners for k, _ in owners):
                continue
            used_rows.add(row)
            used_columns.update(dest)
            used_owners.update(key for key, _ in owners)
            frame.decisions.append((prior, col))
            frame.locked.extend((key, point, float(max(.90, scores[index]))) for key, point in owners)
            self.stats["locked_points"] += len(owners)
            self.stats["matched_pairs"] += 1
            if not ordinary[index]:
                self.stats["wide_locks"] += len(owners)
        return frame

    def protect_fallback(self, frame, tracks, positions, circle, matches, held):
        if not frame.protected:
            return matches, held
        locks = {row for row, _, _ in frame.locked}
        accepted, added = [], []
        for row, col, probability in matches:
            if row in frame.protected and row not in locks and probability < self.fallback_floor:
                mate = frame.mates.get(row)
                if mate is None or mate in locks or not self.fallback_separated(tracks[row], tracks[mate],
                                                                                 positions[col], circle):
                    added.append(col)
                    continue
                self.stats["isolated_geometry_kept"] += 1
            accepted.append((row, col, probability))
        self.stats["fallback_holds"] += len(added)
        return accepted, np.r_[held, np.array(added, dtype=int)]

    def fallback_separated(self, owner, mate, position, circle):
        distance = []
        for track in (owner, mate):
            gap = circle - track.circle
            expected = track.position + gap * track.velocity
            d = residual(position, expected) / NOISE
            distance.append(float(d @ d) / (1 + .50 * (gap - 1)))
        direct, alternative = distance
        return direct <= 8. and alternative - direct >= self.fallback_margin

    def starts(self, frame, remaining):
        if not self.early:
            return
        for prior, index in frame.decisions:
            current = frame.columns[index]
            # A partial pair goes through the original per-target birth path.
            if any(prior.owners) or not remaining[current].all():
                continue
            self.stats["early_pair_starts"] += 1
            yield prior.observations, current

    def remember(self, frame, raw, output, circle, state):
        verified = {index: prior for prior, index in frame.decisions}
        used = {id(prior) for prior in verified.values()}
        observed_owners = set()
        previous = []
        for index, points in enumerate(raw[frame.columns]):
            owners = tuple(output.get(int(point), 0) for point in points[:, POINT])
            confirmed = index in verified and all(owners) and owners[0] != owners[1]
            if index >= frame.strict_count and not confirmed:
                continue
            if confirmed:
                observed_owners.update(owners)
            previous.append(ObservedPair(points, owners, confirmed))
        for prior in state.get("previous_pairs", ()):
            if (prior.confirmed and id(prior) not in used and circle - prior.observations[0, SCAN] < MAX_GAP
                    and not observed_owners.intersection(prior.owners)):
                previous.append(prior)
        state["previous_pairs"] = previous


def train(args):
    from lightgbm import Dataset, train as fit

    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    examples = parallel(collect, [(p, ASSETS["clock"]) for p in paths], args.jobs)
    x = np.concatenate([x for x, _, _ in examples])
    y = np.concatenate([y for _, y, _ in examples])
    params = dict(objective="binary", num_leaves=23, min_data_in_leaf=28, learning_rate=.055, verbosity=-1,
                  num_threads=args.jobs, force_col_wise=True, deterministic=True, seed=15)
    model = fit(params, Dataset(x, label=y, feature_name=list(FEATURES)), num_boost_round=170)
    model.save_model(str(args.model))
    return {"samples": len(y), "positive": int(y.sum()), "features": FEATURES, "parameters": params,
            "evidence": {p.stem: dict(evidence) for p, (_, _, evidence) in zip(paths, examples)}}


def evaluate(spec):
    from lightgbm import Booster
    from gap_recovery import IsolatedGapRecovery
    from raw_timing_detection import RawReturnGate

    path, args = spec
    upstream = [Booster(model_file=str(path)) for path in ASSETS.values()]
    model = Booster(model_file=str(args.model)) if args.mode != "control" else None
    paired = PairReturns(model, args.start_floor, args.link_floor, early=args.mode == "early",
                         wide=args.wide, fallback_floor=args.fallback_floor,
                         fallback_margin=args.fallback_margin) if model else None
    gate = RawReturnGate(Booster(model_file=str(args.raw_model)), args.raw_floor) if args.raw_model else None
    recovery = IsolatedGapRecovery(args.recovery_floor) if args.recovery_floor else None
    run = BirthOnline(*upstream, args.birth_floor, "supplement", paired_birth=True, joint=True, two_hit_pairs=True,
                      pairing=paired, raw_gate=gate, gap_recovery=recovery, birth_gap=args.birth_gap,
                      retain_ambiguous_seeds=args.retain_ambiguous_seeds, association_floor=args.association_floor,
                      association_margin=args.association_margin, supported_birth_floor=args.supported_birth_floor,
                      support_legacy_confirmation=args.support_legacy_confirmation)
    with (patch.object(tracker_compat, "baseline_circle", run.track),
          patch.object(replay, "load_handler", return_value=load_runtime().process_circle)):
        result = replay.replay_packet(path, allowed=TUNING_PACKETS)
    result["pair_counters"] = dict(paired.stats) if paired else {}
    if recovery:
        result["gap_counters"] = dict(recovery.counters)
    print(path.name, round(result["score"]["total"], 4), result["id_switches"], result["split_truth_tracks"],
          result["max_circle_ms"], result["pair_counters"], flush=True)
    return result


def tune(args):
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
    results = parallel(evaluate, [(path, args) for path in paths], args.jobs)
    summary = aggregate_packets(results)
    print("total", summary["score"]["total"], "switches", summary["id_switches"], "splits",
          summary["split_truth_tracks"], "IDF1", summary["idf1"], flush=True)
    return {"summary": summary,
            "packets": [{k: v for k, v in item.items() if k not in ("tracks", "_circle_times")} for item in results]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "tune"))
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--model", type=Path, default=HERE / "models/paired_returns.txt")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("early", "links", "control"), default="early")
    parser.add_argument("--start-floor", type=float, default=.985)
    parser.add_argument("--link-floor", type=float, default=.95)
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--fallback-floor", type=float, default=0.)
    parser.add_argument("--fallback-margin", type=float)
    parser.add_argument("--raw-model", type=Path)
    parser.add_argument("--raw-floor", type=float, default=.025)
    parser.add_argument("--recovery-floor", type=float, default=0.)
    parser.add_argument("--birth-floor", type=float, default=.86)
    parser.add_argument("--birth-gap", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--retain-ambiguous-seeds", action="store_true")
    parser.add_argument("--association-floor", type=float, default=.20)
    parser.add_argument("--association-margin", type=float, default=.30)
    parser.add_argument("--supported-birth-floor", type=float)
    parser.add_argument("--support-legacy-confirmation", action="store_true")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not 0 < args.link_floor <= args.start_floor < 1 or not 0 <= args.fallback_floor <= 1:
        parser.error("invalid start, pair link or single fallback floor")
    if args.fallback_margin is not None and (args.fallback_margin <= 0 or not args.fallback_floor):
        parser.error("a positive single-return margin requires an active fallback floor")
    if not 0 < args.raw_floor < 1:
        parser.error("expected raw return score floor between 0 and 1")
    if not 0 <= args.recovery_floor < .20:
        parser.error("expected gap recovery score floor below the ordinary .20 floor")
    if not 0 < args.birth_floor < 1 or not 0 < args.association_floor < 1:
        parser.error("expected birth and association thresholds between 0 and 1")
    if not args.association_margin >= 0:
        parser.error("expected a nonnegative association margin")
    if args.supported_birth_floor is not None and (
        not args.raw_model or not 0 < args.supported_birth_floor < args.birth_floor
    ):
        parser.error("a lower score floor for independently supported starts requires --raw-model")
    if (args.retain_ambiguous_seeds or args.support_legacy_confirmation) and not args.raw_model:
        parser.error("the selected birth experiment requires --raw-model")
    result = (train if args.action == "train" else tune)(args)
    model_paths = ASSETS | {"paired_returns": args.model}
    if args.raw_model:
        model_paths["raw_return"] = args.raw_model
    result["metadata"] = {
        "packets": args.packets, "mode": args.mode, "start_floor": args.start_floor,
        "link_floor": args.link_floor, "score_version": SCORE_VERSION, "wide": args.wide,
        "fallback_floor": args.fallback_floor,
        "fallback_margin": args.fallback_margin,
        "raw_floor": args.raw_floor if args.raw_model else None,
        "recovery_floor": args.recovery_floor,
        "birth_floor": args.birth_floor,
        "birth_gap": args.birth_gap,
        "retain_ambiguous_seeds": args.retain_ambiguous_seeds,
        "association_floor": args.association_floor,
        "association_margin": args.association_margin,
        "supported_birth_floor": args.supported_birth_floor,
        "support_legacy_confirmation": args.support_legacy_confirmation,
        "source_sha256": {path.name: replay.file_sha256(path)
                          for path in (Path(__file__), HERE / "causal_birth_paths.py")},
        "model_sha256": {kind: replay.file_sha256(path) for kind, path in model_paths.items()
                         if args.mode != "control" or kind != "paired_returns"},
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
