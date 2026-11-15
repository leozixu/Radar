"""Try to attach a just-started public track to an unassigned, still-active track."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter

from lightgbm import Booster, Dataset, train as train_model
import numpy as np

from emission_identity import (
    FEATURES as EMISSION_FEATURES, HERE, INPUT_MODELS, RAW_RESCUE, accepted_emissions, current_raw_probabilities,
    decision_policies, outputs,
)
from identity_rejoin import (
    FEATURES as HISTORY_FEATURES, SCALE, History, candidate_features, proposal_arrays, residual, straight_line,
)
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
from replay_validation import _p99, file_sha256


EXTRA_FEATURES = ("source_raw_gap", "new_history_count", "new_history_span", "shared_scans") + tuple(
    f"{group}_{axis}" for group in ("current_kf", "first_old_forward", "new_backwards", "overlap")
    for axis in ("azimuth", "elevation", "range")
)
FEATURES = HISTORY_FEATURES + EXTRA_FEATURES
PHYSICAL_FEATURES = FEATURES + ("raw_probability", "pair_support", "pair_last_gap", "pair_amp_error",
                                "pair_el_error", "pair_range_error")
EMISSION_MODE = "rescue65_clutter10"
LEGACY_FLOOR = .16
MIN_OUTPUT_GAP = 2
MAX_OUTPUT_GAP = 12
MIN_ESTABLISHED_HITS = 3
MAX_NEW_RAW_HITS = 5
MIN_SCORE_MARGIN = .16


@dataclass(slots=True)
class PacketLineage:
    public: dict = field(default_factory=dict)
    seen: set = field(default_factory=set)
    history: dict = field(default_factory=dict)


@dataclass(slots=True)
class PublicHistory:
    measurements: deque = field(default_factory=lambda: deque(maxlen=9))
    hits: int = 0
    fitted: History | None = None

    def add(self, circle, position, amplitude, raw, quality):
        self.measurements.append((circle, position, amplitude, raw, float(quality)))
        self.hits += 1
        self.fitted = None

    @property
    def latest(self):
        return self.measurements[-1][0]

    def trajectory(self):
        if self.fitted is None:
            scans, positions, amplitudes, sources, quality = zip(*self.measurements)
            time = np.asarray(scans) - scans[-1]
            values = np.asarray(positions, dtype=float)
            values[:, 0] = np.unwrap(values[:, 0], period=1.)
            p, v = straight_line(values, time)
            short_p, short_v = straight_line(values[-4:], time[-4:])
            errors = values - p - time[:, None] * v
            self.fitted = History(
                positions=list(positions), scans=list(scans), amplitudes=list(amplitudes), raw_ids=list(sources),
                quality=quality[-1], long_position=p, long_velocity=v, short_position=short_p, short_velocity=short_v,
                fit=np.sqrt(np.mean(errors**2, axis=0)), amplitude=float(np.mean(amplitudes)),
                amplitude_sd=float(np.std(amplitudes)), hits=self.hits,
            )
        return self.fitted


def bridge_features(old_track, old_history, new_track, current_position, circle):
    prior = list(new_track.history)
    first_circle, _, first_position = prior[0]
    old_scan = old_history.scans[-1]
    new_old_gap = first_circle - old_scan
    old_kf = old_track.position + (circle - old_track.circle) * old_track.velocity
    old_forward = old_history.short_position + new_old_gap * old_history.short_velocity
    new_backwards = current_position + (old_scan - circle) * new_track.velocity
    earlier = dict(zip(old_history.scans, old_history.positions))
    overlap = np.asarray([residual(position, earlier[scan]) for scan, _, position in prior if scan in earlier])
    shared_error = np.abs(overlap).max(axis=0) if len(overlap) else np.full(3, np.nan)
    errors = np.concatenate((
        np.abs(residual(current_position, old_kf)), np.abs(residual(first_position, old_forward)),
        np.abs(residual(old_history.positions[-1], new_backwards)), shared_error,
    )) / np.tile(SCALE, 4)
    return (circle - old_track.circle, len(prior), circle - first_circle, len(overlap), *errors)


class ActiveLineage:
    minimum_history = MIN_ESTABLISHED_HITS
    maximum_new_hits = MAX_NEW_RAW_HITS

    def __init__(self, model=None, floor=.95):
        self.model = model
        self.floor = floor
        self.state = PacketLineage()
        self.evidence = Counter()
        self.last_links = []

    def candidates(self, points, proposals, rows, qualities, raw_state):
        if not rows:
            return None
        lineage = self.state
        active = {track.batch_id: track for track in raw_state["tracks"][0]}
        births = [i for i, row in enumerate(rows)
                  if lineage.public.get(row["BatchID"], row["BatchID"]) not in lineage.seen
                  and active[row["BatchID"]].hits <= self.maximum_new_hits]
        if not births:
            return None
        current_raw = {row["BatchID"] for row in proposals}
        active_unassigned = active.keys() - current_raw
        circle = raw_state["circle"]
        old_raw = [raw for raw in sorted(active_unassigned)
                   if (history := lineage.history.get(lineage.public.get(raw, raw))) is not None
                   and history.hits >= self.minimum_history
                   and MIN_OUTPUT_GAP <= circle - history.latest <= MAX_OUTPUT_GAP]
        if not old_raw:
            return None
        old_history = [lineage.history[lineage.public.get(raw, raw)].trajectory() for raw in old_raw]
        incoming = [rows[i] for i in births]
        arrays = proposal_arrays(points, incoming, qualities[births], raw_state)
        r, c, features = candidate_features(old_history, circle, arrays)
        if not len(r):
            return None
        bridges = np.asarray([
            bridge_features(active[old_raw[i]], old_history[i], active[incoming[j]["BatchID"]], arrays[0][j], circle)
            for i, j in zip(r, c)
        ])
        return old_raw, births, r, c, np.column_stack((features, bridges)).astype(np.float32)

    def remember(self, points, rows, qualities, raw_state):
        lineage = self.state
        circle = raw_state["circle"]
        public_rows = []
        if rows:
            amplitudes = {point["PntID"]: point["Amp"] for point in points}
            for row, quality in zip(rows, qualities):
                raw = row["BatchID"]
                ident = lineage.public.get(raw, raw)
                history = lineage.history.get(ident)
                if history is None:
                    history = lineage.history[ident] = PublicHistory()
                position = np.array((row["Azimuth"], row["Elevation"], row["Distance"]))
                history.add(circle, position, amplitudes[row["PntID"]], raw, quality)
                lineage.seen.add(ident)
                public_rows.append(row | {"BatchID": ident})
        lineage.history = {ident: history for ident, history in lineage.history.items()
                           if circle - history.latest <= MAX_OUTPUT_GAP}
        return public_rows

    def correct(self, points, proposals, rows, qualities, raw_state):
        self.last_links = []
        choices = self.candidates(points, proposals, rows, qualities, raw_state)
        if choices is not None:
            old_raw, births, r, c, features = choices
            p = self.probabilities(features)
            weights = np.zeros((len(old_raw), len(births)))
            weights[r, c] = p
            by_row = np.sort(weights, axis=1)
            by_col = np.sort(weights, axis=0)
            public = self.state.public
            self.evidence["candidates"] += len(r)
            for i, j in zip(r, c):
                probability = weights[i, j]
                if probability < self.floor or probability != by_row[i, -1] or probability != by_col[-1, j]:
                    continue
                old_alternative = by_row[i, -2] if len(births) > 1 else 0.
                new_alternative = by_col[-2, j] if len(old_raw) > 1 else 0.
                if probability < max(old_alternative, new_alternative) + MIN_SCORE_MARGIN:
                    continue
                old, fresh = old_raw[i], rows[births[j]]["BatchID"]
                # Swapping bindings preserves a bijection even when an old raw
                # track later emits another observation or a home ID is reused.
                old_public = public.get(old, old)
                public[old], public[fresh] = public.get(fresh, fresh), old_public
                self.last_links.append((old_public, rows[births[j]]["PntID"]))
                self.evidence["reattached"] += 1
        return self.remember(points, rows, qualities, raw_state)

    def probabilities(self, features):
        return self.model.predict(features, num_threads=1)


def pair_roles(points, rows):
    """Identify one-tick raw neighbors even when only one arm has a public ID."""
    by_tick = {}
    for point in points:
        tick = point["Time"]
        by_tick[tick] = point if tick not in by_tick else None
    roles = {}
    for row in rows:
        tick = row["Time"]
        own = by_tick.get(tick)
        if own is None:
            continue
        matching = []
        for shift in (-1, 1):
            other = by_tick.get(tick + shift)
            if other is None:
                continue
            early, late = (own, other) if shift > 0 else (other, own)
            amplitude = early["Amp"] - late["Amp"]
            azimuth = abs((early["Azimuth"] - late["Azimuth"] + .5) % 1. - .5)
            elevation, distance = late["Elevation"] - early["Elevation"], late["Distance"] - early["Distance"]
            if .000004 <= amplitude <= .026 and azimuth <= .018 and -1e-7 <= elevation <= .011 and abs(distance) <= .0038:
                matching.append((shift, amplitude, elevation, distance))
        if len(matching) == 1:
            roles[row["PntID"]] = matching[0]
    return roles


class CorroboratedActiveLineage(ActiveLineage):
    minimum_history = 2
    maximum_new_hits = 10

    def __init__(self, model=None, floor=.95):
        super().__init__(model, floor)
        self.paired_history = {}
        self.raw_roles = {}
        self.raw_roles_circle = -1

    def pairs(self, points, rows, circle):
        if circle != self.raw_roles_circle:
            self.raw_roles = pair_roles(points, rows)
            self.raw_roles_circle = circle
        return self.raw_roles

    def candidates(self, points, proposals, rows, qualities, raw_state):
        choices = super().candidates(points, proposals, rows, qualities, raw_state)
        if choices is None:
            return None
        old_raw, births, r, c, features = choices
        circle = raw_state["circle"]
        pairs = self.pairs(points, rows, circle)
        probabilities = current_raw_probabilities(points, rows, raw_state)
        added = []
        for i, j in zip(r, c):
            row = rows[births[j]]
            q = probabilities[births[j]]
            old_id = self.state.public.get(old_raw[i], old_raw[i])
            current = pairs.get(row["PntID"])
            if current is None:
                added.append((q, 0, np.nan, np.nan, np.nan, np.nan))
                continue
            role, amp, el, dist = current
            references = [sample for sample in self.paired_history.get(old_id, ())
                          if circle - sample[0] <= MAX_OUTPUT_GAP and sample[1] == role]
            observations = [sample for sample in references if abs(sample[2] - amp) <= .000004]
            support = len(observations)
            if support:
                center = np.median(np.asarray(observations)[:, 2:], axis=0)
                ae, ee, de = np.abs(center - np.array([amp, el, dist]))
                added.append((q, support if support >= .67 * len(references) else 0, circle - observations[-1][0],
                              ae, ee, de))
            else:
                added.append((q, 0, np.nan, np.nan, np.nan, np.nan))
        return old_raw, births, r, c, np.column_stack((features, added)).astype(np.float32)

    def probabilities(self, features):
        p = super().probabilities(features)
        q, support, gap, amplitude, elevation, distance = features[:, len(FEATURES):].T
        corroborated = ((q >= .65) & (support >= 2) & (gap <= 9) & (amplitude <= .0000031)
                        & (elevation <= .0018) & (distance <= .003))
        self.evidence["physically_corroborated_candidates"] += int(corroborated.sum())
        return np.where(corroborated, np.maximum(p, .995), p)

    def remember(self, points, rows, qualities, raw_state):
        corrected = super().remember(points, rows, qualities, raw_state)
        circle = raw_state["circle"]
        if corrected:
            evidence = self.pairs(points, rows, circle)
            for row in corrected:
                signature = evidence.get(row["PntID"])
                if signature is not None:
                    self.paired_history.setdefault(row["BatchID"], deque(maxlen=8)).append((circle, *signature))
        self.paired_history = {ident: path for ident, path in self.paired_history.items()
                               if ident in self.state.history and circle - path[-1][0] <= MAX_OUTPUT_GAP}
        return corrected


def emission_stream(path, allowed, raw_floor=None, association_floor=.20, emission_policy=EMISSION_MODE):
    models = {kind: Booster(model_file=str(HERE / "models" / f"emission_{kind}.txt"))
              for kind in ("detection", "ownership")}
    gate = ({"raw_model": HERE / "models" / "raw_timing_detection.txt", "raw_floor": raw_floor}
            if raw_floor is not None else {})
    thresholds = decision_policies(gate)[emission_policy]
    raw_rescue = RAW_RESCUE[emission_policy][1] if emission_policy in RAW_RESCUE else None
    for points, proposals, probabilities, features, labels, circles, state in outputs(
        path, allowed, HERE / "models" / "paired_returns.txt", wide_pairs=True, fallback_floor=.96,
        association_floor=association_floor, **gate
    ):
        keep = probabilities >= LEGACY_FLOOR
        if proposals:
            detection = models["detection"].predict(features, num_threads=1)
            ownership = models["ownership"].predict(features, num_threads=1)
            young = features[:, EMISSION_FEATURES.index("trusted_proposals")] < 3
            raw_probability = current_raw_probabilities(points, proposals, state) if raw_rescue is not None else None
            keep = accepted_emissions(keep, detection, ownership, young, thresholds, raw_probability, raw_rescue)
        rows = [row for row, accepted in zip(proposals, keep) if accepted]
        yield points, proposals, rows, probabilities[keep], labels, circles, state


def collect(spec):
    path, profile, physical = spec
    guard = CorroboratedActiveLineage() if physical else ActiveLineage()
    names = PHYSICAL_FEATURES if physical else FEATURES
    votes = defaultdict(Counter)
    features_all, labels_all = [], []
    evidence = Counter()
    for points, proposals, rows, qualities, labels, _, state in emission_stream(path, TRAIN_PACKETS, **profile):
        choices = guard.candidates(points, proposals, rows, qualities, state)
        if choices is not None:
            old_raw, births, r, c, features = choices
            for old_index, raw_index, features_row in zip(r, c, features):
                source = votes[old_raw[old_index]]
                incoming = labels[rows[births[raw_index]]["PntID"]]
                if incoming < 0 or not source:
                    continue
                owner, count = source.most_common(1)[0]
                if count < guard.minimum_history or count < .85 * source.total():
                    continue
                is_link = owner == incoming
                features_all.append(features_row)
                labels_all.append(is_link)
                evidence["known"] += 1
                evidence["positive"] += is_link
                if physical and features_row[-5] >= 2:
                    evidence["pair_observations_supported"] += 1
                    evidence["pair_observations_positive"] += is_link
        guard.remember(points, rows, qualities, state)
        for row in rows:
            label = labels[row["PntID"]]
            if label > 0:
                votes[row["BatchID"]][label] += 1
    print(path.stem, "causal candidate training", dict(evidence), flush=True)
    return np.asarray(features_all).reshape(-1, len(names)), np.asarray(labels_all, dtype=bool), evidence


def evaluate(spec):
    path, model, floors, training, profile, physical = spec
    guard_type = CorroboratedActiveLineage if physical else ActiveLineage
    controls = {floor: guard_type(Booster(model_file=str(model)), floor) for floor in floors}
    emitted = {"control": {}} | {str(floor): {} for floor in floors}
    past_truths = {floor: defaultdict(Counter) for floor in floors}
    overhead = {floor: [] for floor in floors}
    total_time = {}
    for points, proposals, rows, qualities, labels, circles, state in emission_stream(
        path, TRAIN_PACKETS if training else TUNING_PACKETS, **profile
    ):
        total_time.update((point["PntID"], point["Time"]) for point in points)
        emitted["control"].update((row["PntID"], row["BatchID"]) for row in rows)
        for floor, guard in controls.items():
            started = perf_counter()
            corrected = guard.correct(points, proposals, rows, qualities, state)
            overhead[floor].append((perf_counter() - started) * 1000.)
            votes = past_truths[floor]
            for old, pid in guard.last_links:
                owner = votes[old].most_common(1)
                if owner and labels[pid] >= 0:
                    key = "same_past_owner" if labels[pid] == owner[0][0] else "different_past_owner"
                else:
                    key = "unscored_past_owner"
                guard.evidence[key] += 1
            emitted[str(floor)].update((row["PntID"], row["BatchID"]) for row in corrected)
            for row in corrected:
                truth = labels[row["PntID"]]
                if truth > 0:
                    votes[row["BatchID"]][truth] += 1
    if any(series.keys() != emitted["control"].keys() for series in emitted.values()):
        raise ValueError("identity-only comparison changed the set of emitted source points")
    metrics = {name: evaluate_packet(labels, decisions, circles, total_time) for name, decisions in emitted.items()}
    events = {str(floor): dict(guard.evidence) for floor, guard in controls.items()}
    print(path.stem, {name: (round(m["score"]["total"], 4), m["id_switches"]) for name, m in metrics.items()},
          events, flush=True)
    return metrics, events, {str(floor): times for floor, times in overhead.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "tune", "train-check"))
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model", type=Path, default=HERE / "models" / "active_lineage_rejoin.txt")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--floor", type=float, nargs="+", default=[.95])
    parser.add_argument("--raw-floor", type=float)
    parser.add_argument("--association-floor", type=float, default=.20)
    parser.add_argument("--emission-policy", choices=tuple(decision_policies(True)), default=EMISSION_MODE)
    parser.add_argument("--physical-evidence", action="store_true")
    args = parser.parse_args()
    if any(not 0 < floor < 1 for floor in args.floor):
        parser.error("all floors must be between 0 and 1")
    if args.raw_floor is not None and not 0 <= args.raw_floor <= 1:
        parser.error("--raw-floor must be between 0 and 1")
    if not 0 < args.association_floor < 1:
        parser.error("--association-floor must be between 0 and 1")
    if args.emission_policy in RAW_RESCUE and args.raw_floor is None:
        parser.error("this emission policy requires --raw-floor")
    if args.physical_evidence and args.raw_floor is None:
        parser.error("--physical-evidence requires --raw-floor")
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS if args.stage == "tune" else TRAIN_PACKETS)
    profile = {"raw_floor": args.raw_floor, "association_floor": args.association_floor,
               "emission_policy": args.emission_policy}
    assets = INPUT_MODELS | {name: HERE / "models" / f"{name}.txt" for name in (
        "paired_returns", "emission_detection", "emission_ownership")}
    if args.raw_floor is not None:
        assets["raw_timing_detection"] = HERE / "models" / "raw_timing_detection.txt"
    upstream_sources = ("paired_returns.py", "causal_birth_paths.py", "emission_identity.py")
    names = PHYSICAL_FEATURES if args.physical_evidence else FEATURES
    info = {"packets": args.packets, "stage": args.stage, "score_version": SCORE_VERSION, "features": names,
            "wide": True, "fallback_floor": .96, **profile,
            "upstream_assets_sha256": {name: file_sha256(path) for name, path in assets.items()},
            "upstream_source_sha256": {name: file_sha256(HERE / name) for name in upstream_sources},
            "source_sha256": file_sha256(Path(__file__))}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.stage == "fit":
            collected = list(pool.map(collect, [(path, profile, args.physical_evidence) for path in paths]))
            x = np.concatenate([data for data, _, _ in collected])
            y = np.concatenate([labels for _, labels, _ in collected])
            if not y.any() or y.all():
                raise ValueError("these training packets did not produce both association classes")
            parameters = dict(objective="binary", num_leaves=7, max_depth=3, min_data_in_leaf=12,
                              learning_rate=.055, verbosity=-1, num_threads=args.jobs, force_col_wise=True,
                              deterministic=True, seed=22)
            model = train_model(parameters, Dataset(x, label=y, feature_name=list(names)), num_boost_round=110)
            model.save_model(str(args.model))
            info.update({"known": len(y), "positive": int(y.sum()), "parameters": parameters,
                         "per_packet": {p.stem: data[2] for p, data in zip(paths, collected)}})
        else:
            checks = list(pool.map(evaluate, [(path, args.model, args.floor, args.stage == "train-check", profile,
                                               args.physical_evidence)
                                                for path in paths]))
            info["summaries"] = {name: aggregate_packets(m[name] for m, _, _ in checks) for name in checks[0][0]}
            for floor in map(str, args.floor):
                times = [duration for _, _, timings in checks for duration in timings[floor]]
                info["summaries"][floor]["guard_p99_ms"] = _p99(times)
                info["summaries"][floor]["guard_max_ms"] = max(times)
            info["per_packet"] = {path.stem: {"metrics": {name: {k: v for k, v in m.items() if k != "tracks"}
                                                               for name, m in metrics.items()}, "events": events}
                                  for path, (metrics, events, _) in zip(paths, checks)}
    info["model_sha256"] = file_sha256(args.model)
    args.report.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
