"""Causal association of quality-filtered tracklet observations."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


SCALE = np.array([0.004, 0.025, 0.0016])
FEATURES = (
    "gap", "observations", "raw_hits", "same_raw", "recent_raw_share", "quality", "previous_quality",
    "amplitude_error", "last_amplitude_error", "reference_amplitude_sd",
    "long_prediction_az", "long_prediction_el", "long_prediction_range",
    "short_prediction_az", "short_prediction_el", "short_prediction_range",
    "anchored_prediction_az", "anchored_prediction_el", "anchored_prediction_range",
    "long_velocity_az", "long_velocity_el", "long_velocity_range",
    "short_velocity_az", "short_velocity_el", "short_velocity_range",
    "raw_velocity_delta_az", "raw_velocity_delta_el", "raw_velocity_delta_range",
    "previous_fit_az", "previous_fit_el", "previous_fit_range",
    "row_proximity_rank", "col_proximity_rank", "row_proximity_margin", "col_proximity_margin",
)


def load_quality_reference():
    """Use the previous 0.16 gate even if the deployment profile changes."""
    from lightgbm import Booster
    from tracker_compat import CompatibilityRuntime, QUALITY_FEATURES

    path = Path(__file__).resolve().parents[1] / "participant_demo" / "models" / "quality.txt"
    model = Booster(model_file=str(path))
    if tuple(model.feature_name()) != QUALITY_FEATURES:
        raise ValueError("incompatible reference quality model feature schema")
    return CompatibilityRuntime(model, .16, {})


def residual(observed, predicted):
    delta = observed - predicted
    delta[..., 0] = (delta[..., 0] + 0.5) % 1.0 - 0.5
    return delta


def straight_line(position, time):
    centered = time - time.mean()
    variance = centered @ centered
    slope = centered @ position / variance if variance > 0 else np.zeros(position.shape[1])
    return position.mean(axis=0) - time.mean() * slope, slope


@dataclass(slots=True)
class History:
    positions: list = field(default_factory=list)
    scans: list = field(default_factory=list)
    amplitudes: list = field(default_factory=list)
    raw_ids: list = field(default_factory=list)
    quality: float = 0.
    long_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    long_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    short_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    short_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fit: np.ndarray = field(default_factory=lambda: np.zeros(3))
    amplitude: float = 0.
    amplitude_sd: float = 0.
    hits: int = 0

    def add(self, scan, position, amplitude, raw_id, quality, raw_velocity):
        self.scans.append(scan)
        self.positions.append(position)
        self.amplitudes.append(amplitude)
        self.raw_ids.append(raw_id)
        self.quality = float(quality)
        self.hits += 1
        if len(self.scans) > 9:
            del self.scans[0], self.positions[0], self.amplitudes[0], self.raw_ids[0]
        if len(self.scans) == 1:
            self.long_position = self.short_position = position
            self.long_velocity = self.short_velocity = raw_velocity
            self.amplitude = amplitude
            return
        time = np.asarray(self.scans) - scan
        values = np.asarray(self.positions, dtype=float)
        values[:, 0] = np.unwrap(values[:, 0], period=1.)
        self.long_position, self.long_velocity = straight_line(values, time)
        self.short_position, self.short_velocity = straight_line(values[-4:], time[-4:])
        self.fit = np.sqrt(np.mean((values - self.long_position - time[:, None] * self.long_velocity) ** 2,
                                   axis=0))
        self.amplitude, self.amplitude_sd = float(np.mean(self.amplitudes)), float(np.std(self.amplitudes))


def proposal_arrays(points, proposals, scores, raw_state):
    point_amplitudes = {point["PntID"]: point["Amp"] for point in points}
    raw_tracks = {track.batch_id: track for track in raw_state["tracks"][0]}
    positions = np.asarray([(r["Azimuth"], r["Elevation"], r["Distance"]) for r in proposals]).reshape(-1, 3)
    amplitudes = np.asarray([point_amplitudes[r["PntID"]] for r in proposals])
    raw_ids = np.asarray([r["BatchID"] for r in proposals], dtype=int)
    hits = np.asarray([raw_tracks[raw].hits for raw in raw_ids])
    velocities = np.asarray([raw_tracks[raw].velocity.copy() for raw in raw_ids]).reshape(-1, 3)
    return positions, amplitudes, raw_ids, hits, velocities, scores


def candidate_features(histories, scan, arrays):
    positions, amplitudes, raw_ids, hits, velocities, scores = arrays
    if not histories or not len(positions):
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty((0, len(FEATURES)), dtype=np.float32)
    gap = scan - np.asarray([history.scans[-1] for history in histories])
    obs = np.asarray([history.hits for history in histories])
    long_velocity = np.asarray([history.long_velocity for history in histories])
    short_velocity = np.asarray([history.short_velocity for history in histories])
    long_position = np.asarray([history.long_position for history in histories])
    short_position = np.asarray([history.short_position for history in histories])
    last_position = np.asarray([history.positions[-1] for history in histories])
    p_long = long_position + gap[:, None] * long_velocity
    p_short = short_position + gap[:, None] * short_velocity
    p_anchor = last_position + gap[:, None] * long_velocity
    d_long = np.abs(residual(positions[None, :, :], p_long[:, None, :])) / SCALE
    d_short = np.abs(residual(positions[None, :, :], p_short[:, None, :])) / SCALE
    d_anchor = np.abs(residual(positions[None, :, :], p_anchor[:, None, :])) / SCALE
    distance = np.minimum(np.minimum(np.sum(d_long ** 2, axis=2), np.sum(d_short ** 2, axis=2)),
                          np.sum(d_anchor ** 2, axis=2)) ** .5
    same = np.asarray([history.raw_ids[-1] for history in histories])[:, None] == raw_ids
    gate = 4.8 + 0.75 * (gap - 1)
    rows, cols = np.nonzero((distance <= gate[:, None]) | (same & (distance <= 15.)))
    if not len(rows):
        return rows, cols, np.empty((0, len(FEATURES)), dtype=np.float32)
    row_best = distance.min(axis=1)
    col_best = distance.min(axis=0)
    rank_rows = np.argsort(np.argsort(distance, axis=1, kind="stable"), axis=1, kind="stable")
    rank_cols = np.argsort(np.argsort(distance, axis=0, kind="stable"), axis=0, kind="stable")
    by_raw = np.column_stack([np.mean(raw_ids[:, None] == np.asarray(history.raw_ids), axis=1)
                              for history in histories]).T
    recent_amplitude = np.asarray([history.amplitude for history in histories])
    last_amplitude = np.asarray([history.amplitudes[-1] for history in histories])
    rows_fit = np.asarray([history.fit for history in histories])
    features = np.column_stack((
        gap[rows], obs[rows], hits[cols], same[rows, cols], by_raw[rows, cols], scores[cols],
        [histories[row].quality for row in rows], np.abs(amplitudes[cols] - recent_amplitude[rows]),
        np.abs(amplitudes[cols] - last_amplitude[rows]), [histories[row].amplitude_sd for row in rows],
        d_long[rows, cols], d_short[rows, cols], d_anchor[rows, cols], long_velocity[rows] / SCALE,
        short_velocity[rows] / SCALE, np.abs(velocities[cols] - long_velocity[rows]) / SCALE,
        rows_fit[rows] / SCALE, rank_rows[rows, cols], rank_cols[rows, cols],
        distance[rows, cols] - row_best[rows], distance[rows, cols] - col_best[cols],
    )).astype(np.float32)
    return rows, cols, features


@dataclass(frozen=True, slots=True)
class IdentityRuntime:
    quality_model: object
    identity_model: object
    threshold: float = .16
    match_probability: float = .40
    same_raw_bonus: float = 0.
    hold_raw: bool = False
    confirm_hits: int = 1
    max_gap: int = 16
    publish_probability: float = 0.
    weak_cross_probability: float = 0.
    hold_nearby_births: float = 0.
    lineage_min_gap: int = 0

    def process_circle(self, points, state):
        from tracker_compat import decisions_with_features

        scan = points[0]["CircleID"] if points else state.get("scan", -1) + 1
        state["scan"] = scan
        raw = state.setdefault("raw", {})
        proposals, quality_features = decisions_with_features(points, raw)
        if not proposals:
            return []
        probabilities = self.quality_model.predict(quality_features, num_threads=1)
        keep = probabilities >= self.threshold
        proposals = [row for row, selected in zip(proposals, keep) if selected]
        if not proposals:
            return []
        arrays = proposal_arrays(points, proposals, probabilities[keep], raw)
        return self.associate(scan, proposals, arrays, state)

    def associate(self, scan, proposals, arrays, state):
        if self.lineage_min_gap:
            return self.stitch(scan, proposals, arrays, state)
        positions, amps, raw_ids, hits, velocity, quality = arrays
        by_id = {bid: h for bid, h in state.get("by_id", {}).items() if scan - h.scans[-1] <= self.max_gap}
        state["by_id"] = by_id
        raw_owner = {h.raw_ids[-1]: bid for bid, h in by_id.items() if h.raw_ids}
        track_ids = list(by_id)
        histories = list(by_id.values())
        rr, cc, features = candidate_features(histories, scan, arrays)
        matched = {}
        closest = np.zeros(len(proposals))
        if len(rr):
            prediction = np.clip(self.identity_model.predict(features, num_threads=1), .001, .999)
            np.maximum.at(closest, cc, prediction)
            utility = np.log(prediction / (1 - prediction))
            utility -= np.log(self.match_probability / (1 - self.match_probability))
            utility += features[:, 3] * self.same_raw_bonus
            weights = np.zeros((len(track_ids), len(proposals) + len(track_ids)))
            weights[rr, cc] = np.maximum(0, utility)
            rows, cols = linear_sum_assignment(weights, maximize=True)
            posterior = np.zeros((len(track_ids), len(proposals)))
            posterior[rr, cc] = prediction
            matched = {c: (track_ids[r], posterior[r, c]) for r, c in zip(rows, cols)
                       if c < len(proposals) and weights[r, c] > 0}
        occupied = {bid for bid, _ in matched.values()}
        pending = state.setdefault("pending", {})
        next_pending = {}
        output = []
        for i, proposal in enumerate(proposals):
            linked = matched.get(i)
            raw_id = raw_ids[i]
            publish = True
            if linked is not None:
                bid, posterior = linked
                if posterior < self.publish_probability:
                    continue
                if posterior < self.weak_cross_probability and raw_id != by_id[bid].raw_ids[-1]:
                    publish = False
            else:
                last_scan, count = pending.get(raw_id, (-2, 0))
                count = count + 1 if scan - last_scan <= 2 else 1
                required = self.confirm_hits
                if self.hold_nearby_births > 0 and closest[i] >= self.hold_nearby_births:
                    required = max(2, required)
                if count < required or (self.hold_raw and raw_id in raw_owner and raw_owner[raw_id] not in occupied):
                    next_pending[raw_id] = (scan, count)
                    continue
                bid = state.get("next_id", 1)
                state["next_id"] = bid + 1
                by_id[bid] = History()
            by_id[bid].add(scan, positions[i], amps[i], raw_id, quality[i], velocity[i])
            if publish:
                output.append(proposal | {"BatchID": bid})
        state["pending"] = next_pending
        return output

    def stitch(self, scan, proposals, arrays, state):
        """Recover only a dormant ID at the first emission of a new source ID."""
        positions, amps, raw_ids, _, velocity, quality = arrays
        aliases = state.setdefault("aliases", {})
        histories = {ident: history for ident, history in state.get("histories", {}).items()
                     if scan - history.scans[-1] <= self.max_gap}
        state["histories"] = histories
        occupied = set()
        associations = {}
        for index in np.argsort(-quality, kind="stable"):
            raw_id = raw_ids[index]
            old_id = aliases.get(raw_id)
            if old_id is not None and old_id not in occupied:
                associations[index] = old_id
                occupied.add(old_id)

        # Previously emitted raw IDs cannot acquire a different latent owner. If
        # two aliases unexpectedly reappear together, allocate a separate ID.
        births = [i for i, raw in enumerate(raw_ids) if raw not in aliases]
        dormant = {ident: history for ident, history in histories.items() if ident not in occupied
                   and history.hits >= 4 and scan - history.scans[-1] >= self.lineage_min_gap}
        if births and dormant:
            subset = tuple(a[births] for a in arrays)
            dormant_ids = list(dormant)
            rows, cols, x = candidate_features(list(dormant.values()), scan, subset)
            if len(rows):
                probabilities = self.identity_model.predict(x, num_threads=1)
                surface = np.zeros((len(dormant), len(births)))
                surface[rows, cols] = probabilities
                by_row = np.sort(surface, axis=1)
                by_col = np.sort(surface, axis=0)
                for r, c in zip(rows, cols):
                    probability = surface[r, c]
                    row_second = by_row[r, -2] if len(births) > 1 else 0.
                    col_second = by_col[-2, c] if len(dormant) > 1 else 0.
                    if (probability >= self.match_probability and probability == by_row[r, -1]
                            and probability == by_col[-1, c] and probability > max(row_second, col_second) + .12):
                        associations[births[c]] = dormant_ids[r]

        output = []
        for index, proposal in enumerate(proposals):
            raw_id = raw_ids[index]
            ident = associations.get(index)
            if ident is None:
                ident = state.get("next_id", 1)
                state["next_id"] = ident + 1
            aliases[raw_id] = ident
            history = histories.setdefault(ident, History())
            history.add(scan, positions[index], amps[index], raw_id, quality[index], velocity[index])
            output.append(proposal | {"BatchID": ident})
        return output
