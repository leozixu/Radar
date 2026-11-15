"""Online radar association, motion states and classifier feature extraction."""

from dataclasses import dataclass, field

import numpy as np

from tracker_assoc import assign, residual, spatial_edges


OUTPUT_FIELDS = ("Time", "Azimuth", "Elevation", "Distance", "CircleID", "PntID")
HISTORY = 8
START_WIDTH = np.array([0.042, 0.12, 0.019])
FOLLOW_WIDTH = np.array([0.014, 0.065, 0.0052])
MAX_WIDTH = np.array([0.12, 0.32, 0.080])
ASSOCIATION_FEATURES = (
    "hits", "gap", "dt", "time_fallback", "batch_exists", "local_points", "track_competitors",
    "amp", "amp_last", "amp_mean", "amp_std", "amp_difference", "amp_change",
    "gate_distance", "nis", "nis_az", "nis_el", "nis_range", "gate_az", "gate_el", "gate_range",
    "abs_prediction_error_az", "abs_prediction_error_el", "abs_prediction_error_range",
    "abs_motion_error_az", "abs_motion_error_el", "abs_motion_error_range",
    "abs_last_speed_error_az", "abs_last_speed_error_el", "abs_last_speed_error_range",
    "abs_speed_az", "abs_speed_el", "abs_speed_range", "speed_az", "speed_el", "speed_range",
    "post_fit_az", "post_fit_el", "post_fit_range", "long_speed_az", "long_speed_el", "long_speed_range",
    "local_rank", "nearest_margin", "age", "recent_hits",
)
BIRTH_FEATURES = ASSOCIATION_FEATURES + (
    "segment_mean_amp", "segment_min_amp", "segment_max_amp", "segment_std_amp",
    "segment_span_az", "segment_span_el", "segment_span_range",
    "segment_efficiency_az", "segment_efficiency_el", "segment_efficiency_range",
    "segment_accel_az", "segment_accel_el", "segment_accel_range",
)


@dataclass(frozen=True, slots=True)
class TrackerSettings:
    measurement: tuple = (3.0e-6, 8.9e-5, 2.7e-7)
    process: tuple = (5.0e-7, 3.0e-6, 1.0e-7)
    use_timestamp: bool = True
    use_acceleration: bool = False
    use_history: bool = True
    use_covariance_gate: bool = True
    gate_scale: float = 1.0
    association_weight: float = 0.37
    mature_association_scale: float = 1.0
    birth_weight: float = 0.12
    mature_birth_scale: float = 1.0
    use_link_confidence: bool = False
    confirmed_bonus: float = 0.24
    confirmed_link_floor: float = 0.0
    robust_sigma: float = 0.0
    protect_geometric_mature: bool = False
    birth_threshold: float = 0.60
    survival_threshold: float = 0.18
    seed_amplitude: float = 0.16
    seed_tentative_matches: bool = False
    seed_alternatives: int = 0
    ambiguity_margin: float = 0.045
    max_tentative_age: int = 8
    max_coast: int = 4
    max_dormant: int = 12
    dormant_threshold: float = 0.58
    recovery_probability: float = 0.70
    recovery_margin: float = 0.22
    segment_recovery: bool = False
    segment_recovery_limit: float = 11.0
    segment_recovery_margin: float = 3.0


@dataclass(frozen=True, slots=True)
class TrackerModels:
    association: object = None
    birth: object = None


@dataclass(slots=True)
class Observation:
    pos: np.ndarray
    amplitude: float
    time: int
    circle: int
    pid: int


@dataclass(slots=True)
class Pending:
    position: np.ndarray
    velocity: np.ndarray
    circle: int
    started: int


@dataclass(slots=True)
class Track:
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    observations: list
    started: int
    hits: int = 1
    batch: int = 0
    confidence: float = 0.0
    speed: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mean_speed: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fit_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pending: list = field(default_factory=list)

    @property
    def last(self):
        return self.observations[-1]


def _prior(covariance, dt, process):
    p, cross, velocity = covariance[:, 0], covariance[:, 1], covariance[:, 2]
    dt = dt[:, None]
    return (p + 2 * dt * cross + dt**2 * velocity + process * dt**3 / 3,
            cross + dt * velocity + process * dt**2 / 2,
            velocity + process * dt)


def _elapsed(times, last_times, gap, use_timestamp):
    timestamp_steps = (times - last_times) / 5000.0
    usable = (np.abs(timestamp_steps - gap) <= np.maximum(0.55, 0.35 * gap)) & (timestamp_steps > 0.10)
    if not use_timestamp:
        usable[:] = False
    return np.where(usable, timestamp_steps, gap), ~usable


def _sample_history(tracks):
    n = len(tracks)
    pos, amp, circles = np.zeros((n, HISTORY, 3)), np.zeros((n, HISTORY)), np.zeros((n, HISTORY), dtype=int)
    size = np.empty(n, dtype=int)
    for index, track in enumerate(tracks):
        values = track.observations
        count = len(values)
        size[index] = count
        pos[index, :count] = [o.pos for o in values]
        amp[index, :count] = [o.amplitude for o in values]
        circles[index, :count] = [o.circle for o in values]
    return pos, amp, circles, size


def _pids(tracks, rows):
    previous = np.full((len(tracks), HISTORY), -1, dtype=np.int64)
    for index, track in enumerate(tracks):
        ids = [observation.pid for observation in reversed(track.observations)]
        previous[index, :len(ids)] = ids
    return previous[rows]


def _log_odds(probability):
    p = np.clip(probability, 0.018, 0.982)
    return np.log(p / (1 - p))


def _classifier(model, features):
    return np.asarray(model.predict(features, num_threads=1), dtype=float)


def _history_features(rows, cols, positions, amp, hpos, hamp, hsize, delta_last, dt, last_speed):
    observed = hamp[rows]
    lengths = hsize[rows]
    valid = np.arange(HISTORY) < lengths[:, None]
    current = amp[cols]
    total = observed.sum(axis=1) + current
    count = lengths + 1
    mean = total / count
    var = (np.square(observed).sum(axis=1) + current**2) / count - mean**2
    first = hpos[rows, 0]
    span = np.abs(residual(positions[cols], first))
    steps = np.abs(residual(hpos[:, 1:], hpos[:, :-1]))
    steps *= (np.arange(1, HISTORY) < hsize[:, None])[:, :, None]
    total_distance = np.abs(delta_last) + steps.sum(axis=1)[rows]
    return np.column_stack((mean,
                            np.minimum(current, np.where(valid, observed, np.inf).min(axis=1)),
                            np.maximum(current, observed.max(axis=1)),
                            np.sqrt(np.maximum(var, 0)), span,
                            span / np.maximum(total_distance, 1e-7),
                            np.abs(delta_last / dt[:, None] - last_speed[rows])))


def _support_pending(tracks, rows, cols, positions, circle, utility):
    """Use unresolved past observations as motion evidence, not emitted assignments."""
    for r, track in enumerate(tracks):
        pending = track.pending
        if not pending:
            continue
        edges = np.flatnonzero(rows == r)
        bonus = np.zeros(len(edges))
        for hypothesis in pending:
            if circle - hypothesis.started > 5:
                continue
            extrapolated = hypothesis.position + hypothesis.velocity * (circle - hypothesis.circle)
            offset = residual(positions[cols[edges]], extrapolated) / FOLLOW_WIDTH
            support = 0.08 * np.exp(-4 * np.sum(np.square(offset), axis=1))
            np.maximum(bonus, support, out=bonus)
        utility[edges] += bonus


def _next_best_by_group(indices, values, count):
    best = np.zeros(count)
    np.maximum.at(best, indices, values)
    matches = np.flatnonzero(values == best[indices])
    _, first = np.unique(indices[matches], return_index=True)
    heads = matches[first]
    runner_up = np.zeros(count)
    eligible = np.ones(len(indices), dtype=bool)
    eligible[heads] = False
    np.maximum.at(runner_up, indices[eligible], values[eligible])
    result = best[indices]
    result[heads] = runner_up[indices[heads]]
    return result


def _candidate_edges(tracks, positions, amps, times, circle, settings, models, observer, pids):
    measurement = np.asarray(settings.measurement)
    process = np.asarray(settings.process)
    hpos, hamp, hcircle, hsize = _sample_history(tracks)
    indexes = np.arange(len(tracks))
    age = np.asarray([circle - track.started for track in tracks])
    hits = np.asarray([track.hits for track in tracks])
    established = np.asarray([track.batch > 0 for track in tracks])
    confidence = np.asarray([track.confidence for track in tracks])
    last_pos = hpos[indexes, hsize - 1]
    last_amp = hamp[indexes, hsize - 1]
    last_time = np.asarray([track.last.time for track in tracks], dtype=np.int64)
    track_gap = circle - hcircle[indexes, hsize - 1]
    track_pos = np.stack([track.position for track in tracks])
    track_vel = np.stack([track.velocity for track in tracks])
    last_speed = np.stack([track.speed for track in tracks])
    long_speed = np.stack([track.mean_speed for track in tracks])
    fit_error = np.stack([track.fit_error for track in tracks])
    covariance = np.stack([track.covariance for track in tracks])
    acceleration = np.stack([track.acceleration for track in tracks]) if settings.use_acceleration else np.zeros_like(track_vel)
    motion_vel = 0.80 * track_vel + 0.20 * long_speed if settings.use_history else track_vel
    step = track_gap[:, None]
    expected = track_pos + step * motion_vel + 0.5 * acceleration * step**2
    position_var = _prior(covariance, track_gap, process)[0]
    base_gate = np.where((hits == 1)[:, None], START_WIDTH, FOLLOW_WIDTH) * settings.gate_scale
    base_gate *= (1 + 0.22 * (step - 1))
    if settings.use_covariance_gate:
        base_gate = np.maximum(base_gate, 3.3 * np.sqrt(position_var + measurement))
    gate = np.minimum(base_gate, MAX_WIDTH)
    search_width = np.minimum(gate + 0.60 * np.abs(motion_vel) + 0.38 * np.abs(acceleration) * step, MAX_WIDTH)
    rows, cols = spatial_edges(positions, expected, search_width)
    if not len(rows):
        return None
    gap = track_gap[rows]
    dt, fallback = _elapsed(times[cols], last_time[rows], gap, settings.use_timestamp)
    predicted = track_pos[rows] + dt[:, None] * motion_vel[rows] + 0.5 * acceleration[rows] * dt[:, None]**2
    error = residual(positions[cols], predicted)
    geom = np.sum(np.square(error / gate[rows]), axis=1)
    keep = geom <= 1.3
    if not keep.any():
        return None
    rows, cols, dt, gap, fallback, geom, error = (a[keep] for a in (rows, cols, dt, gap, fallback, geom, error))
    position_var = _prior(covariance[rows], dt, process)[0]
    norm_error = np.abs(error) / np.sqrt(position_var + measurement)
    nis = np.sum(norm_error**2, axis=1)
    delta_last = residual(positions[cols], last_pos[rows])
    amp_delta = amps[cols] - last_amp[rows]
    mean_amp = hamp.sum(axis=1) / hsize
    amp_std = np.sqrt(np.maximum((hamp**2).sum(axis=1) / hsize - mean_amp**2, 0))
    local_count = np.bincount(rows, minlength=len(tracks))
    competitor_count = np.bincount(cols, minlength=len(positions))
    rate = delta_last / dt[:, None]
    speed_error = np.abs(rate - last_speed[rows])
    motion_error = np.abs(delta_last - long_speed[rows] * dt[:, None])
    # Relative gate ranks tell the learner whether another credible point is close.
    ranking = np.lexsort((geom, rows))
    starts = np.cumsum(local_count) - local_count
    rank = np.empty(len(rows), dtype=int)
    rank[ranking] = np.arange(len(rows)) - starts[rows[ranking]]
    nearest = np.full(len(tracks), np.inf)
    np.minimum.at(nearest, rows, geom)
    feature_columns = (
        hits[rows], gap, dt, fallback, established[rows], local_count[rows], competitor_count[cols],
        amps[cols], last_amp[rows], mean_amp[rows], amp_std[rows], np.abs(amps[cols] - mean_amp[rows]),
        amp_delta, geom, nis, norm_error, gate[rows], np.abs(error), motion_error, speed_error,
        np.abs(rate), rate, fit_error[rows], long_speed[rows], rank, geom - nearest[rows], age[rows], hsize[rows],
    )
    features = np.column_stack(feature_columns).astype(np.float32)
    if observer is not None:
        observer("association", features, _pids(tracks, rows), pids[cols])
    assoc_prob = _classifier(models.association, features) if models.association is not None else None
    birth_prob = np.full(len(rows), np.nan)
    birth_edges = hits[rows] >= 2
    if observer is None and settings.mature_birth_scale == 0 and settings.use_link_confidence and assoc_prob is not None:
        birth_edges &= ~established[rows]
    if birth_edges.any() and (observer is not None or models.birth is not None):
        br, bc = rows[birth_edges], cols[birth_edges]
        birth_features = np.column_stack((features[birth_edges], _history_features(
            br, bc, positions, amps, hpos, hamp, hsize, delta_last[birth_edges], dt[birth_edges], last_speed,
        ))).astype(np.float32)
        if observer is not None:
            observer("birth", birth_features, _pids(tracks, br), pids[bc])
        if models.birth is not None:
            birth_prob[birth_edges] = _classifier(models.birth, birth_features)

    # All lifecycle stages share the same assignment, with explicit zero-cost misses.
    utility = (0.74 - 0.58 * geom - 0.16 * np.minimum(nis / 15, 2.0)
               - 0.08 * (gap - 1) - 0.55 * np.maximum(0.34 - amps[cols], 0)
               - 0.30 * np.abs(mean_amp[rows] - amps[cols]))
    mature = established[rows]
    if settings.protect_geometric_mature:
        geometric_support = (1 - geom - 0.45 * np.maximum(0.35 - amps[cols], 0) / 0.35
                             - 0.22 * np.abs(mean_amp[rows] - amps[cols]))
        recoverable = (assoc_prob >= settings.recovery_probability) if assoc_prob is not None else False
        protect = mature & (geometric_support > 0)
        utility[protect] = geometric_support[protect] - 0.08 * (gap[protect] - 1)
        utility[mature & ~protect & ~np.asarray(recoverable)] = -np.inf
    utility += np.where(mature, settings.confirmed_bonus + 0.16 * confidence[rows],
                        np.where(hits[rows] >= 2, 0.05, -0.18))
    if assoc_prob is not None:
        link_scale = np.where(mature, settings.mature_association_scale, 1.0)
        utility += settings.association_weight * link_scale * _log_odds(assoc_prob)
        utility[mature & (assoc_prob < settings.confirmed_link_floor)] = -np.inf
    if models.birth is not None:
        birth_scale = np.where(established[rows[birth_edges]], settings.mature_birth_scale, 1.0)
        utility[birth_edges] += settings.birth_weight * birth_scale * _log_odds(birth_prob[birth_edges])
    _support_pending(tracks, rows, cols, positions, circle, utility)
    dormant = established[rows] & (gap > settings.max_coast + 1)
    if dormant.any():
        # Require independent evidence that both ends favor this restoration.
        j = np.flatnonzero(dormant)
        row_alternative = _next_best_by_group(rows, utility, len(tracks))
        col_alternative = _next_best_by_group(cols, utility, len(positions))
        if assoc_prob is None:
            probable = (geom[j] < 0.35) & (nis[j] < 20)
        else:
            probable = assoc_prob[j] >= settings.recovery_probability
        other = np.maximum(row_alternative[j], col_alternative[j])
        bad = j[~probable | (utility[j] < other + settings.recovery_margin)]
        utility[bad] = -np.inf
    return rows, cols, utility, established, birth_prob, assoc_prob, dt, geom


def _is_retained(track, circle, settings):
    gap = circle - track.last.circle
    if track.batch:
        horizon = settings.max_dormant if track.confidence >= settings.dormant_threshold else settings.max_coast
        return gap <= horizon + 1
    return cycle_is_tentative(track, gap, circle, settings)


def cycle_is_tentative(track, gap, circle, settings):
    if track.hits == 1:
        return gap <= 2
    return gap <= min(4, settings.max_tentative_age) and circle - track.started <= settings.max_tentative_age


def _update(track, observation, elapsed, measurement, process, robust_sigma):
    step = residual(observation.pos, track.last.pos) / elapsed
    if track.hits == 1:
        track.position = observation.pos.copy()
        track.velocity = step.copy()
        track.mean_speed = step.copy()
        track.covariance = np.stack((measurement, measurement / elapsed, 2 * measurement / elapsed**2))
    else:
        p, cross, velocity = _prior(track.covariance[None, :, :], np.array([elapsed]), process)
        p, cross, velocity = p[0], cross[0], velocity[0]
        prediction = track.position + elapsed * track.velocity
        error = residual(observation.pos, prediction)
        if track.batch and robust_sigma > 0:
            innovation_limit = robust_sigma * np.sqrt(p + measurement)
            error = np.clip(error, -innovation_limit, innovation_limit)
            speed_limit = robust_sigma * np.sqrt(2 * measurement + elapsed**3 * process / 3) / elapsed
            step = np.clip(step, track.mean_speed - speed_limit, track.mean_speed + speed_limit)
        kp, kv = p / (p + measurement), cross / (p + measurement)
        track.position = prediction + kp * error
        track.velocity += kv * error
        track.mean_speed += (step - track.mean_speed) / min(track.hits, 6)
        track.acceleration = 0.70 * track.acceleration + 0.30 * (step - track.speed) / elapsed
        track.fit_error += (np.abs(error) - track.fit_error) / min(track.hits, 6)
        track.covariance = np.stack((kp * measurement, kv * measurement, velocity - kv * cross))
    track.position[0] %= 1.0
    track.hits += 1
    track.speed = step
    track.observations.append(observation)
    if len(track.observations) > HISTORY:
        del track.observations[0]
    track.pending.clear()


def _rule_birth(track, geom):
    observations = track.observations
    if len(observations) < 3:
        return False
    amplitude = sum(observation.amplitude for observation in observations) / len(observations)
    movement = residual(observations[-1].pos, observations[0].pos)
    enough_motion = np.linalg.norm(movement / np.array([0.002, 0.023, 0.0012])) >= 1
    return amplitude >= 0.30 and geom < 0.90 and enough_motion


def _seed_choices(tracks, rows, cols, utility, limit):
    if not limit:
        return {}
    seed_rows = np.asarray([t.hits == 1 for t in tracks])
    edges = np.flatnonzero(seed_rows[rows] & (utility > 0))
    by_point = {}
    for index in edges[np.lexsort((-utility[edges], cols[edges]))]:
        choices = by_point.setdefault(int(cols[index]), [])
        if len(choices) < limit:
            choices.append((int(index), tracks[rows[index]].last))
    return by_point


def _segment_rejoin(candidate, tracks, circle, settings, measurement, process):
    """Find a uniquely supported dormant identity before a segment first emits."""
    newest = candidate.observations[-3:]
    first, last = newest[0], newest[-1]
    points = np.stack([o.pos for o in newest])
    timestamps = np.asarray([o.time for o in newest], dtype=np.int64)
    circles = np.asarray([o.circle for o in newest])
    span, _ = _elapsed(timestamps[-1:], timestamps[0], circles[-1:] - circles[0], settings.use_timestamp)
    rate = residual(last.pos, first.pos) / span[0]
    possible = []
    for old in tracks:
        gap = circle - old.last.circle
        if (not old.batch or old.confidence < settings.dormant_threshold
                or not settings.max_coast + 1 < gap <= settings.max_dormant + 1
                or old.last.circle >= first.circle):
            continue
        dt, _ = _elapsed(timestamps, old.last.time, circles - old.last.circle, settings.use_timestamp)
        velocity = 0.80 * old.velocity + 0.20 * old.mean_speed if settings.use_history else old.velocity
        prediction = old.position + dt[:, None] * velocity
        variance = _prior(np.repeat(old.covariance[None], len(dt), axis=0), dt, process)[0]
        errors = residual(points, prediction)
        if np.any(np.abs(errors) > FOLLOW_WIDTH * (1 + 0.3 * gap)):
            continue
        motion = np.sum(errors**2 / (variance + measurement), axis=1)
        rate_var = old.covariance[2] + 2 * measurement / span[0]**2 + process * gap
        condition = np.mean(motion) + np.sum((rate - velocity)**2 / rate_var)
        if motion.max() <= settings.segment_recovery_limit:
            possible.append((condition, old))
    possible.sort(key=lambda item: item[0])
    if not possible or possible[0][0] > settings.segment_recovery_limit:
        return None
    if len(possible) > 1 and possible[1][0] - possible[0][0] < settings.segment_recovery_margin:
        return None
    return possible[0][1]


def process_candidates(points, state, settings, models=None, observer=None):
    models = models if models is not None else TrackerModels()
    circle = points[0]["CircleID"] if points else state.get("circle", -1) + 1
    state["circle"] = circle
    tracks = [track for track in state.get("tracks", ()) if _is_retained(track, circle, settings)]
    state["tracks"] = tracks
    if not points:
        return []
    position = np.asarray([(p["Azimuth"], p["Elevation"], p["Distance"]) for p in points], dtype=float)
    amp = np.asarray([p["Amp"] for p in points], dtype=float)
    time = np.asarray([p["Time"] for p in points], dtype=np.int64)
    pids = np.asarray([p["PntID"] for p in points], dtype=np.int64) if observer is not None else None
    # Geometric order provides reproducible ties independently of arrival order or PntID.
    order = np.lexsort((amp, time, position[:, 2], position[:, 1], position[:, 0]))
    position, amp, time = position[order], amp[order], time[order]
    if pids is not None:
        pids = pids[order]
    remaining = np.ones(len(points), dtype=bool)
    output = []
    batch = state.get("next_batch_id", 1)
    measurement, process = np.asarray(settings.measurement), np.asarray(settings.process)
    initial_covariance = np.stack((measurement, np.zeros(3), (START_WIDTH / 3)**2))
    observations = {}

    def observe(column):
        if column not in observations:
            identity = int(pids[column]) if pids is not None else -1
            observations[column] = Observation(position[column], float(amp[column]), int(time[column]), circle, identity)
        return observations[column]

    edges = _candidate_edges(tracks, position, amp, time, circle, settings, models, observer, pids) if tracks else None
    if edges is not None:
        rows, columns, utility, established, birth, link, dt, geom = edges
        choices = _seed_choices(tracks, rows, columns, utility, settings.seed_alternatives)
        selected, ambiguous = assign(rows, columns, utility, established, len(tracks), len(points),
                                     settings.ambiguity_margin)
        protected = set()
        for row, hypotheses in ambiguous.items():
            track = tracks[row]
            previous_start = min((p.started for p in track.pending), default=circle)
            if circle - previous_start > 5:
                previous_start = circle
            track.pending = []
            for index in sorted(hypotheses, key=lambda j: (-utility[j], j))[:8]:
                column = columns[index]
                velocity = residual(position[column], track.last.pos) / dt[index]
                track.pending.append(Pending(position[column], 0.70 * track.velocity + 0.30 * velocity,
                                             circle, previous_start))
                remaining[column] = False
                protected.add(column)
        confirmations = []
        for index in selected:
            track, column = tracks[rows[index]], columns[index]
            observed = observe(column)
            probability = birth[index]
            _update(track, observed, dt[index], measurement, process, settings.robust_sigma)
            remaining[column] = False
            if track.batch:
                protected.add(column)
                confidence = link[index] if settings.use_link_confidence and link is not None else probability
                if not np.isnan(confidence):
                    track.confidence = 0.78 * track.confidence + 0.22 * confidence
                if track.confidence >= settings.survival_threshold:
                    output.append((order[column], track.batch))
            elif track.hits >= 3:
                confirms = (_rule_birth(track, geom[index]) if np.isnan(probability)
                            else probability >= settings.birth_threshold)
                if confirms:
                    confirmations.append((track, column, 0.70 if np.isnan(probability) else probability))
            if not track.batch and settings.seed_tentative_matches and column not in protected:
                # A provisional link is not evidence against a fresh start here.
                remaining[column] = True

        committed_observations = set()
        if settings.seed_alternatives:
            # An alternative can share provisional observations but only one
            # path sharing such evidence may confirm an externally visible ID.
            committed_observations = {id(o) for t in tracks if t.batch for o in t.observations}
            confirmations.sort(key=lambda item: item[2], reverse=True)
        for track, column, probability in confirmations:
            history = {id(o) for o in track.observations} if settings.seed_alternatives else set()
            if history & committed_observations:
                continue
            committed_observations.update(history)
            dormant = _segment_rejoin(track, tracks, circle, settings, measurement, process) if settings.segment_recovery else None
            if dormant is None:
                track.confidence = probability
                track.batch = batch
                batch += 1
                track_id = track.batch
            else:
                for observation in track.observations[-3:]:
                    rejoin_dt, _ = _elapsed(np.array([observation.time]), dormant.last.time,
                                            np.array([observation.circle - dormant.last.circle]), settings.use_timestamp)
                    _update(dormant, observation, rejoin_dt[0], measurement, process, settings.robust_sigma)
                dormant.confidence = 0.75 * dormant.confidence + 0.25 * probability
                track_id = dormant.batch
                tracks[:] = [t for t in tracks if t is not track]
            remaining[column] = False
            protected.add(column)
            output.append((order[column], track_id))

        selected_edges = set(selected)
        for column, alternatives in choices.items():
            if column in protected:
                continue
            for index, previous in alternatives:
                if index in selected_edges or id(previous) in committed_observations:
                    continue
                alternate = Track(previous.pos.copy(), np.zeros(3), initial_covariance, [previous], previous.circle)
                _update(alternate, observe(column), dt[index], measurement, process, settings.robust_sigma)
                tracks.append(alternate)
        if committed_observations:
            tracks[:] = [t for t in tracks if t.batch or all(id(o) not in committed_observations for o in t.observations)]
    for column in np.flatnonzero(remaining & (amp >= settings.seed_amplitude)):
        observed = observe(column)
        tracks.append(Track(position[column].copy(), np.zeros(3), initial_covariance, [observed], circle))
    state["next_batch_id"] = batch
    return [{**{field: points[index][field] for field in OUTPUT_FIELDS}, "BatchID": track_id}
            for index, track_id in sorted(output)]
