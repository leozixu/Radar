"""Causal multi-target tracker for the official per-circle Python interface."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


# Coordinates are normalized in the input. Use independent scales for each axis.
START_GATE = np.array([0.036, 0.090, 0.016])
FOLLOW_GATE = np.array([0.016, 0.075, 0.006])
# Continuous acceleration noise per axis, relative to unit measurement variance.
PROCESS_NOISE = np.array([0.30, 0.01, 1.00])
OUTPUT_FIELDS = ("Time", "Azimuth", "Elevation", "Distance", "CircleID", "PntID")
MAX_MISSED_CIRCLES = 3
SEED_MIN_AMPLITUDE = 0.18
CONFIRM_MIN_MEAN_AMPLITUDE = 0.30


@dataclass(slots=True)
class Track:
    position: np.ndarray
    velocity: np.ndarray
    circle: int
    mean_amplitude: float
    first_position: np.ndarray
    hits: int = 1
    batch_id: int = 0
    # Position variance, position–velocity covariance, and velocity variance per axis.
    covariance: np.ndarray | None = field(default=None, init=False)


def residual(observed, predicted):
    delta = observed - predicted
    # Use the shortest azimuth difference across the normalized turn boundary.
    delta[..., 0] = (delta[..., 0] + 0.5) % 1.0 - 0.5
    return delta


def associate(tracks, positions, amplitudes, circle, initial=False):
    """Return one-to-one matches, with an explicit unmatched choice per track."""
    if not tracks or len(positions) == 0:
        return []

    elapsed = np.asarray([circle - track.circle for track in tracks])
    predicted = np.stack([track.position for track in tracks])
    predicted += np.stack([track.velocity for track in tracks]) * elapsed[:, None]
    gate = START_GATE if initial else FOLLOW_GATE
    gate = gate * (1 + 0.35 * (elapsed - 1))[:, None]
    delta = residual(positions[None, :, :], predicted[:, None, :])
    distance = np.sum(np.square(delta / gate[:, None, :]), axis=2)

    # Weak or inconsistent amplitudes can still match an established track.
    previous_amp = np.asarray([track.mean_amplitude for track in tracks])
    amp_change = np.abs(amplitudes[None, :] - previous_amp[:, None])
    amp_cost = 0.45 * np.maximum(0.35 - amplitudes, 0) / 0.35
    cost = distance + amp_cost + 0.22 * amp_change
    cost[distance > 1] = np.inf
    unmatched = np.full((len(tracks), len(tracks)), np.inf)
    np.fill_diagonal(unmatched, 1.0)
    rows, columns = linear_sum_assignment(np.hstack((cost, unmatched)))
    return [(row, col) for row, col in zip(rows, columns) if col < len(positions)]


def update_track(track, position, amplitude, circle):
    elapsed = circle - track.circle
    if track.hits == 1:
        track.velocity = residual(position, track.position) / elapsed
        track.position = position.copy()
        # The velocity uses both measurements, including the current position.
        track.covariance = np.array(
            [np.ones(3), np.full(3, 1 / elapsed), np.full(3, 2 / elapsed**2)]
        )
    else:
        position_var, cross_cov, velocity_var = track.covariance
        # Propagate covariance with the constant-velocity transition and white acceleration.
        prior_position_var = (
            position_var + 2 * elapsed * cross_cov + elapsed**2 * velocity_var
            + PROCESS_NOISE * elapsed**3 / 3
        )
        prior_cross_cov = cross_cov + elapsed * velocity_var + PROCESS_NOISE * elapsed**2 / 2
        prior_velocity_var = velocity_var + PROCESS_NOISE * elapsed
        innovation_var = prior_position_var + 1
        position_gain = prior_position_var / innovation_var
        velocity_gain = prior_cross_cov / innovation_var

        prediction = track.position + elapsed * track.velocity
        error = residual(position, prediction)
        track.position = prediction + position_gain * error
        track.velocity += velocity_gain * error
        # With unit measurement variance the first two posterior entries equal the gains.
        position_var[:] = position_gain
        cross_cov[:] = velocity_gain
        velocity_var[:] = prior_velocity_var - velocity_gain * prior_cross_cov

    track.position[0] %= 1.0
    track.circle = circle
    track.hits += 1
    track.mean_amplitude += (amplitude - track.mean_amplitude) / min(track.hits, 8)


def process_circle(points, state):
    """Select current-circle source points and attach persistent positive IDs."""
    circle = points[0]["CircleID"] if points else state.get("circle", -1) + 1
    state["circle"] = circle
    established, tentative, seeds = state.get("tracks", ([], [], []))
    established = [track for track in established if circle - track.circle <= MAX_MISSED_CIRCLES + 1]
    tentative = [track for track in tentative if circle - track.circle == 1]
    seeds = [track for track in seeds if circle - track.circle == 1]

    positions = np.asarray(
        [(p["Azimuth"], p["Elevation"], p["Distance"]) for p in points], dtype=float
    ).reshape(-1, 3)
    amplitudes = np.asarray([p["Amp"] for p in points], dtype=float)
    remaining = np.arange(len(points))
    matched_tracks = []
    next_tentative = []
    batch_id = state.get("next_batch_id", 1)

    # Established tracks get first choice, then two-hit candidates, then one-hit seeds.
    for tier, tracks in enumerate((established, tentative, seeds)):
        matches = associate(tracks, positions[remaining], amplitudes[remaining], circle, initial=tier == 2)
        keep = np.ones(len(remaining), dtype=bool)
        for track_index, remaining_index in matches:
            track = tracks[track_index]
            point_index = remaining[remaining_index]
            position = positions[point_index]
            update_track(track, position, amplitudes[point_index], circle)
            keep[remaining_index] = False
            if tier == 2:
                next_tentative.append(track)
                continue
            if tier == 1:
                movement = residual(position, track.first_position)
                if (track.mean_amplitude < CONFIRM_MIN_MEAN_AMPLITUDE
                        or np.linalg.norm(movement / np.array([0.002, 0.025, 0.0012])) < 1):
                    continue
                track.batch_id = batch_id
                batch_id += 1
                established.append(track)
            matched_tracks.append((point_index, track.batch_id))
        remaining = remaining[keep]

    next_seeds = [
        Track(positions[index].copy(), np.zeros(3), circle, amplitudes[index], positions[index].copy())
        for index in remaining
        if amplitudes[index] >= SEED_MIN_AMPLITUDE
    ]
    state["tracks"] = (established, next_tentative, next_seeds)
    state["next_batch_id"] = batch_id
    return [
        {**{field: points[index][field] for field in OUTPUT_FIELDS}, "BatchID": track_id}
        for index, track_id in sorted(matched_tracks)
    ]
