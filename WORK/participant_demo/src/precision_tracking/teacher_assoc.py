"""Causal track state, motion estimates, and association features."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import baseline_cv as cv


SEARCH = np.array([0.024, 0.095, 0.009])
NOISE = np.array([0.0030, 0.015, 0.0011])
SCANS = 8
FEATURES = (
    "gap", "hits", "amplitude", "amplitude_last", "amplitude_mean",
    "amplitude_difference", "amplitude_std", "row_neighbors", "column_neighbors",
    "distance", "row_rank", "row_excess", "column_excess",
) + tuple(
    f"{name}_{axis}" for name in (
        "kf", "short", "long", "short_uncertainty", "long_uncertainty",
        "velocity", "recent_velocity_difference", "long_velocity_difference",
    ) for axis in ("az", "el", "range")
)


@dataclass(slots=True)
class Track(cv.Track):
    history: deque = field(default_factory=lambda: deque(maxlen=SCANS))


def spawn(position, amplitude, circle):
    track = Track(position.copy(), np.zeros(3), circle, amplitude, position.copy())
    track.history.append((circle, amplitude, position.copy()))
    return track


def update(track, position, amplitude, circle, probability=1., robust=0.):
    if robust == 0 or track.hits == 1:
        cv.update_track(track, position, amplitude, circle)
    else:
        gap = circle-track.circle
        p, pv, v = track.covariance
        q = cv.PROCESS_NOISE
        p_prior = p+2*gap*pv+gap**2*v+q*gap**3/3
        pv_prior = pv+gap*v+q*gap**2/2
        v_prior = v+gap*q
        r = 1+robust*(1-probability)**2
        kp, kv = p_prior/(p_prior+r), pv_prior/(p_prior+r)
        prediction = track.position+gap*track.velocity
        innovation = cv.residual(position,prediction)
        track.position = prediction+kp*innovation
        track.position[0] %= 1.
        track.velocity += kv*innovation
        track.covariance[0] = (1-kp)*p_prior
        track.covariance[1] = (1-kp)*pv_prior
        track.covariance[2] = v_prior-kv*pv_prior
        track.circle = circle
        track.hits += 1
        track.mean_amplitude += (amplitude-track.mean_amplitude)/min(track.hits,8)/np.sqrt(r)
    # Weak assignments influence the short-term prediction without replacing
    # the history that distinguishes two established IDs during a crossing.
    if robust == 0 or probability >= .55 or len(track.history)<3:
        track.history.append((circle, amplitude, position.copy()))


def history_arrays(tracks):
    positions = np.zeros((len(tracks), SCANS, 3))
    times = np.zeros((len(tracks), SCANS))
    mask = np.zeros((len(tracks), SCANS), dtype=bool)
    amps = np.zeros((len(tracks), SCANS))
    for i, track in enumerate(tracks):
        observations = track.history
        n = len(observations)
        times[i, -n:] = [o[0] for o in observations]
        amps[i, -n:] = [o[1] for o in observations]
        positions[i, -n:] = [o[2] for o in observations]
        mask[i, -n:] = True
    return positions, times, amps, mask


def fit_predict(positions, times, mask, circle, velocity_fallback):
    center = positions[:, -1:, :]
    y = cv.residual(positions, center)
    t = times - times[:, -1:]
    n = mask.sum(axis=1)
    mean_t = (t * mask).sum(axis=1) / n
    mean_y = (y * mask[..., None]).sum(axis=1) / n[:, None]
    x = (t - mean_t[:, None]) * mask
    xx = (x ** 2).sum(axis=1)
    covariance = (x[..., None] * y).sum(axis=1)
    v = np.divide(covariance, xx[:, None], out=velocity_fallback.copy(), where=xx[:, None] > 0)
    relative_now = circle - times[:, -1]
    predicted = center[:, 0] + mean_y + (relative_now - mean_t)[:, None] * v
    error = (y - mean_y[:, None] - x[..., None] * v[:, None]) * mask[..., None]
    deviation = np.sqrt((error ** 2).sum(axis=1) / np.maximum(n[:, None]-2, 1))
    spread = deviation + NOISE
    forecast = 1 + 1 / n + (relative_now - mean_t) ** 2 / np.maximum(xx, 1)
    uncertainty = spread * np.sqrt(forecast[:, None])
    return predicted, v, uncertainty


def candidate_features(tracks, positions, amps, circle):
    elapsed = np.array([circle - t.circle for t in tracks])
    p = np.array([t.position for t in tracks])
    v = np.array([t.velocity for t in tracks])
    expected = p + v * elapsed[:, None]
    delta = cv.residual(positions[None, :, :], expected[:, None, :])
    gate = SEARCH * (1 + .35 * (elapsed - 1))[:, None]
    dist = ((delta / gate[:, None]) ** 2).sum(axis=2)
    r, c = np.nonzero(dist <= 1)
    if len(r) == 0:
        return r, c, np.empty((0, len(FEATURES)), dtype=np.float32)

    hp, ht, ha, hm = history_arrays(tracks)
    short, short_v, short_sigma = fit_predict(hp[:, -3:], ht[:, -3:], hm[:, -3:], circle, v)
    long, long_v, long_sigma = fit_predict(hp, ht, hm, circle, v)
    short_error = abs(cv.residual(positions[c], short[r]))
    long_error = abs(cv.residual(positions[c], long[r]))
    n = hm.sum(axis=1)
    amp_mean = ha.sum(axis=1) / n
    amp_std = np.sqrt(np.maximum((ha ** 2).sum(axis=1) / n - amp_mean ** 2, 0))
    edges = dist[r, c]
    row_count = np.bincount(r, minlength=len(tracks))
    column_count = np.bincount(c, minlength=len(positions))
    row_best = np.full(len(tracks), np.inf)
    np.minimum.at(row_best, r, edges)
    column_best = np.full(len(positions), np.inf)
    np.minimum.at(column_best, c, edges)
    order = np.lexsort((edges, r))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order)) - (row_count.cumsum()-row_count)[r[order]]
    y = np.column_stack((
        elapsed[r], np.array([t.hits for t in tracks])[r], amps[c], ha[r, -1], amp_mean[r],
        abs(amps[c]-amp_mean[r]), amp_std[r], row_count[r], column_count[c],
        edges, rank, edges-row_best[r], edges-column_best[c],
        abs(delta[r, c]) / NOISE, short_error / NOISE, long_error / NOISE,
        short_error / short_sigma[r], long_error / long_sigma[r],
        abs(v[r]) / NOISE, abs(v[r]-short_v[r]) / NOISE, abs(short_v[r]-long_v[r]) / NOISE,
    )).astype(np.float32)
    return r, c, y
