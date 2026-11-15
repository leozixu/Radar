"""Reproduce CV-KF decisions and score each output with its causal history."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from baseline_cv import FOLLOW_GATE, process_circle as baseline_circle, residual


HISTORY = 7
QUALITY_FEATURES = (
    "hits", "amp_running_mean",
    "speed_az", "speed_el", "speed_range",
    "span_per_hit_az", "span_per_hit_el", "span_per_hit_range",
    "amp_oldest", "amp_current", "amp_std", "amp_min", "amp_max",
    "current_error_az", "current_error_el", "current_error_range",
    "mean_error_az", "mean_error_el", "mean_error_range",
    "max_error_az", "max_error_el", "max_error_range",
    "current_gap", "mean_gap", "max_gap",
)


def decisions_with_features(points, state, track_points=None):
    """Return the tracker's decisions and matching rows of causal quality features."""
    history = state.setdefault("quality_history", {})

    def observe(track, position, amplitude, circle):
        observations = history.setdefault(id(track), deque(maxlen=HISTORY))
        gap = circle - track.circle
        error = np.abs(residual(position, track.position + track.velocity * gap) / FOLLOW_GATE)
        if not observations:
            observations.append((track.mean_amplitude, np.nan, np.nan, np.nan, np.nan))
        observations.append((amplitude, *error, gap))

    results = (baseline_circle if track_points is None else track_points)(points, state, observe)
    established, tentative, _ = state["tracks"]
    state["quality_history"] = {
        id(track): history[id(track)] for track in established + tentative if id(track) in history
    }
    if not results:
        return results, np.empty((0, len(QUALITY_FEATURES)), dtype=np.float32)

    by_batch = {track.batch_id: track for track in established}
    current = []
    oldest_amplitudes = []
    samples = np.full((len(results), HISTORY, 5), np.nan)
    for index, result in enumerate(results):
        track = by_batch[result["BatchID"]]
        observations = history[id(track)]
        span = np.abs(residual(track.position, track.first_position) / FOLLOW_GATE) / (track.hits - 1)
        current.append((track.hits, track.mean_amplitude, *np.abs(track.velocity / FOLLOW_GATE), *span))
        oldest_amplitudes.append(observations[0][0])
        samples[index, -len(observations):] = observations

    amp = samples[:, :, 0]
    error = samples[:, :, 1:4]
    gap = samples[:, :, 4]
    features = np.column_stack((
        current, oldest_amplitudes, amp[:, -1], np.nanstd(amp, axis=1),
        np.nanmin(amp, axis=1), np.nanmax(amp, axis=1),
        error[:, -1], np.nanmean(error, axis=1), np.nanmax(error, axis=1),
        gap[:, -1], np.nanmean(gap, axis=1), np.nanmax(gap, axis=1),
    ))
    return results, features.astype(np.float32)


@dataclass(frozen=True, slots=True)
class CompatibilityRuntime:
    quality_model: object
    threshold: float
    metadata: dict

    def process_circle(self, points, state):
        results, features = decisions_with_features(points, state)
        if not results:
            return results
        probabilities = self.quality_model.predict(features, num_threads=1)
        return [result for result, probability in zip(results, probabilities)
                if probability >= self.threshold]
