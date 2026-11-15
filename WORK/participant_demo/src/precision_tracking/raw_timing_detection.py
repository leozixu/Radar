"""Per-return confidence from neighbouring radar measurements in the current scan."""

import numpy as np

from .clock_association import phase_features


FEATURES = ("amplitude", "scan_phase", "phase_alignment", "same_tick_count", "near_tick_count") + tuple(
    f"{neighbour}_{part}"
    for neighbour in ("previous", "previous_2", "next", "next_2")
    for part in ("elapsed", "azimuth_delta", "elevation_delta", "range_delta", "amplitude", "amplitude_difference")
)


class RawReturnGate:
    def __init__(self, model, floor):
        self.model = model
        self.floor = floor

    def probabilities(self, positions, amplitude, times):
        sample = features(positions, amplitude, times)
        # An unaligned or malformed time stream falls back to the regular tracker.
        if len(sample) < 5 or sample[0, 2] < .30 or (sample[:, 3] == 1).mean() < .85:
            return np.full(len(sample), np.nan)
        return self.model.predict(sample, num_threads=1)

    def __call__(self, positions, amplitude, times):
        probability = self.probabilities(positions, amplitude, times)
        return np.isnan(probability) | (probability >= self.floor)


def features(positions, amplitude, time):
    """Use time spacing and source measurements from the available current scan."""
    count = len(time)
    if count == 0:
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    order = np.argsort(time, kind="stable")
    inverse = np.empty(count, dtype=int)
    inverse[order] = np.arange(count)
    positions, amplitude, time = positions[order], amplitude[order], time[order]
    phase = phase_features(positions[:, 0], time, amplitude)
    duplicate = np.searchsorted(time, time, side="right") - np.searchsorted(time, time, side="left")
    local = np.searchsorted(time, time + 8, side="right") - np.searchsorted(time, time - 8, side="left")
    parts = [np.column_stack((amplitude, phase[:, :2], duplicate, local))]
    indices = np.arange(count)
    for step in (-1, -2, 1, 2):
        neighbor = indices + step
        present = (neighbor >= 0) & (neighbor < count)
        neighbor = np.clip(neighbor, 0, count - 1)
        elapsed = abs(time[neighbor] - time)
        delta = positions - positions[neighbor]
        delta[:, 0] = (delta[:, 0] + .5) % 1. - .5
        other_amp = amplitude[neighbor]
        value = np.column_stack((elapsed, delta, other_amp, amplitude - other_amp))
        value[~present] = np.nan
        parts.append(value)
    return np.column_stack(parts)[inverse].astype(np.float32)
