"""Select original points to publish with trusted ownership evidence."""

from collections import deque

import numpy as np

from baseline_cv import FOLLOW_GATE, residual
import tracker_compat

from .clock_association import phase_features


HISTORY = 10


FEATURES = tracker_compat.QUALITY_FEATURES + (
    "legacy_quality", "link_probability", "clock_probability", "scan_phase_error",
    "snr", "previous_proposals", "trusted_proposals", "previous_gap", "trusted_gap",
    "recent_link_mean", "recent_clock_mean", "recent_quality_mean", "amplitude_deviation",
) + tuple(f"{group}_{axis}" for group in ("step", "trusted_forecast", "recent_forecast")
          for axis in ("azimuth", "elevation", "range"))


class EmissionFeatures:
    def __init__(self):
        self.history = {}

    @staticmethod
    def forecast_error(past, position, circle):
        if len(past) < 2:
            return np.full(3, np.nan)
        older, previous = past[-2:]
        velocity = residual(previous[1], older[1]) / (previous[0] - older[0])
        predicted = previous[1] + velocity * (circle - previous[0])
        return np.abs(residual(position, predicted)) / FOLLOW_GATE

    def build(self, points, proposals, quality, probabilities, state, clock):
        live = {track.batch_id for track in state["tracks"][0]}
        self.history = {batch: entries for batch, entries in self.history.items() if batch in live}
        if not proposals:
            return np.empty((0, len(FEATURES)), dtype=np.float32)
        measurements = [(p["Azimuth"], p["Time"], p["Amp"]) for p in points]
        phase = phase_features(*np.array(measurements).T)
        scan_probability = clock.predict(phase, num_threads=1)
        indices = {p["PntID"]: i for i, p in enumerate(points)}
        associations = state["association_confidence"]
        extra = []
        for result, probability in zip(proposals, probabilities):
            ident = result["PntID"]
            index = indices[ident]
            point = points[index]
            past = self.history.setdefault(result["BatchID"], deque(maxlen=HISTORY))
            circle, amplitude = point["CircleID"], point["Amp"]
            position = np.array((point["Azimuth"], point["Elevation"], point["Distance"]))
            link = associations.get(ident, np.nan)
            phase_prob = scan_probability[index]
            trusted = [row for row in past if row[3] >= .80 and row[4] >= .50]
            if past:
                previous_gap = circle - past[-1][0]
                step = np.abs(residual(position, past[-1][1])) / (FOLLOW_GATE * previous_gap)
                prior_links = [row[3] for row in past if np.isfinite(row[3])]
                mean_link = np.mean(prior_links) if prior_links else np.nan
                mean_phase = np.mean([row[4] for row in past])
                mean_quality = np.mean([row[5] for row in past])
                amp_change = abs(amplitude - np.median([row[2] for row in past]))
            else:
                previous_gap = mean_link = mean_phase = mean_quality = amp_change = np.nan
                step = np.full(3, np.nan)
            trusted_gap = circle - trusted[-1][0] if trusted else np.nan
            extra.append((probability, link, phase_prob, phase[index, 0], point["SNR"], len(past),
                          len(trusted), previous_gap, trusted_gap, mean_link, mean_phase, mean_quality,
                          amp_change, *step, *self.forecast_error(trusted, position, circle),
                          *self.forecast_error(list(past), position, circle)))
            past.append((circle, position, amplitude, link, phase_prob, probability))
        return np.column_stack((quality, extra)).astype(np.float32)


def current_raw_probabilities(points, proposals, state):
    source_q = state.get("raw_return_probability")
    if source_q is None:
        return None
    scores_by_point = dict(zip((p["PntID"] for p in points), source_q))
    return np.fromiter((scores_by_point[p["PntID"]] for p in proposals), dtype=float)


def accepted_emissions(baseline, detection, ownership, young, thresholds, raw_probability=None, raw_floor=None):
    reject, rescue, detect_floor = thresholds
    confident_owner = (ownership >= .80) | young
    keep = baseline & ((ownership >= reject) | young)
    keep |= (detection >= rescue) & confident_owner
    if raw_floor is not None:
        keep |= (raw_probability >= raw_floor) & confident_owner
    return keep & (detection >= detect_floor)
