"""Reconnect established pairs over larger motion while preserving their identities."""

from collections import Counter

import numpy as np

from baseline_cv import residual

from .causal_birth_paths import POSITION, SCAN
from .fast_pair_center import FastPairCenter
from .paired_returns import MAX_GAP, PairReturns, pair_offset

ORDINARY_WIDTH = np.array((.038, .140, .020))
BROAD_WIDTH = np.array((.105, .280, .036))
FORECAST_WIDTH = np.array((.058, .160, .021))
ROLE_CHANGE_WIDTH = np.array((.012, .004, .0035))
AMPLITUDE_ERROR = .0000026
PROBABILITY = .97


def compatible_recoveries(frame, tracks, raw, circle, prior_pairs):
    """Return only exclusive wide-motion links that preserve the original frame."""
    if not frame.strict_count or not prior_pairs:
        return []
    taken_pairs = {id(prior) for prior, _ in frame.decisions}
    taken_tracks = {row for row, _, _ in frame.locked}
    taken_points = {point for _, index in frame.decisions for point in frame.columns[index]}
    taken_points.update(point for _, point, _ in frame.locked)
    active = {track.batch_id: index for index, track in enumerate(tracks) if index not in taken_tracks}
    registered = [prior for prior in prior_pairs
                  if prior.confirmed and id(prior) not in taken_pairs
                  and 0 < circle - prior.observations[0, SCAN] <= MAX_GAP
                  and prior.owners[0] != prior.owners[1] and all(owner in active for owner in prior.owners)]
    if not registered:
        return []
    columns = np.asarray([index for index, pair in enumerate(frame.columns[:frame.strict_count])
                          if not any(point in taken_points for point in pair)], dtype=int)
    if not len(columns):
        return []
    previous = np.stack([prior.observations for prior in registered])
    current = raw[frame.columns[columns]]
    old_amp, new_amp = pair_offset(previous), pair_offset(current)
    old_role = residual(previous[:, 1, POSITION], previous[:, 0, POSITION])
    new_role = residual(current[:, 1, POSITION], current[:, 0, POSITION])
    rows, cols = np.nonzero(
        (np.abs(old_amp[:, None] - new_amp[None, :]) <= AMPLITUDE_ERROR)
        & (old_amp[:, None] < 0.) & (new_amp[None, :] < 0.)
        & (old_role[:, None, 1] >= -1e-7) & (new_role[None, :, 1] >= -1e-7)
        & (np.abs(new_role[None, :, 2]) <= .0038))
    if not len(rows):
        return []
    elapsed = circle - previous[rows, 0, SCAN]
    observed = current[cols, :, POSITION]
    motion = np.abs(residual(observed, previous[rows, :, POSITION]).mean(axis=1))
    width = (1. + .60 * (elapsed - 1.))[:, None]
    shape = np.abs(residual(new_role[cols], old_role[rows]))
    viable = (~(motion <= ORDINARY_WIDTH * width).all(axis=1) & (motion <= BROAD_WIDTH * width).all(axis=1)
              & (shape <= ROLE_CHANGE_WIDTH).all(axis=1))
    rows, cols, observed = rows[viable], cols[viable], observed[viable]
    if not len(rows):
        return []
    choices = []
    for index, col, measured in zip(rows, cols, observed):
        provenance = registered[index]
        owners = tuple(active[owner] for owner in provenance.owners)
        tracks_for_pair = [tracks[owner] for owner in owners]
        lag = max(circle - track.circle for track in tracks_for_pair)
        prediction = np.stack([track.position + (circle - track.circle) * track.velocity
                               for track in tracks_for_pair])
        forecast_error = np.abs(residual(measured, prediction).mean(axis=0))
        if (forecast_error <= FORECAST_WIDTH * (1. + .35 * max(0, lag - 1))).all():
            choices.append((provenance, int(columns[col]), owners))
    row_counts = Counter(owner for _, _, owners in choices for owner in owners)
    point_counts = Counter(point for _, index, _ in choices for point in frame.columns[index])
    return [(prior, index, owners) for prior, index, owners in choices
            if all(row_counts[owner] == 1 for owner in owners)
            and all(point_counts[point] == 1 for point in frame.columns[index])]


class RegisteredPairRecovery(PairReturns):
    """Reconnect live pair registrations that exceeded the ordinary motion gate."""

    def begin(self, tracks, raw, times, circle, state):
        frame = super().begin(tracks, raw, times, circle, state)
        for prior, index, owners in compatible_recoveries(frame, tracks, raw, circle, state.get("previous_pairs", ())):
            points = frame.columns[index]
            frame.decisions.append((prior, index))
            frame.locked.extend((owner, int(point), PROBABILITY) for owner, point in zip(owners, points))
        return frame


class BothRecoveries(FastPairCenter, RegisteredPairRecovery):
    """Reconnect registered pairs before admitting previously unowned pairs."""
