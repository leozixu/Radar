"""Fit causal relative elevation and identify reliable association edges."""

import numpy as np


WINDOW = 10
HISTORY = np.arange(-WINDOW, 0, dtype=float)


def fit_relative_elevation(history, anchors):
    """Fit prior elevation differences on common scans, then extrapolate one step."""
    differences = history[:, None, :] - anchors[None, :, :]
    mask = np.isfinite(differences)
    delta = np.where(mask, differences, 0.)
    count = mask.sum(axis=2)
    x_sum = (mask * HISTORY).sum(axis=2)
    x2_sum = (mask * HISTORY**2).sum(axis=2)
    y_sum = delta.sum(axis=2)
    xy_sum = (delta * HISTORY).sum(axis=2)
    denominator = np.maximum(count * x2_sum - x_sum**2, 1.)
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    projected = (y_sum - slope * x_sum) / np.maximum(count, 1)
    residual = np.where(mask, delta - projected[..., None] - slope[..., None] * HISTORY, 0.)
    spread = np.sqrt((residual**2).sum(axis=2) / np.maximum(count - 2, 1))
    recent = mask[..., -3:].any(axis=2)
    return projected, spread, count, recent


def recent_elevation(tracks, circle):
    history = np.full((len(tracks), WINDOW), np.nan)
    for row, track in enumerate(tracks):
        for scan, _, observation in track.history:
            slot = WINDOW + scan - circle
            if 0 <= slot < WINDOW:
                history[row, slot] = observation[1]
    return history


def reliable_anchors(rows, cols, probabilities, selected, min_probability, margin, n_tracks, n_points):
    """A current match must beat other points for its row and other rows for its point."""
    assigned_row = np.full(n_tracks, -1)
    assigned_col = np.full(n_points, -1)
    assigned_row[rows[selected]] = selected
    assigned_col[cols[selected]] = selected
    edge = np.arange(len(rows))
    row_alternatives = assigned_row[rows] != edge
    col_alternatives = assigned_col[cols] != edge
    row_next = np.zeros(n_tracks)
    col_next = np.zeros(n_points)
    np.maximum.at(row_next, rows[row_alternatives], probabilities[row_alternatives])
    np.maximum.at(col_next, cols[col_alternatives], probabilities[col_alternatives])
    confidence = probabilities[selected]
    alternatives = np.maximum(row_next[rows[selected]], col_next[cols[selected]])
    return selected[(confidence >= min_probability) & (confidence - alternatives >= margin)]
