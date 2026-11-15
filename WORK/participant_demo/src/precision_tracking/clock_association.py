"""Radar scan-phase features shared by the precision tracker models."""

import numpy as np

PERIOD_MS = 5000 / 3
FEATURES = ('phase_abs', 'alignment_fraction', 'amplitude')


def wrap(value):
    return (value + 0.5) % 1.0 - 0.5


def phase_features(azimuth, timestamps, amplitudes):
    """Center phase using the densest observed band in the current scan."""
    phase = wrap(azimuth - timestamps / PERIOD_MS)
    counts = np.bincount(np.rint(500 * (phase + 0.5)).astype(int) % 500, minlength=500)
    peak = np.argmax(counts + np.roll(counts, 1) + np.roll(counts, -1))
    centered = wrap(phase - (peak / 500 - 0.5))
    inliers = np.abs(centered) < 0.004
    share = inliers.mean()
    if share >= 0.3:
        error = np.abs(wrap(centered - np.median(centered[inliers])))
    else:
        error = np.full_like(phase, np.nan)
    return np.column_stack((error, np.full(len(phase), share), amplitudes))
