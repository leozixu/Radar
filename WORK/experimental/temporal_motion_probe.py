"""Oracle probe for timestamp and autoregressive motion evidence on clean histories."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from train_tracker import read_packet
from teacher_assoc import cv


WINDOW = 8
LAGS = 12


def local_prediction(time, measurement):
    # The final sample is the candidate time only; its position is withheld from the fit.
    history, target_time = time[:, :-1], time[:, -1]
    center = history - history.mean(axis=1)[:, None]
    variation = np.sum(center**2, axis=1)
    averages = measurement.mean(axis=1)
    slope = np.divide(np.einsum("nw,nwd->nd", center, measurement), variation[:, None],
                      out=np.zeros_like(averages), where=variation[:, None] > 0)
    since_mean = target_time - history.mean(axis=1)
    return averages + since_mean[:, None] * slope


def kf_errors(position):
    """One-scan innovation of the existing CV filter with clean prior assignments."""
    errors = np.zeros_like(position)
    state = position[1].copy()
    velocity = position[1] - position[0]
    variance, link, prior_velocity = np.ones(3), np.ones(3), np.full(3, 2.)
    for i, measurement in enumerate(position[2:], 2):
        innovation = measurement - (state + velocity)
        errors[i] = innovation
        p = variance + 2 * link + prior_velocity + cv.PROCESS_NOISE / 3
        pv = link + prior_velocity + cv.PROCESS_NOISE / 2
        innovation_variance = 1 + p
        variance, link = p / innovation_variance, pv / innovation_variance
        prior_velocity = prior_velocity + cv.PROCESS_NOISE - link * pv
        state = state + velocity + variance * innovation
        velocity = velocity + link * innovation
    return errors


def sequences(path):
    data = read_packet(path)
    points = data[data[:, 8] > 0]
    order = np.lexsort((points[:, 6], points[:, 8]))
    tracks = points[order]
    cuts = np.flatnonzero((np.diff(tracks[:, 8]) != 0) | (np.diff(tracks[:, 6]) != 1)) + 1
    for tracklet in np.split(tracks, cuts):
        if len(tracklet) <= WINDOW + LAGS:
            continue
        time = tracklet[:, 0].astype(float) / 5000
        time -= time[0]
        dt = np.diff(time)
        valid = (dt > .25) & (dt < 1.75)
        time = np.r_[0., np.cumsum(np.where(valid, dt, 1.))]
        position = tracklet[:, 1:4].astype(np.float32).astype(float)
        position[:, 0] = np.unwrap(position[:, 0], period=1.)
        windows = np.moveaxis(sliding_window_view(position, WINDOW + 1, axis=0), 2, 1)
        clock = sliding_window_view(time, WINDOW + 1)
        scans = np.broadcast_to(np.arange(WINDOW + 1), clock.shape)
        reference = windows[:, -1]
        prev = windows[:, :-1]
        scan_prediction = local_prediction(scans, prev)
        clock_prediction = local_prediction(clock, prev)
        short_prediction = local_prediction(scans[:, -4:], prev[:, -3:])
        yield reference - scan_prediction, reference - clock_prediction, reference - short_prediction, kf_errors(position)[WINDOW:]


def collect(paths):
    result = []
    for path in paths:
        rows = list(sequences(path))
        print(path.stem, len(rows), sum(len(a) for a, *_ in rows), flush=True)
        result.extend(rows)
    return result


def ar_design(error):
    view = np.moveaxis(sliding_window_view(error, LAGS + 1, axis=0), 2, 1)
    return view[:, :-1], view[:, -1]


def train_autoregressive(eligible):
    cases = [ar_design(error) for error, *_ in eligible]
    inputs = np.concatenate([history for history, _ in cases])
    targets = np.concatenate([target for _, target in cases])
    prediction_models = []
    for axis in range(3):
        x, y = inputs[:, :, axis], targets[:, axis]
        ridge = np.trace(x.T @ x) / LAGS * .003
        prediction_models.append(np.linalg.solve(x.T @ x + np.eye(LAGS) * ridge, x.T @ y))
    return np.stack(prediction_models, axis=1)


def summaries(error):
    return {"observations": len(error), "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "median_abs": np.median(np.abs(error), axis=0).tolist(),
            "p95_abs": np.quantile(np.abs(error), .95, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=int, nargs="+", required=True)
    parser.add_argument("--probe", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if set(args.fit) & set(args.probe) or args.report.suffix.lower() != ".json":
        parser.error("probe packets must be disjoint from fit packets, with a JSON report")
    fit = packet_paths(DEFAULT_DATA_DIR, args.fit, TRAIN_PACKETS)
    probe = packet_paths(DEFAULT_DATA_DIR, args.probe, TRAIN_PACKETS)
    autoregressive = train_autoregressive(collect(fit))
    by_phase = {"scan_linear": [], "time_linear": [], "scan_linear_short": [], "cv_kf": [], "past_residual_linear": []}
    for scans, clock, short, kf in collect(probe):
        design, target = ar_design(scans)
        by_phase["scan_linear"].append(target)
        by_phase["time_linear"].append(clock[LAGS:])
        by_phase["scan_linear_short"].append(short[LAGS:])
        by_phase["cv_kf"].append(kf[LAGS:])
        by_phase["past_residual_linear"].append(target - np.einsum("nld,ld->nd", design, autoregressive))
    result = {
        "fit_packets": args.fit, "probe_packets": args.probe, "axes": ["azimuth", "elevation", "distance"],
        "lags": LAGS, "window": WINDOW, "autoregressive_coefficients": autoregressive.tolist(),
        "results": {name: summaries(np.concatenate(errors)) for name, errors in by_phase.items()},
        "limit": "Uses earlier true observations without wrong assignments and scores only continuous mature histories."
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
