"""Check which causal measurement signatures survive when pair locks do not apply."""

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np

import teacher_assoc as teacher
from clock_association import PERIOD_MS, wrap
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from train_tracker import read_packet


FINGERPRINTS = ('phase', 'amplitude', 'range_linear', 'range_quadratic', 'range_robust', 'azimuth_clock')


def scan_phase(azimuth, stamp):
    phase = wrap(azimuth - stamp / PERIOD_MS)
    histogram = np.bincount(np.rint(500 * (phase + .5)).astype(int) % 500, minlength=500)
    mode = np.argmax(histogram + np.roll(histogram, 1) + np.roll(histogram, -1))
    phase = wrap(phase - (mode / 500 - .5))
    support = np.abs(phase) <= .004
    return wrap(phase - np.median(phase[support])) if support.mean() >= .3 else np.full(len(phase), np.nan)


def predict_range(times, values, scan):
    """Extrapolate a bounded robust affine/quadratic fit for each target range."""
    mask = np.isfinite(values)
    count = mask.sum(axis=1)
    x = np.where(mask, times - scan, 0.)
    y = np.where(mask, values - values[:, -1:], 0.)
    linear = np.stack((mask, x), axis=-1)
    quadratic = np.concatenate((linear, (x*x)[..., None]), axis=-1)
    penalty = np.diag((0., 0., 80.))
    x_t = linear.transpose(0, 2, 1)
    q_t = quadratic.transpose(0, 2, 1)
    beta = np.linalg.solve(x_t @ linear, x_t @ y[..., None])
    quad = np.linalg.solve(q_t @ quadratic + penalty, q_t @ y[..., None])
    error = y - (linear @ beta)[..., 0]
    scale = .00035 + np.sqrt((error**2).sum(axis=1) / np.maximum(count-2, 1))
    weight = mask / (1.+(error / (2.5*scale[:, None]))**2)
    robust = np.linalg.solve((x_t * weight[:, None]) @ linear, (x_t * weight[:, None]) @ y[..., None])
    shift = values[:, -1]
    return np.column_stack((beta[:, 0, 0] + shift, quad[:, 0, 0] + shift, robust[:, 0, 0]+shift)), scale


def history_signatures(histories, circle):
    depth = 12
    train = np.full((len(histories), depth, 5), np.nan)
    for index, history in enumerate(histories):
        chosen = list(history)[-depth:]
        train[index, -len(chosen):] = chosen
    t, phase, amp, ranges, azimuth_clock = np.moveaxis(train, -1, 0)
    prediction, spread = predict_range(t, ranges, circle)
    with np.errstate(invalid='ignore'):
        signed_phase = np.ma.median(np.ma.masked_invalid(phase), axis=1).filled(np.nan)
    center = azimuth_clock[:, -1:]
    azimuth_clock = np.where(np.isfinite(azimuth_clock), center + wrap(azimuth_clock-center), np.nan)
    clock_prediction, _ = predict_range(t, azimuth_clock, circle)
    return (np.column_stack((signed_phase, np.nanmedian(amp, axis=1), prediction, clock_prediction[:, 0])), spread)


def candidate_errors(observed, predictions, spread):
    delta = observed - predictions
    delta[:, -1] = wrap(delta[:, -1])
    delta[:, 2:5] /= spread[:, None]
    return np.abs(delta)


def summarize(path, reference):
    data = read_packet(path)
    tracks, histories = {}, defaultdict(lambda: deque(maxlen=12))
    counts = defaultdict(Counter)
    errors = defaultdict(list)
    comparison = defaultdict(list)
    past_offsets = np.empty(0)
    for scan in np.split(data, np.flatnonzero(np.diff(data[:, 6])) + 1):
        circle = int(scan[0, 6])
        positions = scan[:, 1:4].astype('f').astype(float)
        amp = scan[:, 4].astype('f').astype(float)
        stamps, labels = scan[:, (0, 8)].astype(np.int64).T
        phase = scan_phase(positions[:, 0], stamps)
        clock_azimuth = wrap(stamps / PERIOD_MS)
        order = np.argsort(stamps, kind='stable')
        a, b = order[:-1], order[1:]
        delta = amp[b] - amp[a]
        pair = (stamps[b] - stamps[a] == 1) & (np.abs(delta) >= .000004) & (np.abs(delta) <= .032)
        pair &= (np.abs(teacher.cv.residual(positions[a], positions[b])) <= [.018, .012, .008]).all(axis=1)
        offsets = np.sort(delta[pair])
        if len(past_offsets):
            matches = np.diff(np.searchsorted(past_offsets, delta[:, None] + [-.0000031, .0000031]), axis=1)[:, 0] > 0
        else:
            matches = np.zeros(len(delta), dtype=bool)
        repeated = np.zeros(len(scan), dtype=bool)
        repeated[a[pair & matches]] = repeated[b[pair & matches]] = True
        past_offsets = offsets

        active = {label: tracks[label] for label in labels if label > 0 and label in tracks
                  and 0 < circle - tracks[label].circle <= 7 and len(histories[label]) >= 5}
        if active:
            owners = np.array(list(active.keys()))
            rows, cols, features = teacher.candidate_features(list(active.values()), positions, amp, circle)
            teacher_scores = reference.predict(features, num_threads=1) if len(rows) else np.empty(0)
            prediction, spread = history_signatures([histories[label] for label in owners], circle)
            observed = np.column_stack((phase, amp, *([positions[:, 2]]*3), clock_azimuth))
            error = candidate_errors(observed[cols], prediction[rows], spread[rows])
            cuts = np.r_[0, np.searchsorted(rows, np.arange(1, len(owners))), len(rows)]
            for start, stop in zip(cuts[:-1], cuts[1:]):
                if stop-start < 2:
                    continue
                x = np.arange(start, stop)
                positive = x[labels[cols[x]] == owners[rows[start]]]
                if len(positive) != 1:
                    continue
                p = positive[0]
                pairing = 'repeat_pair' if repeated[cols[p]] else 'single'
                for name, valid in (('all', labels[cols[x]] >= 0), ('positive', labels[cols[x]] > 0)):
                    eligible = x[valid & (x != p) & (teacher_scores[x] >= .10)]
                    if not len(eligible):
                        continue
                    n = eligible[np.argmax(teacher_scores[eligible])]
                    key = pairing+'_'+name
                    counts[key]['comparisons'] += 1
                    counts[key]['teacher_correct'] += int(teacher_scores[p] > teacher_scores[n])
                    errors[key].append(np.vstack((error[p], error[n])))
                    comparison[key].append([teacher_scores[p], teacher_scores[n]])
        for i, label in enumerate(labels):
            if label <= 0:
                continue
            if label not in tracks or circle - tracks[label].circle > 7:
                tracks[label] = teacher.spawn(positions[i], amp[i], circle)
                histories[label].clear()
            elif circle > tracks[label].circle:
                teacher.update(tracks[label], positions[i], amp[i], circle)
            else:
                continue
            histories[label].append((circle, phase[i], amp[i], positions[i, 2], clock_azimuth[i]))
    detail = {}
    for key, count in counts.items():
        e = np.stack(errors[key])
        teacher_score = np.stack(comparison[key])
        signal = {}
        for j, name in enumerate(FINGERPRINTS):
            measured = np.isfinite(e[:, :, j]).all(axis=1)
            truth, competitor = e[measured, :, j].T
            correct = truth < competitor
            signal[name] = dict(support=int(measured.sum()), correct=int(correct.sum()),
                                median_error=np.median(truth).item(), median_negative=np.median(competitor).item(),
                                teacher_wrong=int((teacher_score[measured, 0] < teacher_score[measured, 1]).sum()),
                                recovered=int((correct & (teacher_score[measured, 0] < teacher_score[measured, 1])).sum()))
        detail[key] = dict(counts=dict(count), signal=signal)
    return detail


def main():
    from lightgbm import Booster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', nargs='+', type=int, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
    model = Booster(model_file=str(args.teacher))
    result = {}
    for path in paths:
        result[path.stem] = summarize(path, model)
        print(path.stem, {key: item['counts'] for key, item in result[path.stem].items()}, flush=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
