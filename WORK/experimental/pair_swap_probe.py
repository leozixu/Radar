"""Train a relative-motion discriminator for reciprocal association swaps."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from teacher_assoc import FEATURES, NOISE, Online, candidate_features, spawn, update
import teacher_assoc
from algorithm import load_runtime
from baseline_cv import residual
import baseline_cv
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
import replay_validation as replay
from train_tracker import read_packet
from tracker_assoc import assign
import tracker_compat


PAIR_NAMES = tuple(f'{role}_{name}' for role in ('left', 'right', 'left_cross', 'right_cross')
                   for name in FEATURES)
PAIR_NAMES += ('direct_log_probability', 'cross_log_probability', 'common_past_scans')
PAIR_NAMES += tuple(f'{kind}_{axis}' for kind in ('difference', 'expected_difference', 'spread',
                                                    'relative_velocity', 'error', 'cross_error')
                   for axis in ('az', 'el', 'range'))


def relative_history(left, right, circle):
    earlier = {scan: pos for scan, _, pos in left.history}
    overlap = [(scan, earlier[scan], pos) for scan, _, pos in right.history if scan in earlier]
    if not overlap:
        delta = residual(left.position, right.position)
        velocity = left.velocity - right.velocity
        return 0, delta + velocity * max(1, circle - max(left.circle, right.circle)), NOISE, velocity
    t, a, b = map(np.asarray, zip(*overlap))
    relative_positions = residual(a, b)
    mean = relative_positions.mean(axis=0)
    center = t - t.mean()
    velocity = ((relative_positions - mean) * center[:, None]).sum(axis=0) / max(1., np.dot(center, center))
    expected = mean + (circle - t.mean()) * velocity
    return len(t), expected, (relative_positions - mean - center[:, None] * velocity).std(axis=0), velocity


def make_features(tracks, current, individual, index, confidence, left, right, a, b, circle):
    i, j, k, m = index[left, a], index[right, b], index[left, b], index[right, a]
    direct, crossed = confidence[i] + confidence[j], confidence[k] + confidence[m]
    count, mean, spread, velocity = relative_history(tracks[left], tracks[right], circle)
    delta = residual(current[a], current[b])
    context = np.concatenate((mean, spread, velocity)) / np.tile(NOISE, 3)

    def arrangement(first, second, left_cross, right_cross, good, alt, change):
        errors = np.r_[np.abs(change - mean), np.abs(-change - mean)] / np.tile(NOISE, 2)
        return np.r_[individual[[first, second, left_cross, right_cross]].ravel(), good, alt,
                     count, change / NOISE, context, errors]

    return (arrangement(i, j, k, m, direct, crossed, delta),
            arrangement(k, m, i, j, crossed, direct, -delta), direct - crossed)


def process(path, teacher, pair=None):
    data = read_packet(path)
    cuts = np.r_[0, np.flatnonzero(np.diff(data[:, 6])) + 1, len(data)]
    previous = {}
    samples = []
    totals = np.zeros(4, dtype=np.int64)
    for start, stop in zip(cuts, cuts[1:]):
        scan = data[start:stop]
        circle = int(scan[0, 6])
        position = scan[:, 1:4].astype(np.float32).astype(float)
        amp = scan[:, 4].astype(np.float32).astype(float)
        ids = scan[:, 8].astype(int)
        truth_columns = {key: j for j, key in enumerate(ids) if key > 0}
        active = [(track, truth_columns[key]) for key, track in previous.items() if key in truth_columns and
                  0 < circle - track.circle <= 5 and track.hits >= 3]
        if len(active) >= 2:
            tracks, true_columns = zip(*active)
            rows, cols, features = candidate_features(tracks, position, amp, circle)
            index = np.full((len(active), len(position)), -1, dtype=int)
            index[rows, cols] = np.arange(len(rows))
            possible = index[:, true_columns] >= 0
            valid = possible.diagonal()
            reciprocal = np.triu(possible & possible.T & valid[:, None] & valid[None, :], 1)
            pair_tracks = np.argwhere(reciprocal)
            if len(pair_tracks):
                log_p = np.log(np.clip(teacher.predict(features, num_threads=1), 1e-6, 1.))
                both = [make_features(tracks, position, features, index, log_p, i, j,
                                      true_columns[i], true_columns[j], circle) for i, j in pair_tracks]
                direct, crossed, delta = map(np.asarray, zip(*both))
                if pair is None:
                    samples.append(np.stack((direct, crossed), axis=1).astype('f'))
                else:
                    q = pair.predict(np.vstack((direct, crossed)), num_threads=1)
                    correct = q[:len(both)] > q[len(both):]
                    legacy = delta > 0
                    totals += (len(both), legacy.sum(), correct.sum(), ((~legacy) & correct).sum())
        for j, key in enumerate(ids):
            if key <= 0:
                continue
            track = previous.get(key)
            if track and 0 < circle - track.circle <= 5:
                update(track, position[j], amp[j], circle)
            else:
                previous[key] = spawn(position[j], amp[j], circle)
    if pair is not None:
        return totals
    return np.concatenate(samples) if samples else np.empty((0, 2, len(PAIR_NAMES)), dtype=np.float32)


class PairCorrector:
    def __init__(self, model, cutoff):
        self.model = model
        self.cutoff = cutoff
        self.swaps = 0

    def associate(self, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity):
        if not tracks or not len(positions):
            return [], np.empty(0, dtype=int)
        rows, cols, x = candidate_features(tracks, positions, amplitudes, circle)
        if not len(rows):
            return [], np.empty(0, dtype=int)
        probabilities = model.predict(x, num_threads=1)
        log_p = np.log(np.clip(probabilities, 1e-6, 1.))
        chosen, competing = assign(rows, cols, log_p - np.log(floor) - mix * x[:, 10],
                                    np.ones(len(tracks), dtype=bool), len(tracks), len(positions), ambiguity)
        selected = np.asarray(chosen, dtype=int)
        held = {int(cols[i]) for candidates in competing.values() for i in candidates}
        left, right = np.triu_indices(len(selected), 1)
        kept_rows, kept_cols = rows[selected], cols[selected]
        qualified = np.asarray([len(track.history) >= 4 for track in tracks])[kept_rows]
        left, right = left[qualified[left] & qualified[right]], right[qualified[left] & qualified[right]]
        repaired = set()
        if len(left):
            index = np.full((len(tracks), len(positions)), -1, dtype=np.int32)
            index[rows, cols] = np.arange(len(rows))
            cross1 = index[kept_rows[left], kept_cols[right]]
            cross2 = index[kept_rows[right], kept_cols[left]]
            acceptable = ((cross1 >= 0) & (cross2 >= 0) &
                          (probabilities[cross1] >= floor / 2) & (probabilities[cross2] >= floor / 2))
            pairs = []
            for a, b in zip(left[acceptable], right[acceptable]):
                row_a, row_b = kept_rows[a], kept_rows[b]
                col_a, col_b = kept_cols[a], kept_cols[b]
                features = make_features(tracks, positions, x, index, log_p,
                                         row_a, row_b, col_a, col_b, circle)
                if features[0][4 * len(FEATURES) + 2] >= 3:
                    pairs.append((a, b, features[0], features[1]))
            if pairs:
                features = np.asarray([x for _, _, primary, other in pairs for x in (primary, other)])
                prediction = self.model.predict(features, num_threads=1).reshape(-1, 2)
                # Require both orientations to agree and repair disjoint pairs only.
                confidence = np.minimum(1. - prediction[:, 0], prediction[:, 1])
                for k in np.argsort(-confidence, kind='stable'):
                    if confidence[k] < self.cutoff:
                        break
                    a, b = pairs[k][:2]
                    if a in repaired or b in repaired:
                        continue
                    selected[a] = index[kept_rows[a], kept_cols[b]]
                    selected[b] = index[kept_rows[b], kept_cols[a]]
                    repaired.update((a, b))
                    self.swaps += 1
        matches = [(int(rows[i]), int(cols[i]), float(max(.60, probabilities[i]) if j in repaired
                                                      else probabilities[i]))
                   for j, i in enumerate(selected)]
        return matches, np.fromiter(held, dtype=int)


def replay_worker(spec):
    path, teacher_path, pair_path, cutoff, ambiguity, *extra = spec
    from lightgbm import Booster

    teacher = Online(Booster(model_file=str(teacher_path)), floor=.20, mix=0., coast=7,
                     ambiguity=ambiguity, robust=4.)
    corrector = PairCorrector(Booster(model_file=str(pair_path)), cutoff)
    quality = load_runtime()
    with ExitStack() as context:
        context.enter_context(patch.object(teacher_assoc, 'assign_confirmed', corrector.associate))
        context.enter_context(patch.object(tracker_compat, 'baseline_circle', teacher.track))
        context.enter_context(patch.object(replay, 'load_handler', return_value=quality.process_circle))
        if extra and extra[0] is not None:
            from young_assoc import YoungAssociation

            seed = Booster(model_file=str(extra[0]))
            young = YoungAssociation({'seed': seed}, {'seed': .12}, geometric_mix=.15)
            context.enter_context(patch.object(baseline_cv, 'associate', young))
        statistics = replay.replay_packet(path, allowed=TUNING_PACKETS)
    print(path.name, round(statistics['score']['total'], 4), statistics['id_switches'], corrector.swaps,
          flush=True)
    return statistics, corrector.swaps


def replay_pairs(paths, args):
    inputs = [(path, args.teacher, args.model, args.cutoff, args.ambiguity, args.seed_model) for path in paths]
    if args.jobs == 1:
        output = list(map(replay_worker, inputs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            output = list(pool.map(replay_worker, inputs))
    packets, changes = zip(*output)
    summary = aggregate_packets(packets)
    metadata = dict(packet_ids=args.packets, cutoff=args.cutoff, ambiguity=args.ambiguity,
                    total_pair_swaps=sum(changes), pair_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest())
    if args.seed_model is not None:
        metadata['seed_sha256'] = hashlib.sha256(args.seed_model.read_bytes()).hexdigest()
    compact = [{k: v for k, v in item.items() if k not in ('tracks', '_circle_times')} for item in packets]
    args.report.write_text(json.dumps(dict(metadata=metadata, summary=summary, packets=compact), indent=2)
                           + '\n', encoding='utf-8')
    print('score', summary['score']['total'], 'switches', summary['id_switches'], 'splits',
          summary['split_truth_tracks'], 'IDF1', summary['idf1'], 'swaps', sum(changes), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'probe', 'tune'))
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--seed-model', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--cutoff', type=float, default=.85)
    parser.add_argument('--ambiguity', type=float, default=.30)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    paths = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS if args.phase == 'tune' else TRAIN_PACKETS)
    if not 0.5 <= args.cutoff <= 1 or args.ambiguity < 0 or args.jobs < 1:
        parser.error('invalid cutoff, ambiguity or job count')
    if args.phase != 'train' and args.report is None:
        parser.error('a report is required for replay or the oracle probe')
    if args.phase == 'tune':
        return replay_pairs(paths, args)
    from lightgbm import Booster, Dataset, train

    teacher = Booster(model_file=str(args.teacher))
    if args.phase == 'train':
        pieces = []
        for path in paths:
            examples = process(path, teacher)
            pieces.append(examples)
            print(path.name, len(examples), flush=True)
        joined = np.concatenate(pieces)
        data = Dataset(joined.reshape(-1, len(PAIR_NAMES)), label=np.tile((1, 0), len(joined)),
                       feature_name=list(PAIR_NAMES))
        model = train(dict(objective='binary', learning_rate=.08, num_leaves=31, min_data_in_leaf=48,
                           force_col_wise=True, num_threads=1, deterministic=True, verbosity=-1),
                      data, num_boost_round=160)
        model.save_model(str(args.model))
        return
    model = Booster(model_file=str(args.model))
    rows = {}
    for path in paths:
        rows[path.name] = process(path, teacher, model).tolist()
        print(path.name, rows[path.name], flush=True)
    result = dict(packet_ids=args.packets, fields=['reciprocal_pairs', 'correct_independent', 'correct_pairwise',
                                                  'rescued_errors'], packets=rows,
                  total=np.sum(list(rows.values()), axis=0).tolist(),
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest())
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['total'], flush=True)


if __name__ == '__main__':
    main()
