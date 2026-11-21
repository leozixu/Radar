"""Attribute dropped truth points in an unchanged precision submission on 001–070."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile


WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, encode_tracks, file_sha256, iter_circle_payloads


STAGES = ('raw_gate', 'association_held', 'no_proposal', 'emission_rejected')


@dataclass(frozen=True, slots=True)
class MissedPoint:
    stage: str
    raw_probability: float
    hold_kind: str | None = None
    protected_batch: int | None = None
    association_probability: float | None = None


class PointStages:
    """Observe real tracker hooks while preserving its proposals and responses."""

    def __init__(self, tracker, pairing):
        self.begin = pairing.begin
        self.protect_fallback = pairing.protect_fallback
        self.track = tracker.track

    def capture_pairs(self, tracks, raw, times, circle, state):
        frame = self.begin(tracks, raw, times, circle, state)
        self.restored_columns = {int(col) for _, idx in frame.decisions for col in frame.columns[idx]}
        return frame

    def capture_holds(self, frame, tracks, positions, circle, matches, held):
        selected, withheld = self.protect_fallback(frame, tracks, positions, circle, matches, held)
        self.held_columns = set(map(int, withheld))
        self.ambiguous_columns = set(map(int, held))
        self.protected_rejected = {
            int(col): (int(tracks[row].batch_id), float(probability)) for row, col, probability in matches
            if int(col) in self.held_columns and int(col) not in self.ambiguous_columns
        }
        return selected, withheld

    def capture_proposals(self, points, state, observe):
        self.proposals = self.track(points, state, observe)
        return self.proposals

    def classify(self, points, state, floor, labels, emitted):
        proposed = {result['PntID'] for result in self.proposals}
        missed = {}
        for idx, (point, probability) in enumerate(zip(points, state['raw_return_probability'], strict=True)):
            pid = point['PntID']
            if labels[pid] <= 0 or pid in emitted:
                continue
            if pid in proposed:
                stage = 'emission_rejected'
            elif probability < floor and idx not in self.restored_columns:
                stage = 'raw_gate'
            elif idx in self.held_columns:
                stage = 'association_held'
            else:
                stage = 'no_proposal'
            hold_kind, protected_batch, association_prob = None, None, None
            if stage == 'association_held':
                if idx in self.ambiguous_columns:
                    hold_kind = 'ordinary_ambiguity'
                elif idx in self.protected_rejected:
                    hold_kind = 'protected_pair_fallback'
                    protected_batch, association_prob = self.protected_rejected[idx]
                else:
                    hold_kind = 'other_hold'
            missed[pid] = MissedPoint(stage, float(probability), hold_kind, protected_batch,
                                      association_prob)
        return missed


def group_misses(labels, circles, outputs, missed):
    truth_by_track = defaultdict(list)
    for pid, label in labels.items():
        if label > 0:
            truth_by_track[label].append(pid)
    stage_counts = Counter(point.stage for point in missed.values())
    by_phase = defaultdict(Counter)
    coverage_loss = Counter()
    dominant = {}
    next_id = max(outputs.values(), default=0) + 1
    for label, pids in truth_by_track.items():
        hits = [pid for pid in pids if pid in outputs]
        batches = Counter(outputs[pid] for pid in hits)
        if batches:
            dominant[label] = batches.most_common(1)[0][0]
            first = min((circles[pid], pid) for pid in hits)
            last = max((circles[pid], pid) for pid in hits)
        else:
            dominant[label] = next_id
            next_id += 1
        for pid in pids:
            if pid not in missed:
                continue
            if not hits:
                phase = 'no_public_hit'
            elif (circles[pid], pid) < first:
                phase = 'before_first_hit'
            elif (circles[pid], pid) > last:
                phase = 'after_last_hit'
            else:
                phase = 'between_hits'
            stage = missed[pid].stage
            by_phase[phase][stage] += 1
            coverage_loss[stage] += 1 / len(pids)
    high_raw = Counter(point.stage for point in missed.values() if point.raw_probability >= .90)
    hold_kinds = Counter(point.hold_kind for point in missed.values() if point.hold_kind)
    protected_quality = defaultdict(Counter)
    for pid, point in missed.items():
        probability = point.association_probability
        if probability is None:
            continue
        bucket = next(bound for bound in (.1, .25, .5, .75, .9, 1.) if probability <= bound)
        profile = protected_quality[f'probability_at_most_{bucket}']
        profile['all'] += 1
        correct = point.protected_batch == dominant[labels[pid]]
        profile['same_as_truth_dominant_id'] += correct
        profile['raw_q90'] += point.raw_probability >= .90
        profile['raw_q90_and_dominant_id'] += point.raw_probability >= .90 and correct
    non_dominant = sum(label > 0 and pid in outputs and outputs[pid] != dominant[label]
                       for pid, label in labels.items())
    detail = {'raw_q90_by_stage': high_raw, 'hold_kinds': hold_kinds,
              'protected_fallback_quality': protected_quality, 'non_dominant_emitted_truth_points': non_dominant}
    return stage_counts, by_phase, coverage_loss, dominant, detail


def audit_packet(path, participant):
    sys.path.insert(0, str(participant / 'src'))
    from algorithm import load_runtime

    runtime = load_runtime(participant / 'models' / 'frozen.json')
    tracker, history = runtime.new_packet()
    pairing = getattr(tracker, 'original', tracker).pairing
    stages = PointStages(tracker, pairing)
    labels, circles, times, outputs, missed = {}, {}, {}, {}, {}
    state = {'_precision_packet': (tracker, history)}
    with (patch.object(tracker, 'track', stages.capture_proposals),
          patch.object(pairing, 'begin', stages.capture_pairs),
          patch.object(pairing, 'protect_fallback', stages.capture_holds)):
        for _, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
            points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
            source = {(p['Time'], p['CircleID'], p['PntID']): (p['Azimuth'], p['Elevation'], p['Distance'])
                      for p in points}
            times.update((pid, stamp) for stamp, _, pid in source)
            tracks = runtime.process_circle(points, state)
            encode_tracks(tracks, source)
            emitted = {p['PntID']: p['BatchID'] for p in tracks}
            outputs.update(emitted)
            missed.update(stages.classify(points, state, runtime.settings.raw_floor, labels, emitted))

    baseline = evaluate_packet(labels, outputs, circles, times)
    if len(missed) != baseline['missed_truth_points']:
        raise AssertionError(f'{path.name}: missing-point attribution fails to reconcile')
    counts, phases, coverage_loss, dominant, detail = group_misses(labels, circles, outputs, missed)
    # Retrospective score ceilings for point recovery, not tracking outcomes.
    ceilings = {}
    for stage in STAGES:
        recovered = {pid: dominant[labels[pid]] for pid, point in missed.items() if point.stage == stage}
        ceilings[stage] = evaluate_packet(labels, outputs | recovered, circles, times)
    print(f'{path.name}: {baseline["score"]["total"]:.4f}, {dict(counts)}, {dict(detail["hold_kinds"])}', flush=True)
    return path.name, baseline, ceilings, counts, phases, coverage_loss, detail


def assemble(results, packet_ids, source_fingerprint):
    baseline = aggregate_packets(result[1] for result in results)
    ceilings = {stage: aggregate_packets(result[2][stage] for result in results)['score']['total']
                for stage in STAGES}
    counts = Counter()
    phases = defaultdict(Counter)
    coverage_loss = Counter()
    by_packet = {}
    high_raw, hold_kinds, protected_quality = Counter(), Counter(), defaultdict(Counter)
    non_dominant = 0
    for name, stats, _, stages, phase_counts, loss, detail in results:
        counts.update(stages)
        coverage_loss.update(loss)
        high_raw.update(detail['raw_q90_by_stage'])
        hold_kinds.update(detail['hold_kinds'])
        non_dominant += detail['non_dominant_emitted_truth_points']
        for bound, values in detail['protected_fallback_quality'].items():
            protected_quality[bound].update(values)
        for phase, values in phase_counts.items():
            phases[phase].update(values)
        by_packet[name] = {'score': stats['score']['total'], 'stage_counts': stages, 'phase_counts': phase_counts,
                           'missed_truth_points': stats['missed_truth_points'], 'details': detail}
    return {
        'metadata': {'packet_ids': packet_ids, 'score_version': SCORE_VERSION,
                     **source_fingerprint},
        'baseline': baseline,
        'missed_truth_points_by_stage': counts,
        'missed_truth_points_by_phase_and_stage': phases,
        'coverage_deduction_by_stage': {stage: loss * 30 / baseline['truth_tracks']
                                        for stage, loss in coverage_loss.items()},
        'retrospective_stage_recovery_ceiling': ceilings,
        'details': {'raw_q90_by_stage': high_raw, 'hold_kinds': hold_kinds,
                    'protected_fallback_quality': protected_quality,
                    'non_dominant_emitted_truth_points': non_dominant},
        'packets': by_packet,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=int, nargs='+', required=True)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    release = parser.add_mutually_exclusive_group()
    release.add_argument('--submission-zip', type=Path)
    release.add_argument('--participant-dir', type=Path, help='Immutable candidate directory with src and models')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    original = args.submission_zip or WORK.parents[1] / 'radar_b02_official_1115_code.zip'
    scratch = nullcontext(args.participant_dir) if args.participant_dir else TemporaryDirectory(prefix='b02-frozen-audit-')
    with scratch as directory:
        participant = Path(directory).resolve()
        if args.participant_dir:
            source_fingerprint = {'config_sha256': file_sha256(participant / 'models' / 'frozen.json'),
                                  'runtime_source_sha256': {path.relative_to(participant).as_posix(): file_sha256(path)
                                                            for path in sorted((participant / 'src').rglob('*.py'))}}
        else:
            with ZipFile(original) as package:
                package.extractall(participant, [name for name in package.namelist()
                                                 if name.startswith(('src/', 'models/'))])
            source_fingerprint = {'submission_zip_sha256': file_sha256(original)}
        if args.jobs == 1:
            results = [audit_packet(path, participant) for path in paths]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as workers:
                results = list(workers.map(audit_packet, paths, [participant] * len(paths)))
    report = assemble(results, args.packets, source_fingerprint)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{report["baseline"]["score"]["total"]:.4f}, {dict(report["missed_truth_points_by_stage"])}', flush=True)


if __name__ == '__main__':
    main()
