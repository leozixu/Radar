"""Historical local score and packet-local diagnostics from exact PntID matches."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import fmean

import numpy as np
from scipy.optimize import linear_sum_assignment


SCORE_VERSION = "b02-local-pntid-v1"
SCORE_WEIGHTS = {
    "coverage": 30,
    "false_alarm": 10,
    "fragmentation": 20,
    "timeliness": 20,
    "tracking_length": 20,
}


_PACKET_COUNTS = (
    "truth_points",
    "output_points",
    "matched_points",
    "clutter_output_points",
    "unknown_output_points",
    "output_tracks",
    "merged_output_tracks",
    "merged_truth_points",
    "identity_true_positive_points",
)


def evaluate_packet(
    labels_by_point: dict[int, int],
    outputs_by_point: dict[int, int],
    circle_by_point: dict[int, int],
    time_by_point: dict[int, int],
) -> dict:
    """Evaluate one packet; output values are the predicted positive BatchIDs."""
    truth_by_track = defaultdict(list)
    for point, label in labels_by_point.items():
        if label > 0:
            truth_by_track[label].append((circle_by_point[point], point))

    truth_counts_by_output_batch = defaultdict(Counter)
    start_by_output_batch = {}
    clutter_count = unknown_count = matched_count = 0
    for point, batch in outputs_by_point.items():
        if point not in labels_by_point:
            raise ValueError(f"Output PntID {point} is absent from the input packet")
        if batch <= 0:
            raise ValueError(f"Output PntID {point} has a nonpositive BatchID")
        timestamp = time_by_point[point]
        start_by_output_batch[batch] = min(start_by_output_batch.get(batch, timestamp), timestamp)
        label = labels_by_point[point]
        if label > 0:
            truth_counts_by_output_batch[batch][label] += 1
            matched_count += 1
        elif label == 0:
            clutter_count += 1
        else:
            unknown_count += 1

    tracks = [
        _evaluate_track(truth_id, sorted(points), outputs_by_point, time_by_point, start_by_output_batch)
        for truth_id, points in sorted(truth_by_track.items())
    ]
    counts = {
        "truth_points": sum(track["truth_points"] for track in tracks),
        "output_points": len(outputs_by_point),
        "matched_points": matched_count,
        "clutter_output_points": clutter_count,
        "unknown_output_points": unknown_count,
        "output_tracks": len(start_by_output_batch),
        "merged_output_tracks": sum(
            len(labels) > 1 for labels in truth_counts_by_output_batch.values()
        ),
        "merged_truth_points": sum(
            labels.total() - max(labels.values()) for labels in truth_counts_by_output_batch.values()
        ),
        "identity_true_positive_points": _identity_matches(truth_by_track, truth_counts_by_output_batch),
    }
    return _summarize(counts, tracks) | {"tracks": tracks}


def _evaluate_track(
    truth_id: int, points: list[tuple[int, int]], outputs_by_point: dict[int, int],
    time_by_point: dict[int, int], start_by_output_batch: dict[int, int],
) -> dict:
    output_batches = Counter()
    after_initial_batches = Counter()
    first_match = None
    first_true_match_time = None
    previous_circle = previous_batch = None
    missed_since_match = False
    id_switches = missed_segments = circle_gaps = 0
    current_run = longest_run = 0
    for index, (circle, point) in enumerate(points):
        batch = outputs_by_point.get(point)
        if batch is None:
            missed_since_match = previous_batch is not None
            continue

        output_batches[batch] += 1
        if index >= 2:
            after_initial_batches[batch] += 1
        timestamp = time_by_point[point]
        if first_true_match_time is None or timestamp < first_true_match_time:
            first_true_match_time = timestamp
        if previous_batch is None:
            first_match = circle
        else:
            id_switches += batch != previous_batch
            missed_segments += missed_since_match
            circle_gaps += circle > previous_circle + 1

        if batch == previous_batch and circle == previous_circle + 1 and not missed_since_match:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
        previous_circle, previous_batch = circle, batch
        missed_since_match = False

    truth_count = len(points)
    matched_count = sum(output_batches.values())
    third_truth_circle = points[2][0] if truth_count >= 3 else None
    # The historical scorer assumes at least three observations. For a shorter
    # synthetic or future track, compare against its last observation instead.
    reference_time = time_by_point[points[min(2, truth_count - 1)][1]]
    matched_start = min((start_by_output_batch[batch] for batch in output_batches), default=None)
    if truth_count < 3:
        after_initial_batches = output_batches
    longest_ratio = max(after_initial_batches.values(), default=0) / max(
        1, truth_count if truth_count < 3 else truth_count - 2
    )
    return {
        "truth_id": truth_id,
        "truth_points": truth_count,
        "matched_points": matched_count,
        "coverage": matched_count / truth_count,
        "output_batches": dict(sorted(output_batches.items())),
        # Switches are between successive reported points on this truth track.
        "id_switches": id_switches,
        # Bounded runs of omitted truth observations. Leading/trailing misses
        # affect coverage, but are not interruptions between reported points.
        "missed_segments": missed_segments,
        # Consecutive reported points separated by at least one scan, whether or
        # not the skipped scans contain any observation of this truth track.
        "circle_gaps": circle_gaps,
        "longest_same_batch_consecutive_circle_matches": longest_run,
        "first_truth_circle": points[0][0],
        "third_truth_observation_circle": third_truth_circle,
        "first_matched_circle": first_match,
        "start_delay_circles": None if first_match is None else first_match - points[0][0],
        "first_true_match_time_ms": first_true_match_time,
        "timeliness_delay_first_hit_ms": (
            None if first_true_match_time is None else max(0, first_true_match_time - reference_time)
        ),
        "fragmentation": 1 / len(output_batches) if output_batches else None,
        "timeliness": _timeliness(matched_start, reference_time),
        "timeliness_first_hit": _timeliness(first_true_match_time, reference_time),
        "tracking_length": min(1., longest_ratio),
        "tracking_length_literal": max(1., longest_ratio),
    }


def _timeliness(start, reference):
    if start is None:
        return 0.
    return 1 - max(0, min(start - reference, 50_000)) / 50_000


def _identity_matches(truth_by_track, counts_by_output):
    if not counts_by_output:
        return 0
    truth_idx = {label: i for i, label in enumerate(truth_by_track)}
    overlaps = np.zeros((len(truth_idx), len(counts_by_output)), dtype=np.int32)
    for column, counts in enumerate(counts_by_output.values()):
        for label, count in counts.items():
            overlaps[truth_idx[label], column] = count
    rows, columns = linear_sum_assignment(overlaps, maximize=True)
    return int(overlaps[rows, columns].sum())


def _score(tracks, false_alarm_ratio):
    if not tracks:
        return None
    scores = {key: fmean(track[key] for track in tracks) for key in (
        "coverage", "timeliness", "timeliness_first_hit", "tracking_length", "tracking_length_literal"
    )}
    fragments = [track["fragmentation"] for track in tracks if track["fragmentation"] is not None]
    scores["fragmentation"] = fmean(fragments) if fragments else 0.
    scores["false_alarm"] = max(0., 1 - false_alarm_ratio)
    weighted = {key: scores[key] * weight for key, weight in SCORE_WEIGHTS.items()}
    total = sum(weighted.values())
    return {
        "version": SCORE_VERSION,
        "matched_tracks": len(fragments),
        "tracks": len(tracks),
        "raw_percent_scores": {key: value * 100 for key, value in scores.items()},
        "weighted_points": weighted,
        "total": total,
        "total_literal_max": total + 20 * (scores["tracking_length_literal"] - scores["tracking_length"]),
        "total_first_hit": total + 20 * (scores["timeliness_first_hit"] - scores["timeliness"]),
    }


def _summarize(counts: dict[str, int], tracks: list[dict]) -> dict:
    truth_points = counts["truth_points"]
    matched_points = counts["matched_points"]
    unmatched = counts["clutter_output_points"] + counts["unknown_output_points"]
    false_alarm_ratio = unmatched / truth_points if truth_points else None
    identity_total = truth_points + counts["output_points"]
    return counts | {
        "truth_tracks": len(tracks),
        "matched_truth_tracks": sum(track["matched_points"] > 0 for track in tracks),
        "missed_truth_points": truth_points - matched_points,
        "unmatched_output_points": unmatched,
        "point_coverage": matched_points / truth_points if truth_points else None,
        "mean_track_coverage": (
            sum(track["coverage"] for track in tracks) / len(tracks) if tracks else None
        ),
        # The denominator follows the task PDF. Unknown-label outputs are
        # unmatched by this exact-label proxy and are also reported separately.
        "false_alarm_ratio": false_alarm_ratio,
        "idf1": 2 * counts["identity_true_positive_points"] / identity_total if identity_total else None,
        "merged_truth_point_ratio": counts["merged_truth_points"] / matched_points if matched_points else None,
        "split_truth_tracks": sum(len(track["output_batches"]) > 1 for track in tracks),
        "id_switches": sum(track["id_switches"] for track in tracks),
        "missed_segments": sum(track["missed_segments"] for track in tracks),
        "circle_gaps": sum(track["circle_gaps"] for track in tracks),
        "score": _score(tracks, false_alarm_ratio),
    }


def aggregate_packets(packet_metrics: Iterable[dict]) -> dict:
    """Combine exact-match diagnostics without merging IDs across packets."""
    counts = dict.fromkeys(_PACKET_COUNTS, 0)
    tracks = []
    for packet in packet_metrics:
        for key in _PACKET_COUNTS:
            counts[key] += packet[key]
        tracks.extend(packet["tracks"])
    return _summarize(counts, tracks)
