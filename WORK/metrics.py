"""Packet-local diagnostics using exact PntID matches to the training CSV.

Ratios are fractions. These diagnostics do not reproduce the official scorer's
track matching or acceptance thresholds.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable


_PACKET_COUNTS = (
    "truth_points",
    "output_points",
    "matched_points",
    "clutter_output_points",
    "unknown_output_points",
    "output_tracks",
    "merged_output_tracks",
)


def evaluate_packet(
    labels_by_point: dict[int, int],
    outputs_by_point: dict[int, int],
    circle_by_point: dict[int, int],
) -> dict:
    """Evaluate one packet; output values are the predicted positive BatchIDs."""
    truth_by_track = defaultdict(list)
    for point, label in labels_by_point.items():
        if label > 0:
            truth_by_track[label].append((circle_by_point[point], point))

    truth_ids_by_output_batch = defaultdict(set)
    clutter_count = unknown_count = matched_count = 0
    for point, batch in outputs_by_point.items():
        if point not in labels_by_point:
            raise ValueError(f"Output PntID {point} is absent from the input packet")
        if batch <= 0:
            raise ValueError(f"Output PntID {point} has a nonpositive BatchID")
        label = labels_by_point[point]
        if label > 0:
            truth_ids_by_output_batch[batch].add(label)
            matched_count += 1
        elif label == 0:
            clutter_count += 1
        else:
            unknown_count += 1

    tracks = [
        _evaluate_track(truth_id, sorted(points), outputs_by_point)
        for truth_id, points in sorted(truth_by_track.items())
    ]
    counts = {
        "truth_points": sum(track["truth_points"] for track in tracks),
        "output_points": len(outputs_by_point),
        "matched_points": matched_count,
        "clutter_output_points": clutter_count,
        "unknown_output_points": unknown_count,
        "output_tracks": len(set(outputs_by_point.values())),
        "merged_output_tracks": sum(
            len(labels) > 1 for labels in truth_ids_by_output_batch.values()
        ),
    }
    return _summarize(counts, tracks) | {"tracks": tracks}


def _evaluate_track(
    truth_id: int, points: list[tuple[int, int]], outputs_by_point: dict[int, int]
) -> dict:
    output_batches = Counter()
    first_match = None
    previous_circle = previous_batch = None
    missed_since_match = False
    id_switches = missed_segments = circle_gaps = 0
    current_run = longest_run = 0
    for circle, point in points:
        batch = outputs_by_point.get(point)
        if batch is None:
            missed_since_match = previous_batch is not None
            continue

        output_batches[batch] += 1
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
    }


def _summarize(counts: dict[str, int], tracks: list[dict]) -> dict:
    truth_points = counts["truth_points"]
    matched_points = counts["matched_points"]
    unmatched = counts["clutter_output_points"] + counts["unknown_output_points"]
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
        "false_alarm_ratio": unmatched / truth_points if truth_points else None,
        "split_truth_tracks": sum(len(track["output_batches"]) > 1 for track in tracks),
        "id_switches": sum(track["id_switches"] for track in tracks),
        "missed_segments": sum(track["missed_segments"] for track in tracks),
        "circle_gaps": sum(track["circle_gaps"] for track in tracks),
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
