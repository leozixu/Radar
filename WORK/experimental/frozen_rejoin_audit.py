"""Measure the score available from causal public-ID reconnects on training packets."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np


WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import TRAIN_PACKETS, packet_paths
from replay_validation import POINT, POINT_KEYS, encode_tracks, file_sha256, iter_circle_payloads

from identity_rejoin import SCALE, residual, straight_line


def latest_fragment_aliases(fragments, span, birth_active=None):
    """Oracle for linking whole non-overlapping public fragments of one known target."""
    aliases = {}
    links = Counter()
    joined = []
    for series in fragments.values():
        tail_by_public_id = {}
        for start, end, raw in sorted(series):
            possible = [
                (previous_end, previous_raw, public_id)
                for public_id, (previous_end, previous_raw) in tail_by_public_id.items()
                if 1 <= start - previous_end <= span
                and (birth_active is None or previous_raw not in birth_active[raw])
            ]
            public_id = raw
            if possible:
                previous_end, previous_raw, public_id = max(possible)
                gap = start - previous_end
                links[f"gap_{gap}"] += 1
                links["linked_fragments"] += 1
                aliases[raw] = public_id
                joined.append((previous_raw, raw))
            tail_by_public_id[public_id] = end, raw
    return aliases, links, joined


def candidate_blockers(joined, birth_active, by_raw, positions, circles):
    """Attribute oracle links that the archived causal candidate gate never sees."""
    counts = Counter()
    for old, new in joined:
        prior = by_raw[old][-9:]
        first = by_raw[new][0]
        gap = circles[first] - circles[prior[-1]]
        snapshot = birth_active[new]
        if old not in snapshot:
            counts["old_already_retired"] += 1
            continue
        if snapshot[old][0] == circles[first]:
            counts["old_has_internal_current_assignment"] += 1
            continue
        if snapshot[new][1] > 10:
            counts["new_raw_track_already_has_over_10_hits"] += 1
            continue
        if len(prior) < 2:
            counts["old_public_history_too_short"] += 1
            continue
        if gap < 2:
            counts["old_public_gap_one"] += 1
            continue
        corners = np.array([positions[pid] for pid in prior])
        corners[:, 0] = np.unwrap(corners[:, 0], period=1.)
        offset = np.array([circles[pid] - circles[prior[-1]] for pid in prior])
        long_pos, long_rate = straight_line(corners, offset)
        short_pos, short_rate = straight_line(corners[-4:], offset[-4:])
        projections = np.stack((long_pos + gap * long_rate, short_pos + gap * short_rate,
                                corners[-1] + gap * long_rate))
        distance = np.linalg.norm(residual(positions[first], projections) / SCALE, axis=1).min()
        if distance > 4.8 + .75 * (gap - 1):
            counts["outside_public_motion_gate"] += 1
        else:
            counts["inside_archived_candidate_gate"] += 1
    return counts


def audit(task):
    path, participant = task
    sys.path.insert(0, str(participant / "src"))
    from algorithm import load_runtime

    runtime = load_runtime(participant / "models" / "frozen.json")
    labels, circles, times, emitted, state = {}, {}, {}, {}, {}
    positions = {}
    by_raw = defaultdict(list)
    birth_active = {}
    for _, payload in iter_circle_payloads(path, labels, circles, allowed=TRAIN_PACKETS):
        points = [dict(zip(POINT_KEYS, row)) for row in POINT.iter_unpack(payload)]
        source = {(p["Time"], p["CircleID"], p["PntID"]): (p["Azimuth"], p["Elevation"], p["Distance"])
                  for p in points}
        times.update((point, stamp) for stamp, _, point in source)
        output = runtime.process_circle(points, state)
        encode_tracks(output, source)
        if output and any(row["BatchID"] not in birth_active for row in output):
            active = {track.batch_id: (track.circle, track.hits) for track in state["tracks"][0]}
            for row in output:
                birth_active.setdefault(row["BatchID"], active)
        for row in output:
            pid, batch = row["PntID"], row["BatchID"]
            by_raw[batch].append(pid)
            emitted[pid] = batch
            positions[pid] = (row["Azimuth"], row["Elevation"], row["Distance"])
    fragments = defaultdict(list)
    counts = Counter()
    for raw, pids in by_raw.items():
        owners = {labels[pid] for pid in pids if labels[pid] > 0}
        counts[f"raw_ids_with_{len(owners)}_true_owners"] += 1
        if len(owners) == 1:
            fragments[owners.pop()].append((circles[pids[0]], circles[pids[-1]], raw))

    truth_offset = max(by_raw, default=0)
    perfect = {pid: truth_offset + labels[pid] if labels[pid] > 0 else batch
               for pid, batch in emitted.items()}
    variants = {"baseline": emitted, "all_matches_correct_identity": perfect}
    diagnostics = {}
    for limit in (12, 24, 48, 120):
        for tag, required_state in (("nonoverlap", None), ("retired", birth_active)):
            aliases, links, joined = latest_fragment_aliases(fragments, limit, required_state)
            name = f"{tag}_{limit}"
            variants[name] = {pid: aliases.get(batch, batch) for pid, batch in emitted.items()}
            diagnostics[name] = links
            if name == "nonoverlap_12":
                diagnostics["archived_gate_on_oracle_links"] = candidate_blockers(
                    joined, birth_active, by_raw, positions, circles,
                )
    scores = {name: evaluate_packet(labels, output, circles, times) for name, output in variants.items()}
    print(path.name, {name: round(score['score']['total'], 4) for name, score in scores.items()}, flush=True)
    return scores, diagnostics, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--participant", type=Path, required=True)
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    tasks = [(path, args.participant.resolve()) for path in paths]
    if args.jobs == 1:
        results = list(map(audit, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(audit, tasks))
    statistics = {name: aggregate_packets(scores[name] for scores, _, _ in results) for name in results[0][0]}
    counts = sum((counts for _, _, counts in results), Counter())
    links = {name: sum((observed[name] for _, observed, _ in results), Counter()) for name in results[0][1]}
    report = {
        "metadata": {
            "packets": args.packets, "score_version": SCORE_VERSION,
            "config_sha256": file_sha256(args.participant / "models" / "frozen.json"),
            "runtime_sha256": file_sha256(args.participant / "src" / "precision_tracking" / "runtime.py"),
        },
        "summary": statistics, "counts": counts, "oracle_links": links,
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print({name: summary['score']['total'] for name, summary in statistics.items()}, flush=True)


if __name__ == "__main__":
    main()
