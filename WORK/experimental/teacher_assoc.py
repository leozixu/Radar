"""Experimental teacher-forced association. Truth is visible only to the training collector."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
import replay_validation as replay
from train_tracker import read_packet

import baseline_cv as cv
import tracker_compat as compat
from tracker_assoc import assign


SEARCH = np.array([0.024, 0.095, 0.009])
NOISE = np.array([0.0030, 0.015, 0.0011])
SCANS = 8
FEATURES = (
    "gap", "hits", "amplitude", "amplitude_last", "amplitude_mean",
    "amplitude_difference", "amplitude_std", "row_neighbors", "column_neighbors",
    "distance", "row_rank", "row_excess", "column_excess",
) + tuple(
    f"{name}_{axis}" for name in (
        "kf", "short", "long", "short_uncertainty", "long_uncertainty",
        "velocity", "recent_velocity_difference", "long_velocity_difference",
    ) for axis in ("az", "el", "range")
)


@dataclass(slots=True)
class Track(cv.Track):
    history: deque = field(default_factory=lambda: deque(maxlen=SCANS))


def spawn(position, amplitude, circle):
    track = Track(position.copy(), np.zeros(3), circle, amplitude, position.copy())
    track.history.append((circle, amplitude, position.copy()))
    return track


def update(track, position, amplitude, circle, probability=1., robust=0.):
    if robust == 0 or track.hits == 1:
        cv.update_track(track, position, amplitude, circle)
    else:
        gap = circle-track.circle
        p, pv, v = track.covariance
        q = cv.PROCESS_NOISE
        p_prior = p+2*gap*pv+gap**2*v+q*gap**3/3
        pv_prior = pv+gap*v+q*gap**2/2
        v_prior = v+gap*q
        r = 1+robust*(1-probability)**2
        kp, kv = p_prior/(p_prior+r), pv_prior/(p_prior+r)
        prediction = track.position+gap*track.velocity
        innovation = cv.residual(position,prediction)
        track.position = prediction+kp*innovation
        track.position[0] %= 1.
        track.velocity += kv*innovation
        track.covariance[0] = (1-kp)*p_prior
        track.covariance[1] = (1-kp)*pv_prior
        track.covariance[2] = v_prior-kv*pv_prior
        track.circle = circle
        track.hits += 1
        track.mean_amplitude += (amplitude-track.mean_amplitude)/min(track.hits,8)/np.sqrt(r)
    # Weak assignments influence the short-term prediction without replacing
    # the history that distinguishes two established IDs during a crossing.
    if robust == 0 or probability >= .55 or len(track.history)<3:
        track.history.append((circle, amplitude, position.copy()))


def history_arrays(tracks):
    positions = np.zeros((len(tracks), SCANS, 3))
    times = np.zeros((len(tracks), SCANS))
    mask = np.zeros((len(tracks), SCANS), dtype=bool)
    amps = np.zeros((len(tracks), SCANS))
    for i, track in enumerate(tracks):
        observations = track.history
        n = len(observations)
        times[i, -n:] = [o[0] for o in observations]
        amps[i, -n:] = [o[1] for o in observations]
        positions[i, -n:] = [o[2] for o in observations]
        mask[i, -n:] = True
    return positions, times, amps, mask


def fit_predict(positions, times, mask, circle, velocity_fallback):
    center = positions[:, -1:, :]
    y = cv.residual(positions, center)
    t = times - times[:, -1:]
    n = mask.sum(axis=1)
    mean_t = (t * mask).sum(axis=1) / n
    mean_y = (y * mask[..., None]).sum(axis=1) / n[:, None]
    x = (t - mean_t[:, None]) * mask
    xx = (x ** 2).sum(axis=1)
    covariance = (x[..., None] * y).sum(axis=1)
    v = np.divide(covariance, xx[:, None], out=velocity_fallback.copy(), where=xx[:, None] > 0)
    relative_now = circle - times[:, -1]
    predicted = center[:, 0] + mean_y + (relative_now - mean_t)[:, None] * v
    error = (y - mean_y[:, None] - x[..., None] * v[:, None]) * mask[..., None]
    deviation = np.sqrt((error ** 2).sum(axis=1) / np.maximum(n[:, None]-2, 1))
    spread = deviation + NOISE
    forecast = 1 + 1 / n + (relative_now - mean_t) ** 2 / np.maximum(xx, 1)
    uncertainty = spread * np.sqrt(forecast[:, None])
    return predicted, v, uncertainty


def candidate_features(tracks, positions, amps, circle):
    elapsed = np.array([circle - t.circle for t in tracks])
    p = np.array([t.position for t in tracks])
    v = np.array([t.velocity for t in tracks])
    expected = p + v * elapsed[:, None]
    delta = cv.residual(positions[None, :, :], expected[:, None, :])
    gate = SEARCH * (1 + .35 * (elapsed - 1))[:, None]
    dist = ((delta / gate[:, None]) ** 2).sum(axis=2)
    r, c = np.nonzero(dist <= 1)
    if len(r) == 0:
        return r, c, np.empty((0, len(FEATURES)), dtype=np.float32)

    hp, ht, ha, hm = history_arrays(tracks)
    short, short_v, short_sigma = fit_predict(hp[:, -3:], ht[:, -3:], hm[:, -3:], circle, v)
    long, long_v, long_sigma = fit_predict(hp, ht, hm, circle, v)
    short_error = abs(cv.residual(positions[c], short[r]))
    long_error = abs(cv.residual(positions[c], long[r]))
    n = hm.sum(axis=1)
    amp_mean = ha.sum(axis=1) / n
    amp_std = np.sqrt(np.maximum((ha ** 2).sum(axis=1) / n - amp_mean ** 2, 0))
    edges = dist[r, c]
    row_count = np.bincount(r, minlength=len(tracks))
    column_count = np.bincount(c, minlength=len(positions))
    row_best = np.full(len(tracks), np.inf)
    np.minimum.at(row_best, r, edges)
    column_best = np.full(len(positions), np.inf)
    np.minimum.at(column_best, c, edges)
    order = np.lexsort((edges, r))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order)) - (row_count.cumsum()-row_count)[r[order]]
    y = np.column_stack((
        elapsed[r], np.array([t.hits for t in tracks])[r], amps[c], ha[r, -1], amp_mean[r],
        abs(amps[c]-amp_mean[r]), amp_std[r], row_count[r], column_count[c],
        edges, rank, edges-row_best[r], edges-column_best[c],
        abs(delta[r, c]) / NOISE, short_error / NOISE, long_error / NOISE,
        short_error / short_sigma[r], long_error / long_sigma[r],
        abs(v[r]) / NOISE, abs(v[r]-short_v[r]) / NOISE, abs(short_v[r]-long_v[r]) / NOISE,
    )).astype(np.float32)
    return r, c, y


def realistic_wrong_measurement(track, point, positions, amps, truth, circle):
    """Find the close competitive return that the original scorer could plausibly prefer."""
    elapsed = circle-track.circle
    prediction = track.position+track.velocity*elapsed
    widths = cv.FOLLOW_GATE*(1+.35*(elapsed-1))
    offsets = cv.residual(positions,prediction)/widths
    distance = (offsets**2).sum(axis=1)
    cost = (distance+.45*np.maximum(.35-amps,0)/.35
            +.22*np.abs(track.mean_amplitude-amps))
    possible = (truth>=0)&(truth!=truth[point])&(distance<=1)&(cost<=cost[point]+.15)
    if not possible.any():
        return None
    return int(np.argmin(np.where(possible,cost,np.inf)))


def teacher_pairs(path, model=None, history_corruption=0.):
    data = read_packet(path)
    previous = {}
    samples, labels = [], []
    gate_hits, gate_total = 0, 0
    injected = 0
    previous_corruption = {}
    rng = np.random.default_rng(1739+int(path.stem[6:]))
    probe = np.zeros((2, 4), dtype=np.int64)
    edges = np.r_[0, np.flatnonzero(np.diff(data[:, 6]))+1, len(data)]
    for a, b in zip(edges, edges[1:]):
        scan = data[a:b]
        circle = int(scan[0, 6])
        p = scan[:, 1:4].astype(np.float32).astype(float)
        amp = scan[:, 4].astype(np.float32).astype(float)
        truth = scan[:, 8].astype(np.int64)
        active = [(ident, t) for ident,t in previous.items() if circle-t.circle <= 5 and t.hits >= 3]
        if active:
            ids, tracks = zip(*active)
            ids = np.array(ids)
            r,c,x = candidate_features(tracks, p, amp, circle)
            truth_positive = truth[c] == ids[r]
            target_is_present = np.isin(ids, truth)
            gate_total += int(target_is_present.sum())
            gate_hits += len(np.unique(r[truth_positive]))
            valid = truth[c] >= 0
            if model is None:
                samples.append(x[valid])
                labels.append(truth_positive[valid])
            elif len(r):
                p_link = model.predict(x, num_threads=1)
                geom = ((x[:, 13:16] * NOISE / cv.FOLLOW_GATE) ** 2).sum(axis=1) / (1+.35*(x[:,0]-1))**2
                legacy = geom + .45*np.maximum(.35-x[:,2],0)/.35 + .22*np.abs(x[:,2]-x[:,4])
                for category, eligible in enumerate((np.ones(len(r),bool),truth[c]>0)):
                    true_rows = np.unique(r[truth_positive & eligible])
                    probe[category,0] += len(true_rows)
                    for i, priority in enumerate((-legacy,p_link)):
                        order = np.lexsort((-priority[eligible],r[eligible]))
                        selected = np.flatnonzero(eligible)[order]
                        winner = selected[np.r_[True,np.diff(r[selected])!=0]] if len(selected) else selected
                        probe[category,1+i] += int(np.sum(truth_positive[winner]))
                    probe[category,3] += int(np.sum(truth_positive & eligible & (p_link >= .23)))
        for j in np.flatnonzero(truth > 0):
            ident = truth[j]
            t = previous.get(ident)
            if t is None or t.circle == circle or circle-t.circle > 5:
                previous[ident] = spawn(p[j], amp[j], circle)
            else:
                observed = j
                if (history_corruption > 0 and t.hits >= 5 and
                        circle-previous_corruption.get(ident,-SCANS) >= SCANS
                        and rng.random() < history_corruption):
                    wrong = realistic_wrong_measurement(t,j,p,amp,truth,circle)
                    if wrong is not None:
                        # The owner ID stays unchanged. The model later sees
                        # both the damaged state and older reliable history.
                        observed = wrong
                        previous_corruption[ident] = circle
                        injected += 1
                update(t, p[observed], amp[observed], circle)
    if model is not None:
        return probe, gate_hits, gate_total
    return np.concatenate(samples), np.concatenate(labels), gate_hits, gate_total, injected


def assign_confirmed(tracks, p, amp, circle, model, floor, mix, ambiguity):
    if not tracks or not len(p):
        return [], np.empty(0,dtype=int)
    r,c,x = candidate_features(tracks, p, amp, circle)
    if not len(r):
        return [], np.empty(0,dtype=int)
    q = model.predict(x, num_threads=1)
    penalty = -np.log(np.maximum(q, 1e-6)) + mix * x[:, 10]
    misses = -np.log(floor)
    if ambiguity > 0:
        chosen, competing = assign(r, c, misses-penalty, np.ones(len(tracks),dtype=bool),
                                    len(tracks), len(p), ambiguity)
        held = {int(c[i]) for cluster in competing.values() for i in cluster}
        return [(int(r[i]),int(c[i]),float(q[i])) for i in chosen], np.fromiter(held,dtype=int)
    cost = np.full((len(tracks), len(p)+len(tracks)), np.inf)
    cost[r,c] = penalty
    cost[np.arange(len(tracks)),len(p)+np.arange(len(tracks))] = misses
    rows, cols = linear_sum_assignment(cost)
    probabilities = np.zeros((len(tracks),len(p)))
    probabilities[r,c] = q
    return [(i,j,probabilities[i,j]) for i,j in zip(rows, cols) if j<len(p)], np.empty(0,dtype=int)


class Online:
    def __init__(self, model, floor, mix=0., coast=5, ambiguity=0., robust=0.):
        self.model = model
        self.floor = floor
        self.mix = mix
        self.coast = coast
        self.ambiguity = ambiguity
        self.robust = robust

    def track(self, points, state, observe=None):
        circle = points[0]["CircleID"] if points else state.get("circle", -1)+1
        state["circle"] = circle
        active, tentative, seeds = state.get("tracks", ([], [], []))
        active = [t for t in active if circle-t.circle <= self.coast]
        tentative = [t for t in tentative if circle-t.circle == 1]
        seeds = [t for t in seeds if circle-t.circle == 1]
        p = np.array([(row["Azimuth"],row["Elevation"],row["Distance"]) for row in points], dtype=float).reshape(-1,3)
        amp = np.array([row["Amp"] for row in points], dtype=float)
        rest = np.arange(len(points))
        next_id = state.get("next_batch_id", 1)
        next_tentative, output = [], []
        edge_confidence = {}
        for tier, tracks in enumerate((active, tentative, seeds)):
            keep = np.ones(len(rest), dtype=bool)
            if tier == 0:
                pairs, held = assign_confirmed(tracks, p[rest], amp[rest], circle, self.model,
                                                 self.floor, self.mix, self.ambiguity)
                keep[held] = False
            else:
                pairs = ((row,col,1.) for row,col in cv.associate(tracks, p[rest], amp[rest], circle, initial=tier==2))
            for row, col, probability in pairs:
                j = rest[col]
                t = tracks[row]
                if observe is not None:
                    observe(t, p[j], amp[j], circle)
                update(t, p[j], amp[j], circle, probability, self.robust if tier == 0 else 0.)
                keep[col] = False
                if tier == 2:
                    next_tentative.append(t)
                    continue
                if tier == 1:
                    delta = cv.residual(p[j], t.first_position)
                    if t.mean_amplitude < cv.CONFIRM_MIN_MEAN_AMPLITUDE or np.linalg.norm(delta/np.array([.002,.025,.0012])) < 1:
                        continue
                    t.batch_id = next_id
                    next_id += 1
                    active.append(t)
                output.append((j,t.batch_id))
                if tier == 0:
                    edge_confidence[points[j]["PntID"]] = probability
            rest = rest[keep]
        state["tracks"] = active, next_tentative, [
            spawn(p[j], amp[j], circle) for j in rest if amp[j] >= cv.SEED_MIN_AMPLITUDE
        ]
        state["next_batch_id"] = next_id
        state["association_confidence"] = edge_confidence
        return [{**{k:points[j][k] for k in cv.OUTPUT_FIELDS}, "BatchID": ident} for j,ident in sorted(output)]


def tune_packet(spec):
    path, args = spec
    from lightgbm import Booster
    from algorithm import load_runtime

    model = Booster(model_file=str(args.weights))
    run = Online(model,args.floor,args.mix,args.coast,args.ambiguity,args.robust)
    handler = load_runtime().process_circle
    with patch.object(compat, "baseline_circle", run.track), patch.object(replay, "load_handler", return_value=handler):
        report = replay.replay_packet(path, allowed=TUNING_PACKETS)
    print(path.name, report["score"]["total"],
          {k:report[k] for k in ("idf1","false_alarm_ratio","id_switches","split_truth_tracks","merged_output_tracks")},flush=True)
    return report


def main():
    from lightgbm import Booster, Dataset, train
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train","probe","tune"))
    parser.add_argument("--packets", type=int, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=.23)
    parser.add_argument("--mix", type=float, default=.2)
    parser.add_argument("--coast", type=int, default=5)
    parser.add_argument("--ambiguity", type=float, default=0.)
    parser.add_argument("--robust", type=float, default=0.)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=140)
    parser.add_argument("--leaves", type=int, default=39)
    parser.add_argument("--history-corruption", type=float, default=0.)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0 <= args.history_corruption < 1 or args.threads < 1 or args.jobs < 1:
        parser.error("invalid training concurrency or corruption probability")
    t0 = time.monotonic()
    if args.phase == "train":
        files = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
        xs, ys = [], []
        counts = []
        for p in files:
            x,y,h,t,injected=teacher_pairs(p,history_corruption=args.history_corruption)
            xs.append(x)
            ys.append(y)
            counts.append({"file":p.name,"edges":len(y),"positives":int(y.sum()),
                           "gate_hits":h,"gate_total":t,"corruptions":injected})
            print(f"{p.name}: {len(y)} edges, {y.sum()} positives, oracle gate {h}/{t} ({h/t:.2%}), corruptions {injected}",flush=True)
        x, y = np.concatenate(xs), np.concatenate(ys)
        model = train({"objective":"binary","learning_rate":.07,"num_leaves":args.leaves,
                       "min_child_samples":40,"num_threads":args.threads,"verbosity":-1,
                       "seed":4311,"force_col_wise":True}, Dataset(x,label=y,feature_name=list(FEATURES)), num_boost_round=args.iterations)
        model.save_model(str(args.weights))
        if args.report is not None:
            payload = {
                "metadata": {"packet_ids":args.packets,"model_sha256":hashlib.sha256(args.weights.read_bytes()).hexdigest()},
                "settings": {"iterations":args.iterations,"leaves":args.leaves,"threads":args.threads,
                             "history_corruption":args.history_corruption,"seed":4311},
                "features": FEATURES, "packets": counts,
            }
            args.report.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    elif args.phase == "probe":
        files = packet_paths(DEFAULT_DATA_DIR, args.packets, TRAIN_PACKETS)
        model = Booster(model_file=str(args.weights))
        total = np.zeros((2,4), dtype=np.int64)
        for p in files:
            d,h,t=teacher_pairs(p, model)
            total += d
            print(p.name,h,t,d.tolist(),flush=True)
        print("categories: all points and positive-only; fields: in-gate, CV-top1, model-top1, model-floor-pass")
        print(total.tolist(),flush=True)
    else:
        files = packet_paths(DEFAULT_DATA_DIR, args.packets, TUNING_PACKETS)
        if args.jobs == 1:
            stats = [tune_packet((p,args)) for p in files]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                stats = list(pool.map(tune_packet, ((p,args) for p in files)))
        s = aggregate_packets(stats)
        times = sorted(ms for item in stats for ms in item.pop("_circle_times"))
        s.update(
            circles_replayed=len(times), p99_circle_ms=times[math.ceil(.99*len(times))-1],
            max_circle_ms=times[-1], parallel_jobs=args.jobs,
        )
        print(json.dumps({k:s[k] for k in ("score","idf1","mean_track_coverage","false_alarm_ratio","id_switches","split_truth_tracks","merged_output_tracks")},ensure_ascii=False),flush=True)
        if args.report is not None:
            subset = [{k:v for k,v in item.items() if k != "tracks"} for item in stats]
            config = {name:getattr(args,name) for name in ("floor","mix","coast","ambiguity","robust")}
            payload = {
                "metadata": {"packet_ids":args.packets,"settings":config,
                             "model_sha256":hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "summary": s, "packets": subset,
            }
            args.report.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"wall_seconds={time.monotonic()-t0:.1f}",flush=True)


if __name__ == "__main__":
    main()
