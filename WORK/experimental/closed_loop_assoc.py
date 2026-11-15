"""Correct association probabilities using real causal tracking histories."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from unittest.mock import patch

import numpy as np
from scipy.special import expit, logit

import teacher_assoc as teacher
import pair_swap_probe as pairs
from algorithm import load_runtime
from baseline_cv import residual
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, TUNING_PACKETS, packet_paths
import replay_validation as replay


ORIGINAL_FEATURES = teacher.candidate_features
EXTRA_NAMES = (
    "teacher_probability", "row_teacher_margin", "column_teacher_margin",
    "latest_trusted_gap", "trusted_count",
) + tuple(f"{name}_{axis}" for name in (
    "drop1_error", "drop2_error", "drop1_scaled", "drop2_scaled",
    "drop1_disagreement", "drop2_disagreement",
) for axis in ("az", "el", "range"))
FEATURE_NAMES = teacher.FEATURES + EXTRA_NAMES


def extend_features(tracks, points, rows, cols, base, probabilities, circle):
    """Compare the live filter with older fits that exclude a recent suspect hit."""
    pos, times, _, mask = teacher.history_arrays(tracks)
    count = mask.sum(axis=1)
    velocity = np.asarray([track.velocity for track in tracks])
    forecast = np.asarray([track.position+track.velocity*(circle-track.circle) for track in tracks])

    def older_fit(drop):
        # Short unstable tracks are handled by the unmodified base features.
        eligible = count[rows] >= drop+3
        valid = count >= drop+3
        selected = np.flatnonzero(valid)
        error, scaled, change = (np.full((len(rows), 3), np.nan) for _ in range(3))
        if eligible.any():
            fit, _, spread = teacher.fit_predict(
                pos[valid, :-drop], times[valid, :-drop], mask[valid, :-drop],
                circle, velocity[valid],
            )
            previous = np.searchsorted(selected, rows[eligible])
            error[eligible] = np.abs(residual(points[cols[eligible]], fit[previous]))
            scaled[eligible] = error[eligible] / spread[previous]
            change[eligible] = np.abs(residual(fit[previous], forecast[rows[eligible]]))
        return error/teacher.NOISE, scaled, change/teacher.NOISE

    e1, n1, d1 = older_fit(1)
    e2, n2, d2 = older_fit(2)
    row_best = np.zeros(len(tracks))
    col_best = np.zeros(len(points))
    np.maximum.at(row_best, rows, probabilities)
    np.maximum.at(col_best, cols, probabilities)
    return np.column_stack((
        base, probabilities, probabilities-row_best[rows], probabilities-col_best[cols],
        circle-times[rows, -1], count[rows], e1, e2, n1, n2, d1, d2,
    )).astype(np.float32)


def stable_owner(labels):
    """Infer the intended owner from sufficient agreement in early observations."""
    votes = Counter(label for label in labels if label >= 0)
    if not votes:
        return None
    label, count = votes.most_common(1)[0]
    known = sum(votes.values())
    if label == 0:
        return 0 if count >= 3 and count == known else None
    if count >= 2 and count >= .75*known:
        return label
    return None


class TrainingJournal:
    def __init__(self):
        self.scans = {}
        self.owners = {}
        self.x = []
        self.y = []
        self.examples = Counter()

    def input(self, circle, points, truth):
        labels = {}
        for point in points:
            key = (point["Azimuth"], point["Elevation"], point["Distance"])
            label = truth[point["PntID"]]
            if key in labels and labels[key] != label:
                labels[key] = -1
            else:
                labels[key] = label
        self.scans[circle] = labels

    def samples(self, tracks, readings, rows, cols, features, circle):
        intended = np.full(len(tracks), -1, dtype=int)
        corrupted = np.zeros(len(tracks), dtype=bool)
        for index, track in enumerate(tracks):
            history = [self.scans.get(scan, {}).get(tuple(value), -1) for scan, _, value in track.history]
            owner = self.owners.get(track.batch_id)
            if owner is None:
                owner = stable_owner(history)
                if owner is not None:
                    self.owners[track.batch_id] = owner
            if owner is not None:
                intended[index] = owner
                corrupted[index] = owner > 0 and any(tag>=0 and tag!=owner for tag in history[-2:])
        scan = self.scans.get(circle,{})
        labels = np.fromiter((scan.get(tuple(point),-1) for point in readings),dtype=int)
        valid = (labels[cols] >= 0) & (intended[rows] >= 0)
        target = (labels[cols] > 0) & (labels[cols] == intended[rows])
        self.x.append(features[valid])
        self.y.append(target[valid])
        self.examples["edges"] += valid.sum()
        self.examples["positive"] += (valid & target).sum()
        self.examples["clutter_track_edges"] += (valid & (intended[rows]==0)).sum()
        self.examples["damaged_history_edges"] += (valid & corrupted[rows]).sum()
        self.examples["damaged_positive"] += (valid & target & corrupted[rows]).sum()


class ResidualAssociation:
    def __init__(self, corrector, model=None, blend=0., journal=None, calibration="shared"):
        self.corrector = corrector
        self.model = model
        self.blend = blend
        self.journal = journal
        self.calibration = calibration

    def associate(self, tracks, readings, amps, circle, base_model, floor, mix, ambiguity):
        if self.model is None and self.journal is None:
            return self.corrector.associate(tracks,readings,amps,circle,base_model,floor,mix,ambiguity)
        geometry = None
        base_p, adjusted_p = None, None

        def capture(*args):
            nonlocal geometry
            geometry = ORIGINAL_FEATURES(*args)
            return geometry

        outer = self

        class Probability:
            def predict(self, base, num_threads=1):
                nonlocal base_p, adjusted_p
                rows, cols, _ = geometry
                p = base_model.predict(base, num_threads=1)
                base_p = p
                extra = extend_features(tracks,readings,rows,cols,base,p,circle)
                if outer.journal is not None:
                    outer.journal.samples(tracks,readings,rows,cols,extra,circle)
                    return p
                corrected = outer.model.predict(extra,num_threads=1)
                original = logit(np.clip(p,.001,.999))
                delta = logit(np.clip(corrected,.001,.999))-original
                adjusted_p = expit(original+outer.blend*np.clip(delta,-2.7,2.7))
                return adjusted_p

        previous_features = pairs.make_features

        def calibrated_pair(tracks, current, individual, index, confidence, *rest):
            original = np.log(np.clip(base_p,1e-6,1.)) if self.calibration=="separate" else confidence
            return previous_features(tracks,current,individual,index,original,*rest)

        with patch.object(pairs,"candidate_features",capture), patch.object(pairs,"make_features",calibrated_pair):
            matches, held = self.corrector.associate(tracks,readings,amps,circle,Probability(),floor,mix,ambiguity)
        if self.calibration=="separate" and adjusted_p is not None:
            indexes = {(r,c):i for i,(r,c) in enumerate(zip(*geometry[:2]))}
            calibrated = []
            for r,c,q in matches:
                i = indexes[r,c]
                calibrated.append((r,c,float(max(.60,base_p[i]) if q > adjusted_p[i] else base_p[i])))
            matches = calibrated
        return matches, held


def open_policy(args, journal=None):
    from lightgbm import Booster

    anchor = Booster(model_file=str(args.teacher))
    base = teacher.Online(anchor,.20,0.,7,.30,4.)
    corrector = pairs.PairCorrector(Booster(model_file=str(args.pair)),.85)
    model = None if journal is not None or args.blend == 0. else Booster(model_file=str(args.model))
    return base, ResidualAssociation(corrector,model,args.blend,journal=journal,calibration=args.calibration)


def collect_worker(spec):
    path,args = spec
    journal = TrainingJournal()
    tracker, policy = open_policy(args,journal=journal)
    state,labels,circles = {},{},{}
    with patch.object(teacher,"assign_confirmed",policy.associate), patch.object(
        teacher.compat,"baseline_circle",tracker.track
    ):
        operate = load_runtime().process_circle
        for circle,payload in replay.iter_circle_payloads(path,labels,circles,allowed=TRAIN_PACKETS):
            points = [dict(zip(replay.POINT_KEYS,item)) for item in replay.POINT.iter_unpack(payload)]
            journal.input(circle,points,labels)
            operate(points,state)
    print(path.name,dict(journal.examples),flush=True)
    return np.concatenate(journal.x), np.concatenate(journal.y), {
        key:int(value) for key,value in journal.examples.items()
    }


def replay_worker(spec):
    path,args = spec
    tracker,policy = open_policy(args)
    with patch.object(teacher,"assign_confirmed",policy.associate), patch.object(
        teacher.compat,"baseline_circle",tracker.track
    ), patch.object(replay,"load_handler",return_value=load_runtime().process_circle):
        result = replay.replay_packet(path,allowed=TUNING_PACKETS)
    print(path.name,result["score"]["total"],result["idf1"],result["id_switches"],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("train","tune"))
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--pair",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--packets",type=int,nargs="+",required=True)
    parser.add_argument("--blend",type=float,default=.40)
    parser.add_argument("--calibration",choices=("shared","separate"),default="shared")
    parser.add_argument("--jobs",type=int,default=1)
    args = parser.parse_args()
    if not 0 <= args.blend <= 1 or not 1 <= args.jobs <= 2:
        parser.error("invalid blend or job count")
    training = args.phase=="train"
    paths = packet_paths(DEFAULT_DATA_DIR,args.packets,TRAIN_PACKETS if training else TUNING_PACKETS)
    worker = collect_worker if training else replay_worker
    specs = [(path,args) for path in paths]
    if args.jobs==1:
        output = list(map(worker,specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            output = list(pool.map(worker,specs))
    metadata = {
        "packet_ids":args.packets, "blend":args.blend, "calibration":args.calibration,
        "source_sha256":replay.file_sha256(Path(__file__)),
        "teacher_sha256":replay.file_sha256(args.teacher), "pair_sha256":replay.file_sha256(args.pair),
    }
    if training:
        from lightgbm import Dataset, train

        x,y,counts = zip(*output)
        training_data = Dataset(np.concatenate(x),label=np.concatenate(y),feature_name=list(FEATURE_NAMES))
        model = train(dict(objective="binary",learning_rate=.07,num_leaves=39,min_child_samples=80,
                           force_col_wise=True,deterministic=True,seed=731,verbosity=-1,
                           num_threads=args.jobs),training_data,num_boost_round=160)
        model.save_model(str(args.model))
        data = dict(metadata=metadata,packets=[{"file":path.name,**count} for path,count in zip(paths,counts)])
    else:
        data = dict(metadata=metadata,summary=aggregate_packets(output),
                    packets=[{k:v for k,v in item.items() if k not in ("tracks","_circle_times")} for item in output])
        summary = data["summary"]
        print("score",summary["score"]["total"],"IDF1",summary["idf1"],"switches",summary["id_switches"],
              "splits",summary["split_truth_tracks"],flush=True)
    if training or args.blend > 0:
        metadata["model_sha256"] = replay.file_sha256(args.model)
    args.report.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
