"""Test one-scan alternative motion states after withheld or doubtful links."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import teacher_assoc as teacher
import pair_swap_probe as pairs
from algorithm import load_runtime
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay


def copy_motion(track):
    result = copy(track)
    result.position = track.position.copy()
    result.velocity = track.velocity.copy()
    result.covariance = track.covariance.copy()
    result.history = deque(track.history,maxlen=track.history.maxlen)
    return result


class DeferredBranches:
    """Use a bounded counterfactual state without changing any published ID."""

    def __init__(self, corrector, strength=.70, mode="both", doubtful=.65):
        self.corrector = corrector
        self.strength = strength
        self.mode = mode
        self.doubtful = doubtful
        self.pending = {}
        self.held_branches = 0
        self.coasting_branches = 0
        self.boosted = 0
        self.changed = 0

    def score_branches(self, tracks, points, amps, circle, model, rows, cols, p):
        branches, origin, priors = [], [], []
        for row, track in enumerate(tracks):
            stash = self.pending.get(track.batch_id)
            if stash is None or stash[0] != circle-1:
                continue
            for state,prior in stash[1]:
                branches.append(state)
                origin.append(row)
                priors.append(prior)
        adjusted = p.copy()
        if not branches:
            return adjusted
        r,c,features = teacher.candidate_features(branches,points,amps,circle)
        if not len(r):
            return adjusted
        predicted = model.predict(features,num_threads=1)*np.asarray(priors)[r]
        candidate_to_edge = {pair:i for i,pair in enumerate(zip(rows,cols))}
        for branch, column, probability in zip(r,c,predicted):
            edge = candidate_to_edge.get((origin[branch],column))
            if edge is None or probability < .55 or probability <= p[edge]:
                continue
            q = p[edge] + self.strength*(probability-p[edge])
            adjusted[edge] = max(adjusted[edge],q)
        self.boosted += int((adjusted>p).sum())
        return adjusted

    def associate(self, tracks, points, amps, circle, model, floor, mix, ambiguity):
        matrix = None
        p = None
        updated = None
        original_features = pairs.candidate_features
        original_pair = pairs.make_features

        def capture(*args):
            nonlocal matrix
            matrix = original_features(*args)
            return matrix

        def compare_pair(tracks, points, x, index, confidence, *rest):
            return original_pair(tracks,points,x,index,np.log(np.clip(p,1e-6,1.)),*rest)

        owner = self

        class Probabilities:
            def predict(self, features, num_threads=1):
                nonlocal p,updated
                p = model.predict(features,num_threads=1)
                updated = owner.score_branches(tracks,points,amps,circle,model,*matrix[:2],p)
                return updated

        with patch.object(pairs,"candidate_features",capture), patch.object(pairs,"make_features",compare_pair):
            chosen,held = self.corrector.associate(tracks,points,amps,circle,Probabilities(),floor,mix,ambiguity)
        next_pending = {}
        if p is None:
            self.pending = next_pending
            return chosen,held
        rows, cols = matrix[:2]
        indexes = {pair:i for i,pair in enumerate(zip(rows,cols))}
        held_set = set(held)
        assigned = {r for r,_,_ in chosen}
        for r,c,q in chosen:
            i = indexes[r,c]
            if updated[i]>p[i] and p[i]<floor:
                self.changed += 1
            if self.mode=="held" or p[i]>=self.doubtful or len(tracks[r].history)<5:
                continue
            t = tracks[r]
            next_pending[t.batch_id]=(circle,[(copy_motion(t),.85+.15*(1-p[i]))])
            self.coasting_branches += 1
        if held_set and self.mode!="coast":
            for r in np.unique(rows[np.isin(cols,list(held_set))]):
                t = tracks[r]
                if r in assigned or len(t.history)<5:
                    continue
                plausible = np.flatnonzero((rows==r)&(p>=floor)&np.isin(cols,list(held_set)))
                possible = plausible[np.argsort(-p[plausible],kind="stable")[:2]]
                branches = []
                for i in possible:
                    c=cols[i]
                    state=copy_motion(t)
                    teacher.update(state,points[c],amps[c],circle,p[i],2.)
                    branches.append((state,.80+.20*p[i]))
                if branches:
                    next_pending[t.batch_id]=(circle,branches)
                    self.held_branches += len(branches)
        self.pending = next_pending
        return chosen,held


def run_packet(spec):
    path,args = spec
    from lightgbm import Booster

    base = teacher.Online(Booster(model_file=str(args.teacher)),.20,0.,7,.30,4.)
    backend = DeferredBranches(pairs.PairCorrector(Booster(model_file=str(args.pair)),.85),
                               args.strength,args.mode,args.doubtful)
    with patch.object(teacher,"assign_confirmed",backend.associate), patch.object(
        teacher.compat,"baseline_circle",base.track
    ), patch.object(replay,"load_handler",return_value=load_runtime().process_circle):
        result = replay.replay_packet(path,allowed=TUNING_PACKETS)
    counts = {name:getattr(backend,name) for name in ("held_branches","coasting_branches","boosted","changed")}
    print(path.name,result["score"]["total"],result["id_switches"],counts,flush=True)
    return result,counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--pair",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--packets",type=int,nargs="+",required=True)
    parser.add_argument("--strength",type=float,default=.70)
    parser.add_argument("--doubtful",type=float,default=.65)
    parser.add_argument("--mode",choices=("both","coast","held"),default="both")
    parser.add_argument("--jobs",type=int,default=1)
    args=parser.parse_args()
    if not 0<=args.strength<=1 or not .20<=args.doubtful<=1 or not 1<=args.jobs<=2:
        parser.error("invalid strength, confidence or job count")
    specs=[(path,args) for path in packet_paths(DEFAULT_DATA_DIR,args.packets,TUNING_PACKETS)]
    if args.jobs==1:
        report=list(map(run_packet,specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            report=list(pool.map(run_packet,specs))
    packets,counts=zip(*report)
    summary=aggregate_packets(packets)
    print("score",summary["score"]["total"],"IDF1",summary["idf1"],"switches",summary["id_switches"],
          "splits",summary["split_truth_tracks"],flush=True)
    data={
        "metadata": {"packet_ids":args.packets,"strength":args.strength,"mode":args.mode,
                     "doubtful":args.doubtful,"source_sha256":replay.file_sha256(Path(__file__)),
                     "teacher_sha256":replay.file_sha256(args.teacher),"pair_sha256":replay.file_sha256(args.pair)},
        "summary":summary,
        "packets":[{k:v for k,v in item.items() if k not in ("tracks","_circle_times")} | events
                   for item,events in report],
    }
    args.report.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
