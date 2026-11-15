"""Measure whether an older causal forecast restores missed candidate gates."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import closed_loop_assoc as closed
from algorithm import load_runtime
from baseline_cv import residual
from packet_scope import DEFAULT_DATA_DIR, TRAIN_PACKETS, packet_paths
import replay_validation as replay
import teacher_assoc as teacher


class GateAudit(closed.TrainingJournal):
    def __init__(self, model=None):
        super().__init__()
        self.model=model

    def samples(self, tracks, readings, rows, cols, features, circle):
        true_column = defaultdict(list)
        scanned = self.scans.get(circle,{})
        for idx,point in enumerate(readings):
            label=scanned.get(tuple(point),-1)
            if label>0:
                true_column[label].append(idx)
        present_edges = {(int(r),int(c)):i for i,(r,c) in enumerate(zip(rows,cols))}
        owners = np.full(len(tracks),-1,dtype=int)
        damaged_rows = np.zeros(len(tracks),dtype=bool)
        for r,track in enumerate(tracks):
            history = [self.scans.get(t,{}).get(tuple(pos),-1) for t,_,pos in track.history]
            owner = self.owners.get(track.batch_id)
            if owner is None:
                owner=closed.stable_owner(history)
                if owner is not None:
                    self.owners[track.batch_id]=owner
            if owner is not None:
                owners[r]=owner
            damaged=owner is not None and owner>0 and any(label>=0 and label!=owner for label in history[-2:])
            damaged_rows[r]=damaged
            candidates=true_column.get(owner)
            if not candidates:
                continue
            prefix="damaged" if damaged else "clean"
            self.examples[prefix+"_owner_present"]+=1
            correct_edges=[present_edges.get((r,c)) for c in candidates]
            correct_edges=[i for i in correct_edges if i is not None]
            if correct_edges:
                self.examples[prefix+"_in_gate"]+=1
                if max(features[i,len(teacher.FEATURES)] for i in correct_edges)>=.20:
                    self.examples[prefix+"_above_floor"]+=1
                continue
            self.examples[prefix+"_missed_gate"]+=1
            # Test only the alternative linear gates, not the unknown online ranking.
            good=[]
            for drop in (1,2):
                past=list(track.history)[:-drop]
                if len(past)<3:
                    continue
                pos=np.asarray([[p for _,_,p in past]])
                time=np.asarray([[t for t,_,_ in past]])
                expected,_,_=teacher.fit_predict(pos,time,np.ones(time.shape,dtype=bool),
                                                 circle,track.velocity[None,:])
                width=teacher.SEARCH*(1+.35*(max(1.,circle-time[0,-1])-1.))
                errors=(residual(readings[candidates],expected[0])/width)**2
                if np.any(errors.sum(axis=1)<=1):
                    good.append(drop)
            if good:
                self.examples[prefix+"_rescued_by_either_older_gate"]+=1
        if self.model is not None:
            self.score_recoveries(readings,rows,cols,features,scanned,owners,damaged_rows)

    def score_recoveries(self, readings, rows, cols, features, scanned, owners, damaged):
        labels=np.array([scanned.get(tuple(point),-1) for point in readings])
        low=features[:,len(teacher.FEATURES)]<.20
        eligible=(owners[rows]>=0)&(labels[cols]>=0)&low
        if not eligible.any():
            return
        data=features[eligible]
        q=self.model.predict(data,num_threads=1)
        matching=(owners[rows[eligible]]==labels[cols[eligible]])&(labels[cols[eligible]]>0)
        damaged_edges=damaged[rows[eligible]]
        d1_start=closed.FEATURE_NAMES.index("drop1_scaled_az")
        d2_start=closed.FEATURE_NAMES.index("drop2_scaled_az")
        supported=np.fmin((data[:,d1_start:d1_start+3]**2).sum(axis=1),
                          (data[:,d2_start:d2_start+3]**2).sum(axis=1))<=4.
        self.examples["low_original_true_edges"]+=int(matching.sum())
        self.examples["low_original_damaged_true_edges"]+=int((matching&damaged_edges).sum())
        for threshold in (.60,.75,.85,.90):
            for condition,guard in (("any",np.ones(len(q),dtype=bool)),("older_fit",supported)):
                selected=(q>=threshold)&guard
                prefix=f"rescue_{threshold:.2f}_{condition}"
                self.examples[prefix+"_edges"]+=int(selected.sum())
                self.examples[prefix+"_true"]+=int((selected&matching).sum())
                self.examples[prefix+"_damaged_true"]+=int((selected&matching&damaged_edges).sum())


def audit(path,args):
    from lightgbm import Booster

    journal=GateAudit(Booster(model_file=str(args.model)) if args.model else None)
    track,policy=closed.open_policy(args,journal=journal)
    state,truth,scans={},{},{}
    with patch.object(teacher,"assign_confirmed",policy.associate), patch.object(
        teacher.compat,"baseline_circle",track.track
    ):
        operate=load_runtime().process_circle
        for circle,payload in replay.iter_circle_payloads(path,truth,scans,allowed=TRAIN_PACKETS):
            points=[dict(zip(replay.POINT_KEYS,row)) for row in replay.POINT.iter_unpack(payload)]
            journal.input(circle,points,truth)
            operate(points,state)
    return dict(journal.examples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--pair",type=Path,required=True)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--packets",type=int,nargs="+",required=True)
    args=parser.parse_args()
    args.blend=0.
    args.calibration="shared"
    rows={}
    total=Counter()
    for path in packet_paths(DEFAULT_DATA_DIR,args.packets,TRAIN_PACKETS):
        row=audit(path,args)
        print(path.name,row,flush=True)
        rows[path.name]=row
        total.update(row)
    args.report.write_text(json.dumps(dict(packet_ids=args.packets,packets=rows,total=total),indent=2)+"\n",
                           encoding="utf-8")
    print(dict(total),flush=True)


if __name__=="__main__":
    main()
