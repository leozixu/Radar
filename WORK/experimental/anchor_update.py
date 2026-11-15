"""Protect the KF and identity history against a single outlying mature association."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import teacher_assoc as teacher
import pair_swap_probe as pairs
from baseline_cv import residual
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay


class GuardedUpdate:
    """Downweight state assimilation when older measured motion contradicts a hit."""

    def __init__(self, update, cutoff, strength):
        self.update = update
        self.cutoff = cutoff
        self.strength = strength
        self.checked = 0
        self.weakened = 0

    def __call__(self, track, position, amplitude, circle, probability=1., robust=0.):
        if (track.batch_id > 0 and robust > 0 and len(track.history)>=6
                and 0<circle-track.history[-1][0]<=2):
            self.checked+=1
            measurements=list(track.history)
            pos=np.asarray([[p for _,_,p in measurements]])
            times=np.asarray([[t for t,_,_ in measurements]])
            expected,_,spread=teacher.fit_predict(pos,times,np.ones(times.shape,dtype=bool),
                                                  circle,track.velocity[None,:])
            # Only the horizontal bearing and range are used. Elevation jitter
            # is handled by the separate group-wise assignment.
            deviation=np.abs(residual(position,expected[0])/spread[0])[[0,2]].max()
            if deviation > self.cutoff:
                self.weakened+=1
                probability=min(probability,.45)
                robust=max(robust,self.strength*(deviation/self.cutoff))
        return self.update(track,position,amplitude,circle,probability,robust)


def replay_worker(spec):
    path,args=spec
    guard=GuardedUpdate(teacher.update,args.cutoff,args.strength)
    with patch.object(teacher,"update",guard):
        result,swaps=pairs.replay_worker((path,args.teacher,args.pair,.85,.30))
    counters={"updates_checked":guard.checked,"updates_weakened":guard.weakened,"swaps":swaps}
    print(path.name,counters,flush=True)
    return result,counters


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--pair",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--packets",type=int,nargs="+",required=True)
    parser.add_argument("--cutoff",type=float,default=1.65)
    parser.add_argument("--strength",type=float,default=14.)
    parser.add_argument("--jobs",type=int,default=1)
    args=parser.parse_args()
    if args.cutoff<=0. or args.strength<=0. or not 1<=args.jobs<=2:
        parser.error("positive limits and one or two workers required")
    files=packet_paths(DEFAULT_DATA_DIR,args.packets,TUNING_PACKETS)
    specs=[(path,args) for path in files]
    if args.jobs==1:
        reports=list(map(replay_worker,specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            reports=list(executor.map(replay_worker,specs))
    packets,counts=zip(*reports)
    summary=aggregate_packets(packets)
    result={
        "metadata":{"packet_ids":args.packets,"cutoff":args.cutoff,"strength":args.strength,
                    "source_sha256":replay.file_sha256(Path(__file__)),
                    "teacher_sha256":replay.file_sha256(args.teacher),
                    "pair_sha256":replay.file_sha256(args.pair)},
        "summary":summary,
        "packets":[{k:v for k,v in item.items() if k not in ("_circle_times","tracks")} | counter
                   for item,counter in reports],
    }
    args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("score",summary["score"]["total"],"IDF1",summary["idf1"],"switches",summary["id_switches"],
          "splits",summary["split_truth_tracks"],flush=True)


if __name__=="__main__":
    main()
