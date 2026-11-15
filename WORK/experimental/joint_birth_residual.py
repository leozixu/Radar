"""Score remaining links after joint locks without altering their physical calibration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import expit, logit

import closed_loop_assoc as closed
import pair_swap_probe as pairs
from algorithm import load_runtime
from causal_birth_paths import BirthOnline
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR, TUNING_PACKETS, packet_paths
import replay_validation as replay
import tracker_compat


class ResidualAfterLocks:
    """Preserve trusted joint decisions and calibrate only unreserved candidate costs."""

    def __init__(self, solver, scorer, blend):
        self.solver = solver
        self.scorer = scorer
        self.blend = blend

    def associate(self, tracks, positions, amps, circle, raw_model, floor, mix, ambiguity):
        if self.blend==0 or not tracks or not len(positions):
            return self.solver.associate(tracks,positions,amps,circle,raw_model,floor,mix,ambiguity)
        # JointElevationMatcher's caller has a cached edge list with locked
        # rows and columns removed. Never regenerate the missing edges.
        compact = pairs.candidate_features(tracks,positions,amps,circle)
        rows,cols,base=compact
        if not len(rows):
            return self.solver.associate(tracks,positions,amps,circle,raw_model,floor,mix,ambiguity)
        raw = raw_model.predict(base,num_threads=1)
        augmented = closed.extend_features(tracks,positions,rows,cols,base,raw,circle)
        fitted = self.scorer.predict(augmented,num_threads=1)
        raw_logit = logit(np.clip(raw,.001,.999))
        correction = logit(np.clip(fitted,.001,.999))-raw_logit
        assignment = expit(raw_logit+self.blend*np.clip(correction,-2.7,2.7))
        physical_scale=np.log(np.clip(raw,1e-6,1.))
        make_features=pairs.make_features

        def calibrated(tracks,current,features,index,_unused,*rest):
            return make_features(tracks,current,features,index,physical_scale,*rest)

        with patch.object(pairs,"candidate_features",return_value=compact), patch.object(
            pairs,"make_features",calibrated
        ):
            matches,held=self.solver.associate(
                tracks,positions,amps,circle,SimpleNamespace(predict=lambda *a,**k:assignment),floor,mix,ambiguity)
        indices={(r,c):k for k,(r,c) in enumerate(zip(rows,cols))}
        output=[]
        for r,c,q in matches:
            k=indices[r,c]
            # .60 is the established floor applied by the reciprocal physical swap.
            output.append((r,c,float(max(.60,raw[k]) if q>assignment[k] else raw[k])))
        return output,held


def run_packet(spec):
    path,args=spec
    from lightgbm import Booster

    model=lambda name: Booster(model_file=str(getattr(args,name)))
    run=BirthOnline(model("teacher"),model("pair"),model("birth"),model("clock"),.86,"supplement",
                    paired_birth=True,joint=True,two_hit_pairs=getattr(args,"two_hit_pairs",False))
    run.matcher.base=ResidualAfterLocks(run.matcher.base,model("residual"),args.blend)
    with patch.object(tracker_compat,"baseline_circle",run.track), patch.object(
        replay,"load_handler",return_value=load_runtime().process_circle
    ):
        report=replay.replay_packet(path,allowed=TUNING_PACKETS)
    report["births"]=run.confirmations
    report["paired_births"]=run.paired_confirmations
    print(path.name,report["score"]["total"],report["id_switches"],report["split_truth_tracks"],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("teacher","pair","birth","clock","residual","report"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--packets",type=int,nargs="+",required=True)
    parser.add_argument("--blend",type=float,default=.55)
    parser.add_argument("--jobs",type=int,choices=(1,2),default=1)
    parser.add_argument("--two-hit-pairs",action="store_true")
    args=parser.parse_args()
    if not 0.<=args.blend<=1.:
        parser.error("blend must be between 0 and 1")
    specs=[(p,args) for p in packet_paths(DEFAULT_DATA_DIR,args.packets,TUNING_PACKETS)]
    if args.jobs==1:
        packets=list(map(run_packet,specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            packets=list(pool.map(run_packet,specs))
    summary=aggregate_packets(packets)
    result={
        "metadata":{"packet_ids":args.packets,"blend":args.blend,"two_hit_pairs":args.two_hit_pairs,
                    "source_sha256":replay.file_sha256(Path(__file__)),
                    "sha256":{k:replay.file_sha256(getattr(args,k)) for k in
                              ("teacher","pair","birth","clock","residual")}},
        "summary":summary,
        "packets":[{k:v for k,v in item.items() if k not in ("tracks","_circle_times")} for item in packets],
    }
    args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("score",summary["score"]["total"],"IDF1",summary["idf1"],"switches",summary["id_switches"],
          "splits",summary["split_truth_tracks"],flush=True)


if __name__=="__main__":
    main()
