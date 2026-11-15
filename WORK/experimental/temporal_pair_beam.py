"""Continue ambiguous two-ID assignments over several causal scans."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

import teacher_assoc as teacher
from algorithm import load_runtime
from baseline_cv import residual
from causal_birth_paths import BirthOnline
from deferred_branch_assoc import copy_motion
import joint_elevation_pairs as joint
from metrics import aggregate_packets
from packet_scope import DEFAULT_DATA_DIR,TUNING_PACKETS,packet_paths
import pair_swap_probe as pairs
import replay_validation as replay
import tracker_compat


@dataclass(slots=True)
class Hypothesis:
    tracks: tuple
    log_score: float


@dataclass(slots=True)
class Extension:
    parent: Hypothesis
    columns: tuple
    probabilities: tuple
    score: float


@dataclass(slots=True)
class PairMemory:
    ids: tuple
    original: tuple
    hypotheses: list
    started: int
    last_scan: int


class ObservedProbability:
    """Record only the initial whole-scene prediction, before joint locks."""

    def __init__(self, model):
        self.model = model
        self.raw = None

    def predict(self, features, num_threads=1):
        probabilities = self.model.predict(features,num_threads=num_threads)
        self.raw = probabilities
        return probabilities


def replace_motion(track, state):
    # Keep the same Track object for the existing quality-filter history.
    track.position=state.position.copy()
    track.velocity=state.velocity.copy()
    track.covariance=state.covariance.copy()
    track.circle=state.circle
    track.mean_amplitude=state.mean_amplitude
    track.hits=state.hits
    track.history.clear()
    track.history.extend(state.history)


def phase_cost(first,second,expected,spread):
    delta=residual(first,second)
    error=np.abs(residual(delta,expected))/(spread+2**.5*teacher.NOISE)
    return .12*np.square(np.minimum(error,3.)).sum()


def amplitude_signature(first,second,circle):
    observed={t:amp for t,amp,_ in first.history}
    shared=[(t,observed[t]-amp) for t,amp,_ in second.history if t in observed]
    if len(shared)<5:
        return None
    time,difference=np.asarray(shared).T
    center=time-time.mean()
    trend=np.dot(center,difference-difference.mean())/max(1.,np.dot(center,center))
    errors=difference-difference.mean()-center*trend
    if np.max(np.abs(errors))>2e-6 or circle-time.max()>9:
        return None
    return float(difference.mean()+(circle-time.mean())*trend)


class TemporalPairBeam:
    def __init__(self,base,trigger,margin,scans,max_groups,width,partial=False,pair_amplitude=False):
        self.base=base
        self.trigger=trigger
        self.margin=margin
        self.scans=scans
        self.max_groups=max_groups
        self.width=width
        self.partial=partial
        self.pair_amplitude=pair_amplitude
        self.groups=[]
        self.events=Counter()

    def _extend(self, group, points, amps, circle, model, floor, available):
        active_points=np.flatnonzero(available)
        if not len(active_points):
            return []
        history_n,expected,spread,_=pairs.relative_history(*group.original,circle)
        amplitude_delta=amplitude_signature(*group.original,circle) if self.pair_amplitude else None
        branches=[]
        for parent in group.hypotheses:
            row,col,feature=teacher.candidate_features(
                parent.tracks,points[active_points],amps[active_points],circle)
            probability=model.predict(feature,num_threads=1) if len(row) else np.empty(0)
            options=[]
            for track_row in (0,1):
                hits=np.flatnonzero((row==track_row)&(probability>=floor))
                hits=hits[np.argsort(-probability[hits],kind="stable")[:3]]
                options.append([(int(active_points[col[i]]),float(probability[i])) for i in hits]+[(-1,floor)])
            for first,second in product(*options):
                columns=(first[0],second[0])
                if columns[0]>=0 and columns[0]==columns[1]:
                    continue
                p=(first[1],second[1])
                score=parent.log_score+math.log(p[0]/floor)+math.log(p[1]/floor)
                if columns[0]>=0 and columns[1]>=0 and history_n>=3:
                    score-=phase_cost(points[columns[0]],points[columns[1]],expected,spread)
                    if amplitude_delta is not None:
                        error=abs(amps[columns[0]]-amps[columns[1]]-amplitude_delta)
                        score-=min(1.8,error/1.5e-5)
                branches.append(Extension(parent,columns,p,score))
        return sorted(branches,key=lambda branch:-branch.score)

    def _advance(self, ranked, points, amps, circle):
        output=[]
        for step in ranked[:self.width]:
            parent=tuple(copy_motion(t) for t in step.parent.tracks)
            for track,column,p in zip(parent,step.columns,step.probabilities):
                if column>=0:
                    # The beam is conditional on the retained assignment; it is
                    # not an emitted point until observed evidence resolves IDs.
                    teacher.update(track,points[column],amps[column],circle,max(.56,p),4.)
            output.append(Hypothesis(parent,step.score-ranked[0].score))
        return output

    def _new_group(self, rows, tracks, indexes, probs, points, amps, circle, floor, options):
        source=tuple(copy_motion(tracks[r]) for r in rows)
        group=PairMemory(tuple(tracks[r].batch_id for r in rows),source,[],circle,circle)
        initial=Hypothesis(source,0.)
        ranked=[]
        common,expected,spread,_=pairs.relative_history(*source,circle)
        for a,b in product(*options):
            if (a>=0 and a==b) or (a<0 and b<0):
                continue
            p=(float(probs[indexes[rows[0],a]]) if a>=0 else floor,
               float(probs[indexes[rows[1],b]]) if b>=0 else floor)
            score=math.log(p[0]/floor)+math.log(p[1]/floor)
            if min(a,b)>=0 and common>=3:
                score-=phase_cost(points[a],points[b],expected,spread)
            ranked.append(Extension(initial,(a,b),p,score))
        ranked.sort(key=lambda branch:-branch.score)
        if len(ranked)<2 or ranked[0].score-ranked[1].score>self.margin:
            return None
        group.hypotheses=self._advance(ranked,points,amps,circle)
        return group

    def _capture(self, tracks, points, amps, circle, model, floor, mix, ambiguity):
        candidates=None
        locked=None
        get_edges=pairs.candidate_features
        select=joint.select_joint_options
        predictor=ObservedProbability(model)

        def record_edges(*args):
            nonlocal candidates
            candidates=get_edges(*args)
            return candidates

        def record_lock(*args):
            nonlocal locked
            locked=select(*args)
            return locked

        with patch.object(pairs,"candidate_features",record_edges), patch.object(
            joint,"select_joint_options",record_lock
        ):
            matches,held=self.base.associate(tracks,points,amps,circle,predictor,floor,mix,ambiguity)
        return matches,held,candidates,locked,predictor.raw

    def _held_pairs(self, tracks, rows, cols, indexes, p, held, matches, floor, occupied):
        matched={r for r,_,_ in matches}
        by_point={}
        for r,c in zip(rows,cols):
            if c not in held or r in matched or r in occupied or len(tracks[r].history)<5:
                continue
            if p[indexes[r,c]]>=floor:
                by_point.setdefault(c,[]).append(r)
        for c, contenders in by_point.items():
            for a,b in combinations(contenders,2):
                # Prefer two established tracks and never remove a sound
                # selected observation belonging to another row.
                score=float(p[indexes[a,c]]+p[indexes[b,c]])
                yield score,(a,b)

    def _selected_pairs(self, tracks, points, indexes, features, p, matches, locked, circle, occupied):
        by_column = {c:r for r,c,_ in matches}
        by_row = {r:c for r,c,_ in matches}
        logp = np.log(np.clip(p,1e-6,1.))
        for a,c2 in indexes:
            c1=by_row.get(a)
            b=by_column.get(c2,-1)
            if c1 is None or b<=a or (b,c1) not in indexes:
                continue
            if a in locked or b in locked or a in occupied or b in occupied:
                continue
            if min(len(tracks[a].history),len(tracks[b].history))<5:
                continue
            edges=[indexes[a,c1],indexes[b,c2],indexes[a,c2],indexes[b,c1]]
            if min(p[edges])<.20:
                continue
            if abs(logp[edges[:2]].sum()-logp[edges[2:]].sum())>.55:
                continue
            ix=np.full((len(tracks),len(points)),-1,dtype=int)
            for j,e in zip(((a,c1),(b,c2),(a,c2),(b,c1)),edges):
                ix[j]=e
            direct,crossed,_=pairs.make_features(tracks,points,features,ix,logp,a,b,c1,c2,circle)
            q=self.base.base.model.predict(np.asarray([direct,crossed]),num_threads=1)
            if max(q)<.71 and min(q)>.29:
                yield float(sum(p[edges])),(a,b),(c1,c2)

    def associate(self, tracks, points, amps, circle, model, floor, mix, ambiguity):
        # Anomalously large scenes retain the bounded original assignment.
        if not tracks or not len(points) or len(points)>1200:
            self.groups=[]
            return self.base.associate(tracks,points,amps,circle,model,floor,mix,ambiguity)
        matches, held, data, locked, p=self._capture(tracks,points,amps,circle,model,floor,mix,ambiguity)
        if p is None:
            self.groups=[]
            return matches,held
        rows,cols,features=data
        indexes={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rows,cols))}
        locked_rows=set(int(rows[i]) for i in locked) if locked is not None else set()
        held_set=set(int(c) for c in held)
        row_by_id={t.batch_id:i for i,t in enumerate(tracks)}
        retained=[]
        occupied=set()
        reserved_columns=set()
        for group in self.groups:
            group_rows=tuple(row_by_id.get(i,-1) for i in group.ids)
            if -1 in group_rows or set(group_rows)&locked_rows:
                self.events["cancelled_joint_or_expired"]+=1
                continue
            if group.last_scan!=circle-1:
                self.events["cancelled_gap"]+=1
                continue
            outside=[match for match in matches if match[0] not in group_rows]
            available=np.ones(len(points),dtype=bool)
            available[[c for _,c,_ in outside]]=False
            available[list(reserved_columns)]=False
            ranked=self._extend(group,points,amps,circle,model,floor,available)
            if not ranked:
                self.events["empty_hypotheses"]+=1
                continue
            winner=ranked[0]
            challenger=next((branch for branch in ranked if branch.columns!=winner.columns),None)
            margin=winner.score-challenger.score if challenger is not None else np.inf
            complete=min(winner.columns)>=0
            accepted=(0,1) if complete and margin>=self.margin else ()
            if not accepted and self.partial:
                choices=[]
                for i,c in enumerate(winner.columns):
                    if c<0 or winner.probabilities[i]<.62:
                        continue
                    rival=next((branch for branch in ranked if branch.columns[i]!=c),None)
                    if rival is None or winner.score-rival.score>=self.margin:
                        choices.append(i)
                accepted=tuple(choices)
            if accepted:
                matches=outside
                for i in accepted:
                    r,c,q,prior=group_rows[i],winner.columns[i],winner.probabilities[i],winner.parent.tracks[i]
                    replace_motion(tracks[r],prior)
                    matches.append((r,c,max(.60,q)))
                    held_set.discard(c)
                held_set.update(winner.columns[i] for i in range(2) if i not in accepted and winner.columns[i]>=0)
                occupied.update(group_rows)
                self.events["committed" if len(accepted)==2 else "committed_partial"]+=1
                self.events["committed_latent_scans"]+=circle-group.started
            elif circle-group.started<self.scans-1:
                matches=outside
                group.hypotheses=self._advance(ranked,points,amps,circle)
                group.last_scan=circle
                for choice in ranked[:self.width]:
                    reserved_columns.update(c for c in choice.columns if c>=0)
                held_set.update(reserved_columns)
                retained.append(group)
                occupied.update(group_rows)
                self.events["deferred_again"]+=1
            else:
                self.events["unresolved_deadline"]+=1
                occupied.update(group_rows)
        self.groups=retained
        if len(retained)>=self.max_groups:
            return matches,np.fromiter(sorted(held_set),dtype=int)

        candidates=[]
        if held_set-reserved_columns:
            for score,(a,b) in self._held_pairs(
                tracks,rows,cols,indexes,p,held_set-reserved_columns,matches,floor,occupied
            ):
                candidates.append((score,a,b,("held",)))
        if self.trigger=="both":
            for score,(a,b),columns in self._selected_pairs(
                tracks,points,indexes,features,p,matches,locked_rows,circle,occupied
            ):
                candidates.append((score,a,b,columns))
        for _,a,b,what in sorted(candidates,key=lambda item:-item[0]):
            if a in occupied or b in occupied:
                continue
            if what==("held",):
                options=[]
                for r in (a,b):
                    interesting=[c for c in held_set-reserved_columns if (r,c) in indexes and p[indexes[r,c]]>=floor]
                    options.append(sorted(interesting,key=lambda c:-p[indexes[r,c]])[:2]+[-1])
            else:
                options=[list(what),list(what)]
            group=self._new_group((a,b),tracks,indexes,p,points,amps,circle,floor,options)
            if group is None:
                continue
            self.groups.append(group)
            occupied.update((a,b))
            reserved_columns.update(c for columns in options for c in columns if c>=0)
            if what!=("held",):
                matches=[match for match in matches if match[0] not in (a,b)]
                held_set.update(what)
                self.events["matched_crossings_started"]+=1
            else:
                self.events["held_started"]+=1
            if len(self.groups)>=self.max_groups:
                break
        held_set.difference_update(c for _,c,_ in matches)
        return matches,np.fromiter(sorted(held_set),dtype=int)


def replay_packet(spec):
    path,args=spec
    from lightgbm import Booster

    load=lambda name:Booster(model_file=str(getattr(args,name)))
    engine=BirthOnline(load("teacher"),load("pair"),load("birth"),load("clock"),
                       .86,"supplement",paired_birth=True,joint=True,two_hit_pairs=True)
    beam=TemporalPairBeam(engine.matcher,args.trigger,args.margin,args.scans,args.groups,args.width,
                          args.partial,args.pair_amplitude)
    engine.matcher=beam
    with patch.object(tracker_compat,"baseline_circle",engine.track), patch.object(
        replay,"load_handler",return_value=load_runtime().process_circle
    ):
        result=replay.replay_packet(path,allowed=TUNING_PACKETS)
    print(path.name,result["score"]["total"],result["id_switches"],dict(beam.events),flush=True)
    return result,beam.events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("teacher","pair","birth","clock","report"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--packets",type=int,nargs="+",required=True)
    parser.add_argument("--trigger",choices=("held","both"),default="held")
    parser.add_argument("--margin",type=float,default=1.05)
    parser.add_argument("--scans",type=int,choices=(2,3,4),default=3)
    parser.add_argument("--groups",type=int,choices=(1,2),default=2)
    parser.add_argument("--width",type=int,choices=(2,3,4),default=4)
    parser.add_argument("--partial",action="store_true")
    parser.add_argument("--pair-amplitude",action="store_true")
    parser.add_argument("--jobs",type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    if args.margin<=0. or args.report.suffix.lower()!=".json":
        parser.error("positive ambiguity margin and JSON report required")
    paths=packet_paths(DEFAULT_DATA_DIR,args.packets,TUNING_PACKETS)
    specs=[(path,args) for path in paths]
    if args.jobs==1:
        reports=list(map(replay_packet,specs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            reports=list(executor.map(replay_packet,specs))
    packets,counters=zip(*reports)
    summary=aggregate_packets(packets)
    data={
        "metadata":{"packet_ids":args.packets,
                    **{key:getattr(args,key) for key in ("trigger","margin","scans","groups","width",
                                                        "partial","pair_amplitude","jobs")},
                    "source_sha256":replay.file_sha256(Path(__file__)),
                    "sha256":{key:replay.file_sha256(getattr(args,key)) for key in ("teacher","pair","birth","clock")}},
        "summary":summary,"events":dict(sum(counters,Counter())),
        "packets":[{k:v for k,v in stats.items() if k not in ("tracks","_circle_times")} | dict(events)
                   for stats,events in reports],
    }
    args.report.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(summary["score"]["total"],summary["idf1"],summary["id_switches"],summary["split_truth_tracks"],
          data["events"],flush=True)


if __name__=="__main__":
    main()
