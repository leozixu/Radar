"""Time the protected paired-return/emission recipe with the regular replay codec."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK))
import replay_validation as replay

import lightgbm
import numpy as np
import scipy

from emission_identity import (
    FEATURES, INPUT_MODELS, RAW_RESCUE, EmissionFeatures, accepted_emissions,
    current_raw_probabilities, decision_policies, make_models,
)
from causal_birth_paths import BirthOnline
from packet_scope import DEFAULT_DATA_DIR, packet_paths
from paired_returns import PairReturns
import tracker_compat


PACKETS = (79,)
DECISION = "rescue65_clutter10"
PAIR_MODEL = HERE / "models" / "paired_returns.txt"
REFERENCE = WORK / "reports" / "paired_returns_protected_emission_tuning.json"
COMPAT_PROFILE = replay.PARTICIPANT_DIR / "models" / "compat.json"
LEGACY_THRESHOLD = .16
TRACKER_SETTINGS = {"floor": .86, "mode": "supplement", "paired_birth": True,
                    "joint": True, "two_hit_pairs": True}
PAIR_SETTINGS = {"start_floor": .985, "link_floor": .95, "wide": True, "fallback_floor": .96}


def model(path):
    return lightgbm.Booster(model_file=str(path))


def emission_handler(legacy, clock_model, emission_models, scan_sizes, policy):
    history = EmissionFeatures()
    trusted_col = FEATURES.index("trusted_proposals")
    thresholds = decision_policies(True)[policy]
    raw_floor = RAW_RESCUE[policy][1] if policy in RAW_RESCUE else None

    def process(points, state):
        scan_sizes.append(len(points))
        proposals, quality = tracker_compat.decisions_with_features(points, state)
        probabilities = legacy.quality_model.predict(quality, num_threads=1) if proposals else np.empty(0)
        features = history.build(points, proposals, quality, probabilities, state, clock_model)
        if not proposals:
            return []
        detection = emission_models["detection"].predict(features, num_threads=1)
        ownership = emission_models["ownership"].predict(features, num_threads=1)
        young = features[:, trusted_col] < 3
        raw_probability = current_raw_probabilities(points, proposals, state) if raw_floor is not None else None
        keep = accepted_emissions(probabilities >= LEGACY_THRESHOLD, detection, ownership,
                                  young, thresholds, raw_probability, raw_floor)
        return [result for result, accepted in zip(proposals, keep) if accepted]

    return process


def file_key(path):
    path = path.resolve()
    return (path.relative_to(WORK) if path.is_relative_to(WORK) else path).as_posix()


def fingerprints(paths):
    return {file_key(path): replay.file_sha256(path) for path in paths}


def load_reference(path, raw_hash, raw_floor, emission_hashes, association_floor):
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (reference["paired_returns_sha256"] != replay.file_sha256(PAIR_MODEL)
            or reference["wide_pairs"] != PAIR_SETTINGS["wide"]
            or reference["fallback_floor"] != PAIR_SETTINGS["fallback_floor"]):
        raise ValueError("成对权重或阈值与调参报告不一致")
    if reference.get("raw_model_sha256") != raw_hash or reference.get("raw_floor") != raw_floor:
        raise ValueError("原始点筛选配置与调参报告不一致")
    if reference["models"] != emission_hashes:
        raise ValueError("回传权重与调参报告不一致")
    if reference.get("association_floor", .20) != association_floor:
        raise ValueError("关联阈值与调参报告不一致")
    return reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-model", type=Path, help="原始点筛选权重，默认禁用筛选")
    parser.add_argument("--raw-floor", type=float, help="筛选置信度，启用时默认 0.025")
    parser.add_argument("--association-floor", type=float, default=.20)
    parser.add_argument("--emission-policy", choices=tuple(decision_policies(True)), default=DECISION)
    parser.add_argument("--reference-report", type=Path, help="核对同一配置的 079 分数")
    args = parser.parse_args(argv)
    if args.raw_model is None and args.raw_floor is not None:
        parser.error("--raw-floor 需要 --raw-model")
    if args.raw_model is None and args.emission_policy in RAW_RESCUE:
        parser.error("所选回传策略需要 --raw-model")
    raw_floor = None if args.raw_model is None else (.025 if args.raw_floor is None else args.raw_floor)
    if raw_floor is not None and not 0 < raw_floor < 1:
        parser.error("--raw-floor 必须介于 0 和 1 之间")
    if not 0 < args.association_floor < 1:
        parser.error("--association-floor 必须介于 0 和 1 之间")
    reference_path = args.reference_report or (REFERENCE if args.raw_model is None else None)
    protected_paths = (REFERENCE.resolve(), reference_path.resolve()) if reference_path else (REFERENCE.resolve(),)
    if args.report.resolve().suffix.lower() != ".json" or args.report.resolve() in protected_paths:
        parser.error("报告应写入新的 .json 路径")
    path, = packet_paths(DEFAULT_DATA_DIR, PACKETS, PACKETS)

    experimental_sources = [HERE / f"{name}.py" for name in (
        "emission_identity", "causal_birth_paths", "paired_returns", "clock_association",
        "teacher_assoc", "pair_swap_probe",
    )]
    raw_paths = [] if args.raw_model is None else [args.raw_model]
    if raw_paths:
        experimental_sources.append(HERE / "raw_timing_detection.py")
    sources = fingerprints([Path(__file__), *experimental_sources, WORK / "replay_validation.py",
                            WORK / "metrics.py", WORK / "packet_scope.py"])
    model_paths = {name: HERE / "models" / f"emission_{name}.txt" for name in ("detection", "ownership")}
    models = fingerprints([*INPUT_MODELS.values(), PAIR_MODEL, *model_paths.values(), *raw_paths])
    raw_hash = models[file_key(args.raw_model)] if args.raw_model else None
    emission_hashes = {kind: models[file_key(p)] for kind, p in model_paths.items()}
    reference = load_reference(reference_path, raw_hash, raw_floor, emission_hashes,
                               args.association_floor) if reference_path else None
    historical = reference["packet_scores"][reference["packets"].index(79)][args.emission_policy] if reference else None

    from algorithm import load_runtime

    legacy = load_runtime(COMPAT_PROFILE)
    inputs = make_models()
    raw_gate = None
    if args.raw_model:
        from raw_timing_detection import RawReturnGate
        raw_gate = RawReturnGate(model(args.raw_model), raw_floor)
    tracker_settings = TRACKER_SETTINGS | {"association_floor": args.association_floor}
    tracker = BirthOnline(inputs["teacher"], inputs["pair"], inputs["birth"], inputs["clock"],
                          **tracker_settings, pairing=PairReturns(model(PAIR_MODEL), **PAIR_SETTINGS), raw_gate=raw_gate)
    scan_sizes = []
    handler = emission_handler(legacy, inputs["clock"], {name: model(p) for name, p in model_paths.items()},
                               scan_sizes, args.emission_policy)
    with patch.object(tracker_compat, "baseline_circle", tracker.track), \
            patch.object(replay, "load_handler", return_value=handler):
        result = replay.replay_packet(path, allowed=PACKETS)
    if historical is not None and result["score"] != historical:
        raise ValueError(f"079 得分与调参报告不一致：{result['score']['total']:.12f} / {historical['total']:.12f}")
    times = result.pop("_circle_times")
    result.pop("tracks")
    largest = max(range(len(scan_sizes)), key=scan_sizes.__getitem__)
    result["largest_input_scan"] = {"circle": largest, "points": scan_sizes[largest], "time_ms": times[largest]}
    result["parallel_jobs"] = 1
    result["mean_circle_ms"] = result["processing_ms"] / result["circles_replayed"]
    report = {
        "metadata": {
            "packet_ids": PACKETS, "decision": args.emission_policy,
            "emission_thresholds": decision_policies(True)[args.emission_policy],
            "emission_raw_floor": RAW_RESCUE[args.emission_policy][1] if args.emission_policy in RAW_RESCUE else None,
            "pairing": PAIR_SETTINGS, "tracker": tracker_settings,
            "raw_model_sha256": raw_hash, "raw_floor": raw_floor,
            "reference_tuning_score": reference["summaries"][args.emission_policy]["score"]["total"] if reference else None,
            "reference_report_sha256": replay.file_sha256(reference_path) if reference_path else None,
            "exact_reference_score_match": True if reference else None,
            "source_sha256": sources | replay.runtime_source_fingerprints(), "model_sha256": models,
            "compatibility_config": legacy.metadata, "effective_legacy_threshold": LEGACY_THRESHOLD,
            "timed_region": "POINT unpack, input snapshot, causal tracker, TRACK validation and encode",
            "environment": {"platform": platform.platform(), "python": platform.python_version(),
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "lightgbm": lightgbm.__version__,
                            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS")},
        },
        "summary": result,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replay.print_summary(path.name, result)
    print(f"Slowest circle: {result['slowest_circle']}")


if __name__ == "__main__":
    main()
