"""Fit causal association and short-segment models on radar packets 001–070."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, gettempdir

import numpy as np
from scipy.optimize import nnls

from packet_scope import TRAIN_PACKETS, packet_paths, validate_packet_path


WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR.parent / "DATA" / "data"
RUNTIME_DIR = WORK_DIR / "participant_demo"
SRC_DIR = RUNTIME_DIR / "src"
DEFAULT_CACHE = Path(gettempdir()) / "radar-tracker-training"
CSV_FIELDS = (
    "Pt_RadarTime", "Pt_Azi", "Pt_El", "Pt_Distance", "Pt_Amp", "Pt_SNR",
    "Pt_Circle", "Pt_PtID", "Batch_ID",
)
KINDS = ("association", "birth")
SEED = 20261115

# Preserve cross-target competition when subsampling common clutter edges.
SAMPLE_LIMITS = {
    "association": (16000, 12000, 8000),
    "birth": (14000, 6000, 6000),
}


def read_packet(path):
    path = validate_packet_path(path, TRAIN_PACKETS)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = tuple(next(csv.reader(handle), []))
        if set(CSV_FIELDS).difference(header):
            raise ValueError(f"{path.name} 的 CSV 字段不完整")
        rows = np.loadtxt(handle, delimiter=",", usecols=[header.index(name) for name in CSV_FIELDS], ndmin=2)
    if rows.size == 0 or not np.isfinite(rows).all():
        raise ValueError(f"{path.name} 缺少点迹或包含非有限数值")
    for column in (0, 6, 7, 8):
        values = rows[:, column]
        minimum = -1 if column == 8 else 0
        if np.any((values < minimum) | (values > 0xFFFFFFFF) | (values != np.floor(values))):
            raise ValueError(f"{path.name} 的 {CSV_FIELDS[column]} 包含无效整数")
    if np.any(np.diff(rows[:, 6]) < 0) or len(np.unique(rows[:, 7])) != len(rows):
        raise ValueError(f"{path.name} 有倒序的扫描圈或重复的点号")
    return rows


def wrap_azimuth(delta):
    delta[..., 0] = (delta[..., 0] + 0.5) % 1.0 - 0.5
    return delta


def motion_residuals(path):
    rows = read_packet(path)
    truth = rows[rows[:, 8] > 0]
    truth = truth[np.lexsort((truth[:, 6], truth[:, 8]))]
    segments = np.split(truth, np.flatnonzero(np.diff(truth[:, 8])) + 1)
    all_residuals = []
    for gap in (1, 2, 3):
        residuals = []
        for segment in segments:
            circles = segment[:, 6]
            middle = np.searchsorted(circles, circles + gap)
            end = np.searchsorted(circles, circles + 2 * gap)
            start = np.flatnonzero(end < len(circles))
            mid, last = middle[start], end[start]
            valid = (circles[mid] - circles[start] == gap) & (circles[last] - circles[start] == 2 * gap)
            pos = segment[:, 1:4]
            before = wrap_azimuth(pos[mid[valid]] - pos[start[valid]])
            after = wrap_azimuth(pos[last[valid]] - pos[mid[valid]])
            residuals.append(after - before)
        all_residuals.append(np.concatenate(residuals).astype(np.float32))
    return all_residuals


def calibrate_motion(paths, jobs):
    if jobs == 1:
        packets = list(map(motion_residuals, paths))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            packets = list(pool.map(motion_residuals, paths))
    dispersions, sample_counts = [], []
    for chunks in zip(*packets):
        samples = np.concatenate(chunks)
        if not len(samples):
            raise ValueError("没有足够的连续真值观测可用于运动标定")
        dispersions.append(1.4826 * np.median(np.abs(samples - np.median(samples, axis=0)), axis=0))
        sample_counts.append(len(samples))
    # For adjacent equal-length CV intervals, the second difference has variance
    # 6R + 2/3 * gap^3 * Q under white measurement and acceleration noise.
    scales = np.column_stack((np.full(3, 6.0), np.arange(1, 4) ** 3 * (2 / 3)))
    r, q = np.array([nnls(scales, d ** 2)[0] for d in np.asarray(dispersions).T]).T
    r = np.maximum(r, np.maximum((np.asarray(dispersions)[0] ** 2) / 24, 1e-12))
    q = np.maximum(q, r / 100)
    return {
        "settings": {"measurement": r.tolist(), "process": q.tolist()},
        "method": "MAD second difference at gaps 1–3 fitted to CV innovation dispersion",
        "sample_counts": sample_counts,
        "residual_dispersion": [x.tolist() for x in dispersions],
    }


class LabelLookup:
    def __init__(self, point_ids, labels):
        order = np.argsort(point_ids)
        self.ids = point_ids[order]
        self.labels = labels[order]
        self.dense = None
        if self.ids[-1] < 4 * len(self.ids):
            self.dense = np.full(self.ids[-1] + 2, -1, dtype=np.int64)
            self.dense[self.ids] = self.labels

    def __call__(self, point_ids):
        if self.dense is not None:
            return self.dense[point_ids]  # -1 refers to an extra entry reserved for padding.
        where = np.minimum(np.searchsorted(self.ids, point_ids), len(self.ids) - 1)
        return np.where(self.ids[where] == point_ids, self.labels[where], -1)


def label_samples(kind, earlier_ids, earlier_labels, current_labels):
    """Return -1 to omit, or a sampling class (true edge, competing target, clutter)."""
    present = earlier_ids >= 0
    known = present & (earlier_labels >= 0)
    multiclass = np.full(len(current_labels), -1, dtype=np.int8)
    if kind == "birth":
        eligible = (present.sum(axis=1) >= 2) & np.all(known | ~present, axis=1)
        eligible &= current_labels >= 0
        true = (current_labels > 0) & np.all((earlier_labels == current_labels[:, None]) | ~present, axis=1)
    else:
        # Recover a supported identity even just after one wrong observation.
        positives = np.where(earlier_labels > 0, earlier_labels, -1)
        same_identity = (positives[:, :, None] == positives[:, None, :]) & (positives[:, :, None] > 0)
        counts = same_identity.sum(axis=2)
        winner = counts.argmax(axis=1)
        support = counts[np.arange(len(counts)), winner]
        identity = positives[np.arange(len(counts)), winner]
        known_count = known.sum(axis=1)
        unambiguous = (support * 2 > known_count) & ((support >= 2) | (present.sum(axis=1) == 1))
        pure_clutter = (known_count > 0) & np.all((earlier_labels == 0) | ~present, axis=1)
        eligible = (current_labels >= 0) & (unambiguous | pure_clutter)
        true = unambiguous & (identity == current_labels)
    multiclass[eligible & true] = 0
    multiclass[eligible & ~true & (current_labels > 0)] = 1
    multiclass[eligible & ~true & (current_labels == 0)] = 2
    return multiclass


class SampleCollector:
    def __init__(self, labels, rng):
        self.labels = labels
        self.rng = rng
        self.blocks = {kind: [[], [], []] for kind in KINDS}
        self.retained_counts = {kind: [0, 0, 0] for kind in KINDS}
        self.original_counts = {kind: [0, 0, 0] for kind in KINDS}

    def __call__(self, kind, features, earlier_ids, current_ids):
        earlier_ids = np.asarray(earlier_ids, dtype=np.int64)
        classes = label_samples(kind, earlier_ids, self.labels(earlier_ids), self.labels(current_ids))
        for index, (limit, blocks) in enumerate(zip(SAMPLE_LIMITS[kind], self.blocks[kind])):
            rows = np.flatnonzero(classes == index)
            self.original_counts[kind][index] += len(rows)
            if len(rows):
                # Attach independent priorities. The smallest priorities form a
                # uniform sample across the complete packet, regardless of circles.
                priority = self.rng.random(len(rows), dtype=np.float32)
                blocks.append((priority, features[rows].copy()))
                self.retained_counts[kind][index] += len(rows)
                if self.retained_counts[kind][index] > 2 * limit:
                    blocks[:] = [self._truncate(blocks, limit)]
                    self.retained_counts[kind][index] = len(blocks[0][0])

    @staticmethod
    def _truncate(blocks, limit):
        if len(blocks) == 1 and len(blocks[0][0]) <= limit:
            return blocks[0]
        priorities = np.concatenate([scores for scores, _ in blocks])
        values = np.concatenate([samples for _, samples in blocks])
        if len(priorities) > limit:
            selected = np.argpartition(priorities, limit - 1)[:limit]
            return priorities[selected], values[selected]
        return priorities, values

    def arrays(self, feature_counts):
        result = {}
        for kind in KINDS:
            for index, (limit, blocks) in enumerate(zip(SAMPLE_LIMITS[kind], self.blocks[kind])):
                feature_count = feature_counts[kind]
                if blocks:
                    values = self._truncate(blocks, limit)[1]
                else:
                    values = np.empty((0, feature_count), dtype=np.float32)
                result[f"{kind}_{index}"] = values
            result[f"{kind}_original_counts"] = np.asarray(self.original_counts[kind], dtype=np.int64)
        return result


def import_tracker():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import tracker

    return tracker


def collect_packet(path, cache_path, packet, settings_dict, seed_profile):
    if cache_path.is_file():
        return
    tracker = import_tracker()
    bootstrap_models = None
    if seed_profile is not None:
        from algorithm import load_runtime

        bootstrap_models = load_runtime(seed_profile).models
    settings = tracker.TrackerSettings(**settings_dict)
    rows = read_packet(path)
    radar_time, circles, point_ids, truth = rows[:, [0, 6, 7, 8]].astype(np.int64).T
    observations = rows[:, 1:6].astype(np.float32)
    lookup = LabelLookup(point_ids, truth)
    collector = SampleCollector(lookup, np.random.default_rng(SEED + packet))
    state = {}
    boundaries = np.searchsorted(circles, np.arange(int(circles[-1]) + 2))
    for circle, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        points = [
            {
                "Time": int(t), "Azimuth": float(obs[0]), "Elevation": float(obs[1]), "Distance": float(obs[2]),
                "Amp": float(obs[3]), "SNR": float(obs[4]), "CircleID": circle, "PntID": int(pid),
            }
            for t, obs, pid in zip(radar_time[start:end], observations[start:end], point_ids[start:end])
        ]
        tracker.process_candidates(points, state, settings, bootstrap_models, observer=collector)
    samples = collector.arrays({"association": len(tracker.ASSOCIATION_FEATURES), "birth": len(tracker.BIRTH_FEATURES)})
    with NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez(temporary, **samples)
    try:
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def training_key(settings, feature_names, seed_profile):
    sources = [Path(__file__), SRC_DIR / "algorithm.py", *sorted(SRC_DIR.glob("tracker*.py"))]
    source_hashes = {path.name: digest(path) for path in sources}
    seed_artifacts = None
    if seed_profile is not None:
        from algorithm import load_runtime

        seed_artifacts = load_runtime(seed_profile).metadata
    recipe = {"settings": settings, "features": feature_names, "sources": source_hashes,
              "bootstrap_artifacts": seed_artifacts, "sample_limits": SAMPLE_LIMITS, "seed": SEED}
    return hashlib.sha256(json_bytes(recipe)).hexdigest(), recipe


def training_entries(paths, packet_ids, cache_dir, key):
    cache_dir = cache_dir / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for packet, path in zip(packet_ids, paths):
        input_digest = digest(path)
        cache_path = cache_dir / f"{path.stem}-{input_digest[:16]}.npz"
        entries.append((path, cache_path, packet, input_digest))
    return entries


def collect_training(entries, settings, seed_profile, jobs):
    missing = [(path, cache, packet) for path, cache, packet, _ in entries if not cache.is_file()]
    print(f"训练候选数据：{len(entries) - len(missing)} 包缓存，{len(missing)} 包待采集", flush=True)
    if jobs == 1:
        for path, cache, packet in missing:
            collect_packet(path, cache, packet, settings, seed_profile)
            print(f"已采集 {path.stem}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(collect_packet, path, cache, packet, settings, seed_profile): path
                   for path, cache, packet in missing}
        for future in as_completed(futures):
            future.result()
            print(f"已采集 {futures[future].stem}", flush=True)


def load_examples(entries, kind, feature_count):
    counts = np.zeros(3, dtype=np.int64)
    for _, cache, _, _ in entries:
        with np.load(cache, allow_pickle=False) as data:
            counts += [len(data[f"{kind}_{i}"]) for i in range(3)]
    features = np.empty((counts.sum(), feature_count), dtype=np.float32)
    labels = np.empty(len(features), dtype=np.int8)
    original = np.zeros(3, dtype=np.int64)
    offset = 0
    for _, cache, _, _ in entries:
        with np.load(cache, allow_pickle=False) as data:
            original += data[f"{kind}_original_counts"]
            for index in range(3):
                block = data[f"{kind}_{index}"]
                end = offset + len(block)
                features[offset:end] = block
                labels[offset:end] = index == 0
                offset = end
    if not counts[0] or not counts[1:].sum():
        raise ValueError(f"{kind} 正负样本不足，需增加包数或检查候选门控")
    return features, labels, counts.tolist(), original.tolist()


def fit_models(entries, feature_names, out_dir, rounds, jobs):
    import lightgbm as lgb

    out_dir.mkdir(parents=True, exist_ok=True)
    model_stats = {}
    params = {
        "objective": "binary", "learning_rate": 0.055, "num_leaves": 31, "min_data_in_leaf": 96,
        "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 2.0,
        "max_bin": 127, "seed": SEED, "deterministic": True, "force_col_wise": True,
        "num_threads": jobs, "verbosity": -1,
    }
    for kind in KINDS:
        features, target, sampled, original = load_examples(entries, kind, len(feature_names[kind]))
        print(f"训练 {kind} 模型，真边 / 竞争目标 / 杂波采样数：{sampled}", flush=True)
        dataset = lgb.Dataset(features, label=target, feature_name=list(feature_names[kind]))
        booster = lgb.train(params, dataset, num_boost_round=rounds)
        path = out_dir / f"{kind}.txt"
        booster.save_model(path)
        model_stats[kind] = {"sha256": digest(path), "sampled": sampled, "available": original,
                             "feature_importance": dict(zip(booster.feature_name(), booster.feature_importance().tolist()))}
    return {"parameters": params, "num_boost_round": rounds, "lightgbm_version": lgb.__version__, "models": model_stats}


def settings_from(path):
    with path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    return document.get("settings", document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "collect", "fit", "train"), default="train")
    parser.add_argument("--packets", type=int, nargs="+", default=TRAIN_PACKETS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--settings", type=Path, help="标定报告或 TrackerSettings JSON；默认用所选训练包标定")
    parser.add_argument("--seed-profile", type=Path, help="可选，先前模型的配置文件用于闭环候选生成")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=RUNTIME_DIR / "models")
    parser.add_argument("--report", type=Path, default=WORK_DIR / "training_report.json")
    parser.add_argument("--rounds", type=int, default=160)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.rounds < 1:
        parser.error("--jobs 和 --rounds 必须是正整数")
    try:
        paths = packet_paths(args.data_dir, args.packets, TRAIN_PACKETS)
    except ValueError as exc:
        parser.error(str(exc))
    paths = tuple(paths)
    report = {"training_packets": list(args.packets), "phase": args.phase}
    if args.phase == "calibrate" or args.settings is None:
        calibration = calibrate_motion(paths, args.jobs)
        report.update(calibration)
        settings = calibration["settings"]
        print(f"运动标定：{settings}", flush=True)
    else:
        settings = settings_from(args.settings)
    if args.phase == "calibrate":
        report["input_sha256"] = {path.name: digest(path) for path in paths}
        write_json(args.report, report)
        return 0

    tracker = import_tracker()
    feature_names = {"association": tracker.ASSOCIATION_FEATURES, "birth": tracker.BIRTH_FEATURES}
    key, recipe = training_key(settings, feature_names, args.seed_profile)
    entries = training_entries(paths, args.packets, args.cache_dir, key)
    if args.phase != "fit":
        collect_training(entries, settings, args.seed_profile, args.jobs)
    report.update({"settings": settings, "candidate_cache_key": key, "recipe": recipe,
                   "input_sha256": {path.name: sha for path, _, _, sha in entries}})
    if args.phase in ("fit", "train"):
        report.update(fit_models(entries, feature_names, args.output, args.rounds, args.jobs))
    write_json(args.report, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
