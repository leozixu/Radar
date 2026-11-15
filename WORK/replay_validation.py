"""逐圈回放训练 CSV，使用真实参赛客户端的编解码和输出检查。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
from operator import itemgetter
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import ModuleType

from metrics import SCORE_VERSION, aggregate_packets, evaluate_packet
from packet_scope import (
    ACCEPTANCE_PACKETS, DEFAULT_DATA_DIR, DEVELOPMENT_PACKETS, TUNING_PACKETS,
    packet_paths, validate_packet_path,
)


WORK_DIR = Path(__file__).resolve().parent
PARTICIPANT_DIR = WORK_DIR / "participant_demo"
DEFAULT_CONFIG = PARTICIPANT_DIR / "models" / "frozen.json"
BASELINE_REV = "0b77719"
BASELINE_SOURCE = "WORK/participant_demo/src/algorithm.py"
CSV_FIELDS = (
    "Pt_RadarTime", "Pt_Azi", "Pt_El", "Pt_Distance", "Pt_Amp", "Pt_SNR",
    "Pt_Circle", "Pt_PtID", "Batch_ID",
)

sys.path.insert(0, str(PARTICIPANT_DIR / "src"))
from client import POINT, POINT_KEYS, TRACK, encode_tracks


def file_sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@lru_cache(maxsize=1)
def baseline_source():
    try:
        return subprocess.check_output(
            ["git", "show", f"{BASELINE_REV}:{BASELINE_SOURCE}"], cwd=WORK_DIR.parent,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"无法读取基线 {BASELINE_REV}：{exc.stderr.decode(errors='replace')}") from exc


@lru_cache(maxsize=None)
def load_handler(engine, config):
    if engine == "baseline":
        name = "radar_baseline_0b77719"
        module = ModuleType(name)
        sys.modules[name] = module
        exec(compile(baseline_source(), BASELINE_SOURCE, "exec"), module.__dict__)
        return module.process_circle
    from algorithm import load_runtime
    return load_runtime(config).process_circle


def runtime_source_fingerprints():
    files = [PARTICIPANT_DIR / name for name in (
        "requirements.txt", "Dockerfile", "run.sh", "run_client.sh"
    )]
    files.extend(path for path in (PARTICIPANT_DIR / "src").rglob("*.py"))
    return {path.relative_to(WORK_DIR).as_posix(): file_sha256(path) for path in sorted(files)}


def runtime_fingerprints():
    models = sorted((PARTICIPANT_DIR / "models").rglob("*"))
    return runtime_source_fingerprints() | {
        path.relative_to(WORK_DIR).as_posix(): file_sha256(path) for path in models if path.is_file()
    }


def engine_metadata(engine, config):
    if engine == "baseline":
        return {"engine": engine, "revision": BASELINE_REV,
                "algorithm_sha256": hashlib.sha256(baseline_source()).hexdigest()}
    config_path = Path(config) if config else DEFAULT_CONFIG
    from algorithm import load_runtime
    runtime = load_runtime(config_path)
    return {
        "engine": engine,
        "config": config_path.name,
        "config_sha256": file_sha256(config_path),
        "algorithm_sha256": file_sha256(PARTICIPANT_DIR / "src" / "algorithm.py"),
        "runtime_source_sha256": runtime_source_fingerprints(),
        "runtime": runtime.metadata,
    }


def exact_integer(value, field, minimum, maximum=None):
    integer_text = value[:-2] if value.endswith(".0") else value
    try:
        number = int(integer_text)
    except ValueError:
        number = Decimal(value)
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f"{field} 必须为整数，收到 {value!r}")
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{field} 超出有效范围，收到 {value!r}")
    return int(number)


def iter_circle_payloads(path, labels_by_point, circle_by_point, allowed=DEVELOPMENT_PACKETS):
    """逐圈生成 POINTS 负载，真值单独保存在计分字典里。"""
    path = validate_packet_path(path, allowed)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, [])
        missing = set(CSV_FIELDS).difference(header)
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{', '.join(sorted(missing))}")
        fields = itemgetter(*(header.index(field) for field in CSV_FIELDS))
        circle_id = 0
        payload = bytearray()
        for row in reader:
            try:
                radar_time, azimuth, elevation, distance, amp, snr, circle, point, label = fields(row)
                circle = exact_integer(circle, "Pt_Circle", 0, 0xFFFFFFFF)
                point = exact_integer(point, "Pt_PtID", 0, 0xFFFFFFFF)
                radar_time = exact_integer(radar_time, "Pt_RadarTime", 0, 0xFFFFFFFF)
                label = exact_integer(label, "Batch_ID", -1)
                coordinates = tuple(map(float, (azimuth, elevation, distance, amp, snr)))
                if not all(map(math.isfinite, coordinates)):
                    raise ValueError("测量字段必须为有限数值")
                record = POINT.pack(radar_time, *coordinates, circle, point)
                if circle < circle_id:
                    raise ValueError(f"圈号倒序：{circle} 在 {circle_id} 之后")
                if point in labels_by_point:
                    raise ValueError(f"点号 {point} 在包内重复")
            except (IndexError, InvalidOperation, OverflowError, struct.error, ValueError) as exc:
                raise ValueError(f"{path.name} 第 {reader.line_num} 行：{exc}") from exc

            while circle_id < circle:
                yield circle_id, payload
                circle_id += 1
                payload = bytearray()
            payload.extend(record)
            labels_by_point[point] = label
            circle_by_point[point] = circle

        if not payload:
            raise ValueError(f"{path.name} 没有点迹，无法确定扫描圈数")
        yield circle_id, payload


def replay_packet(path, engine="candidate", config=None, allowed=DEVELOPMENT_PACKETS):
    path = validate_packet_path(path, allowed)
    process_circle = load_handler(engine, config)
    labels_by_point = {}
    circle_by_point = {}
    time_by_point = {}
    outputs_by_point = {}
    state = {}
    circles_replayed = 0
    empty_circles = 0
    circle_times = []
    max_circle_ms = 0.0
    slowest_circle = None
    for circle_id, payload in iter_circle_payloads(path, labels_by_point, circle_by_point, allowed):
        started = time.perf_counter()
        points = [dict(zip(POINT_KEYS, record)) for record in POINT.iter_unpack(payload)]
        source_points = {
            (p["Time"], p["CircleID"], p["PntID"]):
            (p["Azimuth"], p["Elevation"], p["Distance"]) for p in points
        }
        try:
            tracks = process_circle(points, state)
            frame = encode_tracks(tracks, source_points)
        except Exception as exc:
            raise RuntimeError(f"{path.name} 第 {circle_id} 圈处理失败：{exc}") from exc
        duration_ms = (time.perf_counter() - started) * 1000
        circle_times.append(duration_ms)
        if slowest_circle is None or duration_ms > max_circle_ms:
            max_circle_ms = duration_ms
            slowest_circle = circle_id
        circles_replayed += 1
        empty_circles += not payload
        time_by_point.update((point, stamp) for stamp, _, point in source_points)
        for _, _, _, _, batch_id, _, point_id in TRACK.iter_unpack(memoryview(frame)[5:-4]):
            outputs_by_point[point_id] = batch_id

    return {
        "file": path.name,
        "input_points": len(labels_by_point),
        "circles_replayed": circles_replayed,
        "empty_circles": empty_circles,
        "processing_ms": sum(circle_times),
        "p99_circle_ms": _p99(circle_times),
        "max_circle_ms": max_circle_ms,
        "slowest_circle": slowest_circle,
        "_circle_times": circle_times,
        **evaluate_packet(labels_by_point, outputs_by_point, circle_by_point, time_by_point),
    }


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def print_summary(label, stats):
    score = stats["score"]
    points = "—" if score is None else f"{score['total']:.4f}"
    print(
        f"{label} | {stats['circles_replayed']:,} 圈 | 本地分 {points} | "
        f"平均覆盖率 {percent(stats['mean_track_coverage'])} | "
        f"虚警率 {percent(stats['false_alarm_ratio'])} | IDF1 {percent(stats['idf1'])} | "
        f"P99 / 最慢单圈 {stats['p99_circle_ms']:.2f} / {stats['max_circle_ms']:.2f} ms",
        flush=True,
    )


def _p99(values):
    return sorted(values)[max(0, math.ceil(0.99 * len(values)) - 1)]


def _run_packet(args):
    return replay_packet(*args)


def replay_packets(paths, jobs, engine, config, allowed):
    tasks = [(path, engine, config, allowed) for path in paths]
    if jobs == 1:
        yield from map(_run_packet, tasks)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        yield from pool.map(_run_packet, tasks)


def run_evaluation(data_dir, packets, *, jobs, engine, config=None, report=None,
                   allowed=DEVELOPMENT_PACKETS, include_tracks=True):
    packets = tuple(packets)
    paths = packet_paths(data_dir, packets, allowed)
    if report is not None:
        report = report.resolve()
        if report.suffix.lower() != ".json":
            raise ValueError("评分报告路径必须指向 .json 文件")
        if report.exists() and any(path.exists() and os.path.samefile(report, path) for path in paths):
            raise ValueError("报告路径不能指向输入 CSV")
    metadata = {
        **engine_metadata(engine, config),
        "packet_ids": list(packets),
        "score_version": SCORE_VERSION,
        "score_source_sha256": file_sha256(WORK_DIR / "metrics.py"),
    }
    print("按 CSV 真值和点号计算本地代理分。", flush=True)
    packet_stats = []
    circle_times = []
    for stats in replay_packets(paths, jobs, engine, config, allowed):
        circle_times.extend(stats.pop("_circle_times"))
        packet_stats.append(stats)
        print_summary(stats["file"], stats)

    slowest = max(packet_stats, key=lambda stats: stats["max_circle_ms"])
    totals = aggregate_packets(packet_stats)
    totals.update({
        key: sum(stats[key] for stats in packet_stats)
        for key in ("input_points", "circles_replayed", "empty_circles", "processing_ms")
    })
    totals.update({
        "packets_replayed": len(packet_stats),
        "parallel_jobs": jobs,
        "mean_circle_ms": totals["processing_ms"] / totals["circles_replayed"],
        "p99_circle_ms": _p99(circle_times),
        "max_circle_ms": slowest["max_circle_ms"],
        "slowest_file": slowest["file"],
        "slowest_circle": slowest["slowest_circle"],
    })
    print_summary(f"合计 {len(packet_stats)} 包", totals)
    print(
        f"漏报 {totals['missed_truth_points']:,} 点 | "
        f"上报杂波 {totals['clutter_output_points']:,} 点 | "
        f"上报未确定 {totals['unknown_output_points']:,} 点 | "
        f"拆轨 {totals['split_truth_tracks']:,} 条 | "
        f"合轨 {totals['merged_output_tracks']:,} 条 | "
        f"换号 {totals['id_switches']:,} 次",
        flush=True,
    )
    if not include_tracks:
        for stats in packet_stats:
            del stats["tracks"]
    result = {"metadata": metadata, "summary": totals, "packets": packet_stats}
    if report:
        with report.open("w", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--packets", type=int, nargs="+", metavar="编号",
                           help="只回放所列 001–100 包，默认 071–085")
    selection.add_argument("--all", action="store_true", help="回放 001–100")
    selection.add_argument("--acceptance", action="store_true", help="回放 086–100")
    parser.add_argument("--engine", choices=("baseline", "candidate"), default="candidate",
                        help="回放冻结候选或原始 Git 基线，默认 candidate")
    parser.add_argument("--config", type=Path, help="候选实验配置，默认使用交付配置")
    parser.add_argument("--jobs", type=int, default=1, help="并行回放的包数，默认为 1")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="CSV 目录，默认为 project/DATA/data")
    parser.add_argument("--report", type=Path, help="保存包含逐包和逐航迹指标的 JSON")
    parser.add_argument("--summary-only", action="store_true", help="报告省略逐航迹记录")
    args = parser.parse_args(argv)
    packets = args.packets or (DEVELOPMENT_PACKETS if args.all else (
        ACCEPTANCE_PACKETS if args.acceptance else TUNING_PACKETS
    ))
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")
    if args.engine == "baseline" and args.config:
        parser.error("--config 仅用于候选算法")

    try:
        packet_paths(args.data_dir, packets)
        if args.engine == "candidate" and set(packets).intersection(ACCEPTANCE_PACKETS):
            if args.config is not None:
                parser.error("086–100 验收只使用交付配置，不能传入 --config")
            from evaluate_holdout import verify_frozen_candidate
            verify_frozen_candidate()
        run_evaluation(args.data_dir, packets, jobs=args.jobs, engine=args.engine, config=args.config,
                       report=args.report, include_tracks=not args.summary_only)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"回放失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
