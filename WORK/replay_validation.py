"""逐圈回放训练 CSV，使用真实参赛客户端的编解码和输出检查。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from decimal import Decimal, InvalidOperation
import json
import math
from operator import itemgetter
from pathlib import Path
import struct
import sys
import time

from metrics import aggregate_packets, evaluate_packet


WORK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = WORK_DIR.parent / "DATA" / "data"
VALIDATION_PACKETS = (
    6, 9, 14, 35, 37, 45, 56, 65, 74, 82, 86, 100, 116, 125, 154, 156,
    165, 182, 188, 196,
)
TRAINING_PACKETS = range(1, 201)
CSV_FIELDS = (
    "Pt_RadarTime", "Pt_Azi", "Pt_El", "Pt_Distance", "Pt_Amp", "Pt_SNR",
    "Pt_Circle", "Pt_PtID", "Batch_ID",
)

sys.path.insert(0, str(WORK_DIR / "participant_demo" / "src"))
from algorithm import process_circle
from client import POINT, POINT_KEYS, TRACK, encode_tracks


def exact_integer(value, field, minimum, maximum=None):
    number = Decimal(value)
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{field} 必须为整数，收到 {value!r}")
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{field} 超出有效范围，收到 {value!r}")
    return int(number)


def iter_circle_payloads(path, labels_by_point, circle_by_point):
    """逐圈生成 POINTS 负载，真值单独保存在计分字典里。"""
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


def replay_packet(path):
    labels_by_point = {}
    circle_by_point = {}
    outputs_by_point = {}
    state = {}
    circles_replayed = 0
    empty_circles = 0
    elapsed_ms = 0.0
    max_circle_ms = 0.0
    slowest_circle = None
    for circle_id, payload in iter_circle_payloads(path, labels_by_point, circle_by_point):
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
        elapsed_ms += duration_ms
        if slowest_circle is None or duration_ms > max_circle_ms:
            max_circle_ms = duration_ms
            slowest_circle = circle_id
        circles_replayed += 1
        empty_circles += not payload
        for _, _, _, _, batch_id, _, point_id in TRACK.iter_unpack(memoryview(frame)[5:-4]):
            outputs_by_point[point_id] = batch_id

    return {
        "file": path.name,
        "input_points": len(labels_by_point),
        "circles_replayed": circles_replayed,
        "empty_circles": empty_circles,
        "processing_ms": elapsed_ms,
        "max_circle_ms": max_circle_ms,
        "slowest_circle": slowest_circle,
        **evaluate_packet(labels_by_point, outputs_by_point, circle_by_point),
    }


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def print_summary(label, stats):
    print(
        f"{label} | {stats['circles_replayed']:,} 圈 | "
        f"输出 {stats['output_points']:,} 点 | "
        f"有命中点的真值航迹 {stats['matched_truth_tracks']:,}/{stats['truth_tracks']:,} | "
        f"点覆盖率 {percent(stats['point_coverage'])} | "
        f"平均覆盖率 {percent(stats['mean_track_coverage'])} | "
        f"虚警率 {percent(stats['false_alarm_ratio'])} | "
        f"最慢单圈 {stats['max_circle_ms']:.2f} ms",
        flush=True,
    )


def replay_packets(paths, jobs):
    if jobs == 1:
        yield from map(replay_packet, paths)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        yield from pool.map(replay_packet, paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--packets", type=int, nargs="+", metavar="编号",
                           help="只回放所列验证包，例如 --packets 6 9")
    selection.add_argument("--all", action="store_true", help="回放 001 至 200 全部训练包")
    parser.add_argument("--jobs", type=int, default=1, help="并行回放的包数，默认为 1")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="CSV 目录，默认为 project/DATA/data")
    parser.add_argument("--report", type=Path, help="保存包含逐包和逐航迹指标的 JSON")
    args = parser.parse_args(argv)
    packets = args.packets or (TRAINING_PACKETS if args.all else VALIDATION_PACKETS)
    invalid_packets = set(args.packets or ()).difference(VALIDATION_PACKETS)
    if invalid_packets:
        parser.error(f"编号不属于固定验证集：{', '.join(map(str, sorted(invalid_packets)))}")
    if len(packets) != len(set(packets)):
        parser.error("验证包编号不能重复")
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")

    try:
        print("按 CSV 真值和点号计算本地指标。", flush=True)
        paths = [args.data_dir / f"train_{packet:03}.csv" for packet in packets]
        packet_stats = []
        for stats in replay_packets(paths, args.jobs):
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
            "parallel_jobs": args.jobs,
            "mean_circle_ms": totals["processing_ms"] / max(1, totals["circles_replayed"]),
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
            f"中途漏报 {totals['missed_segments']:,} 段 | "
            f"换号 {totals['id_switches']:,} 次",
            flush=True,
        )
        if args.report:
            with args.report.open("w", encoding="utf-8") as report:
                json.dump({"summary": totals, "packets": packet_stats}, report,
                          ensure_ascii=False, allow_nan=False, indent=2)
                report.write("\n")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"回放失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
