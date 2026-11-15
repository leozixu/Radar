"""Dataset partitions shared by training and evaluation."""

from pathlib import Path
import re


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "DATA" / "data"
TRAIN_PACKETS = tuple(range(1, 71))
TUNING_PACKETS = tuple(range(71, 86))
ACCEPTANCE_PACKETS = tuple(range(86, 101))
DEVELOPMENT_PACKETS = TRAIN_PACKETS + TUNING_PACKETS + ACCEPTANCE_PACKETS
HOLDOUT_PACKETS = tuple(range(101, 201))


def validate_packet_path(path, allowed=DEVELOPMENT_PACKETS):
    """Reject out-of-scope names and links to out-of-scope CSVs before open."""
    path = Path(path)
    match = re.fullmatch(r"train_(\d{3})\.csv", path.name)
    if match is None or int(match[1]) not in allowed:
        raise ValueError(f"包不属于当前数据范围：{path.name}")
    if path.resolve().name != path.name:
        raise ValueError(f"数据包路径解析为不同的文件名：{path}")
    return path


def packet_paths(data_dir, ids, allowed=DEVELOPMENT_PACKETS):
    """Validate the entire selection before returning any path to a worker."""
    ids = tuple(ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("需要至少一个包号，且包号不能重复")
    invalid = [packet for packet in ids if type(packet) is not int or packet not in allowed]
    if invalid:
        raise ValueError(f"包号不属于当前数据范围：{', '.join(map(str, invalid))}")
    return tuple(validate_packet_path(Path(data_dir) / f"train_{i:03}.csv", allowed) for i in ids)
