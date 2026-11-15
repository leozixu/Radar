"""交付后手动验证 101–200，使用已固化的源码、模型和配置。"""

import argparse
import json
from pathlib import Path
import sys

from metrics import SCORE_VERSION
from packet_scope import HOLDOUT_PACKETS
from replay_validation import (
    WORK_DIR, engine_metadata, file_sha256, run_evaluation, runtime_fingerprints,
)


FROZEN_MANIFEST = WORK_DIR / "frozen_candidate.json"


def candidate_fingerprints():
    return runtime_fingerprints() | {
        name: file_sha256(WORK_DIR / name) for name in (
            "evaluate_holdout.py", "metrics.py", "packet_scope.py", "replay_validation.py",
        )
    }


def freeze_candidate():
    manifest = {
        "engine": engine_metadata("candidate", None),
        "score_version": SCORE_VERSION,
        "sha256": candidate_fingerprints(),
    }
    FROZEN_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_frozen_candidate():
    with FROZEN_MANIFEST.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sha256"), dict):
        raise ValueError("冻结校验清单格式无效")
    expected = manifest.get("sha256")
    actual = candidate_fingerprints()
    if expected != actual or manifest.get("score_version") != SCORE_VERSION:
        changed = sorted(path for path in actual.keys() | expected.keys()
                         if actual.get(path) != expected.get(path))
        raise ValueError(f"冻结校验失败，修改的文件：{', '.join(changed) or '评分版本'}")
    if manifest.get("engine") != engine_metadata("candidate", None):
        raise ValueError("冻结校验失败，运行配置或模型与交付记录不一致")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="train_101.csv 至 train_200.csv 所在目录")
    parser.add_argument("--report", type=Path, help="手动验收 JSON 报告")
    parser.add_argument("--jobs", type=int, default=1, help="并行回放的包数，默认顺序执行")
    parser.add_argument("--freeze", action="store_true", help="生成交付校验清单，仅读取程序和模型")
    args = parser.parse_args(argv)
    if args.freeze and (args.data_dir is not None or args.report is not None):
        parser.error("--freeze 不能与数据目录或评分报告一起使用")
    if not args.freeze and (args.data_dir is None or args.report is None):
        parser.error("手动回放须显式传入 --data-dir 和 --report")
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")
    try:
        if args.freeze:
            freeze_candidate()
            print(f"已生成交付校验清单 {FROZEN_MANIFEST}")
        else:
            verify_frozen_candidate()
            run_evaluation(args.data_dir, HOLDOUT_PACKETS, allowed=HOLDOUT_PACKETS,
                           jobs=args.jobs, engine="candidate", report=args.report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"手动验证失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
