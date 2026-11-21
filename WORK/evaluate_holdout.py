"""使用已固化的源码、模型和配置进行回归、第一阶段验收或手动终测。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from metrics import SCORE_VERSION
from packet_scope import ACCEPTANCE_PACKETS, MANUAL_FINAL_PACKETS, STAGE_ONE_PACKETS
from replay_validation import (
    WORK_DIR, engine_metadata, file_sha256, run_evaluation, runtime_fingerprints,
)


FROZEN_MANIFEST = WORK_DIR / "frozen_candidate.json"
STAGE_ONE_MINIMUM_SCORE = 98.0
PHASE_PACKETS = {
    "regression": ACCEPTANCE_PACKETS,
    "stage1": STAGE_ONE_PACKETS,
    "manual-final": MANUAL_FINAL_PACKETS,
}


def candidate_fingerprints():
    return runtime_fingerprints() | {
        name: file_sha256(WORK_DIR / name) for name in (
            "evaluate_holdout.py", "metrics.py", "packet_scope.py", "replay_validation.py",
        )
    }


def freeze_candidate(manifest_path=FROZEN_MANIFEST):
    manifest = {
        "engine": engine_metadata("candidate", None),
        "score_version": SCORE_VERSION,
        "sha256": candidate_fingerprints(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_frozen_candidate(manifest_path=FROZEN_MANIFEST):
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
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
    return hashlib.sha256(contents).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", choices=PHASE_PACKETS,
                      help="regression 回放 086–100，stage1 验收 101–140，manual-final 手动终测 141–200")
    mode.add_argument("--freeze", action="store_true", help="生成冻结校验清单，仅读取程序和模型")
    parser.add_argument("--manifest", type=Path, default=FROZEN_MANIFEST, help="冻结校验清单路径")
    parser.add_argument("--data-dir", type=Path, help="所选阶段 CSV 文件所在目录")
    parser.add_argument("--report", type=Path, help="评分 JSON 报告")
    parser.add_argument("--jobs", type=int, default=1, help="并行回放的包数，默认顺序执行")
    args = parser.parse_args(argv)
    if args.freeze and (args.data_dir is not None or args.report is not None):
        parser.error("--freeze 不能与数据目录或评分报告一起使用")
    if not args.freeze and (args.data_dir is None or args.report is None):
        parser.error("回放须显式传入 --data-dir 和 --report")
    if args.report is not None and args.report.resolve() == args.manifest.resolve():
        parser.error("评分报告不能覆盖冻结清单")
    if args.manifest.suffix.lower() != ".json":
        parser.error("冻结清单路径必须指向 .json 文件")
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")
    try:
        if args.freeze:
            freeze_candidate(args.manifest)
            print(f"已生成冻结校验清单 {args.manifest}")
            return 0
        manifest_sha256 = verify_frozen_candidate(args.manifest)
        packets = PHASE_PACKETS[args.phase]
        metadata = {"evaluation_phase": args.phase, "frozen_manifest_sha256": manifest_sha256}
        if args.phase == "stage1":
            metadata["minimum_score"] = STAGE_ONE_MINIMUM_SCORE
        result = run_evaluation(args.data_dir, packets, allowed=packets,
                                jobs=args.jobs, engine="candidate", report=args.report,
                                report_metadata=metadata)
        if args.phase == "stage1":
            score = result["summary"]["score"]
            if score is None or not math.isfinite(score["total"]):
                raise ValueError("第一阶段回放未生成有效总分")
            total = score["total"]
            passed = total >= STAGE_ONE_MINIMUM_SCORE
            print(f"第一阶段验收{'通过' if passed else '未通过'}，总分 {total!r}，要求 {STAGE_ONE_MINIMUM_SCORE:g}")
            return 0 if passed else 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"冻结验证失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
