# B02 雷达逐圈跟踪

参赛处理函数是 [`process_circle(points, state)`](participant_demo/src/algorithm.py)。客户端逐圈接收点迹，回传本圈选中的原始点和包内 `BatchID`。空结果也立即回包，新数据包使用全新的状态。容器入口和协议见 [运行说明](participant_demo/README.md)。

后续航迹身份优化的模型、调参分数与复跑命令见 [身份连续性实验](experimental/README.md)。本页记录当前冻结提交包的验收与手动验证方法。

## 固定数据范围

| 包号 | 用途 |
| --- | --- |
| 001–070 | 训练、特征标定和诊断 |
| 071–085 | 调参及消融，含 `079` 性能压力包 |
| 086–100 | 冻结后验收 |
| 101–200 | 交付后由用户手动验证 |

在 `WORK` 下安装依赖和运行回放。默认回放 `071–085`，`--all` 仅指 `001–100`，`--packets` 也只接受此范围。计时对比使用单进程，其余回放可加 `--jobs`。

```powershell
python -m pip install -r participant_demo/requirements.txt
python -B replay_validation.py --engine baseline --summary-only --report reports/baseline_tuning_replay.json
python -B replay_validation.py --engine candidate --summary-only --report reports/precision_tuning_replay.json
python -B replay_validation.py --engine candidate --packets 79 --summary-only --report reports/precision_stress_079_replay.json
```

基线为 Git 提交 `0b77719` 的跟踪器。回放经过参赛客户端的编解码和输出检查。训练标签只用于生成监督目标，回放标签只用于计算报告。历史根目录报告和 `training_200_report.json` 属于此前实验，本轮选型和验收使用固定包号重新计分。

`participant_demo/models/frozen.json` 指定参赛默认配置。冻结版用扫描内时序特征过滤杂波，利用相邻雷达时刻和跨圈稳定的有符号幅度差关联双目标，再对剩余点迹作带漏检选项的一对一分配。高速双目标的起批和续接同时放宽运动门并校验原始幅度差与角色方向。已登记双轨最多保留 12 圈，第 8–12 圈需成对证据才能接回原 ID。回包只使用本圈原始点，航迹和回传状态在包首重置。

CV 保持按圈差预测，雷达时刻用于点迹判别和配对。跨圈多假设与 CV/CA 对照保留在实验记录中。旧的 [质量过滤配置](participant_demo/models/compat.json) 与 [关联重构配置](participant_demo/models/learned.json) 可用于重现实验。冻结模型训练包、特征和各阶段消融见[实验记录](experimental/README.md)。以下命令把旧配置的实验权重写入单独目录。

```powershell
python -B train_compat_quality.py --jobs 4 --model-dir experiments\compat --report experiments\compat\training_report.json
python -B train_tracker.py --phase train --settings training_calibration.json --jobs 4 --output experiments\models --report experiments\training_report.json
python -B replay_validation.py --config participant_demo/models/learned.json --summary-only
```

消融通过 `replay_validation.py --config <实验配置>` 显式加载。候选回放选择 `086–100` 时，工具会校验交付版本并禁止实验配置覆盖。

## 评分口径

本地总分沿用历史五项代理分，分别给平均航迹覆盖率、虚警、零碎度、起批及时性和最长同 ID 跟踪赋权 30、10、20、20、20。最长同 ID 项排除真值前两次观测，以同一输出 ID 的命中数计算。虚警分按 `0` 和 `-1` 的已回传点计数。报告同时给出首次真实命中及时性、换号、合轨和 IDF1，便于检查身份错误。平台实际匹配口径以官方评分为准。

## 冻结验收结果

部署入口在 `071–085` 精确复现实验版的逐包得分及身份计数，合计 **96.0047**。校验清单锁定运行源码、配置和权重后，才启动 `086–100` 的候选验收。[部署回放](reports/precision_runtime_tuning.json) · [冻结清单](frozen_candidate.json)

| `086–100` 验收项 | 基线 | 冻结版 |
| --- | ---: | ---: |
| 总分 | 75.8009 | **95.7062** |
| 首次真实命中口径总分 | 75.4986 | 95.7007 |
| 平均航迹覆盖率 | 92.10% | 94.46% |
| 虚警比例 | 58.91% | 3.09% |
| IDF1 | 56.10% | 95.31% |
| 拆轨 | 1,243 | 118 |
| 换号 | 7,004 | 180 |
| 合轨 | 1,119 | 60 |

合计通过 95.00 分验收线。15 包中 14 包达到 95 分，最低为 `091` 的 94.5772。零碎度 19.32/20、最长同 ID 跟踪 19.08/20、虚警 3.09%，均达到分项目标。平均覆盖率 94.46% 距 97% 目标 2.54 个百分点，起批及时性 19.28/20 距 19.40 目标 0.12 分。[同包原始基线](reports/precision_baseline_acceptance.json) · [冻结验收](reports/precision_frozen_acceptance.json)

`train_079` 单进程回放包含单圈 8,797 点的压力输入。冻结版 P99 为 51.42 ms，最慢圈 184.52 ms，满足 100 ms / 1 s 工程预算。[压力报告](reports/precision_frozen_stress_079.json)

冻结入口的合成验证覆盖交叉目标、高速双轨方位环绕、空圈、跳圈重连与跨包重置。在单点轨迹起批后，将幅度降为 0.11，并连续两圈传入异常雷达时刻，原 ID 仍保持。每圈只有一个点、幅度自始为 0.20–0.28 的合成场景连续观察 12 圈仍无法起批，低幅度的孤立目标起批仍有改进空间。

冻结镜像 `radar-b02:paired-precision-frozen` 的 41 个运行文件与冻结清单一致，21 项容器测试通过。`001–005.dat` 完成真实 TCP 联调，共 3,596 圈，接收 908,636 个点并回传 95,315 个点。原版 `serve.exe` 校验了五份结果文件和五份逐圈耗时文件。[容器联调报告](reports/paired_precision_frozen_docker_tcp.json)

## 交付后手动验证

在 `WORK` 下运行独立入口。它先校验冻结源码、配置和模型，然后读取 `101–200` 并写入新报告。

```powershell
python -B evaluate_holdout.py --data-dir ..\DATA\data --report holdout_101_200.json
```
