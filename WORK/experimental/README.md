# 航迹身份连续性实验

训练和诊断限定在 `001–070`，配置比较限定在 `071–085`。下列模型和脚本位于独立实验目录，现有参赛配置仍由 `participant_demo/models/frozen.json` 指定。`086–100` 保留供候选冻结后本地验收，`101–200` 由用户手动验证。

## 按干净历史训练关联模型

`teacher_assoc.py` 以训练真值构造正确的过去观测链，并在当前圈的空间门内生成正负关联样本。在线只取已收到的观测，使用原 KF 预测、短历史和长历史回归、残差和幅度特征，计算全局一对一匹配。低置信匹配降低滤波更新权重，接近等价的全局分配可暂缓该圈输出。原短轨质量过滤仍参与回传。

| `071–085` 全集 | 冻结版 | 27 包训练的关联模型 | 加成对修正 |
| --- | ---: | ---: | ---: |
| 本地代理分 | 80.4336 | 84.7978 | 85.5240 |
| IDF1 | 70.39% | 78.29% | 80.17% |
| 换号 | 7,334 | 5,565 | 4,502 |
| 拆轨 | 1,282 | 965 | 912 |
| 合轨 | 1,200 | 961 | 889 |
| 平均航迹覆盖率 | 91.79% | 92.32% | 92.33% |
| 虚警比例 | 8.67% | 7.33% | 7.32% |

从 `WORK` 回放这一配置：

```powershell
python -B experimental/teacher_assoc.py tune --packets 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 --weights experimental/teacher_assoc_wide.txt --floor .20 --mix 0 --coast 7 --ambiguity .30 --robust 4 --jobs 2 --report reports/teacher_assoc_wide_replay.json
```

权重使用下列训练命令拟合，输出到新文件以便比较 SHA-256。

```powershell
python -B experimental/teacher_assoc.py train --packets 1 3 6 8 11 14 17 20 23 25 27 30 32 35 38 40 44 47 49 52 55 57 60 62 65 68 70 --weights experimental/teacher_assoc_wide_refit.txt --threads 1 --iterations 210 --leaves 51
```

评分对照见 [冻结版](../reports/candidate_tuning.json)、[关联模型](../reports/teacher_assoc_wide_tuning.json) 和 [成对修正](../reports/teacher_pair_swap_tuning.json)。当前实验距 95 分仍有差距，剩余扣分最高的是航迹零碎度（14.81/20）、起批及时性（16.88/20）和最长同 ID 跟踪（16.86/20）。

成对版在 15 包上全部提分。若保留它当前的输出点集合、用首次真点命中计算起批，并给零碎度和最长同 ID 跟踪都记满分，乐观上界仍只有 93.49 分。后续实验还需增加真实观测覆盖或更早确认航迹。

成对版通过正式回包编码检查。`079` 的单进程回放 P99 为 19.88 ms、最慢圈 211.98 ms。两项成对关联回归分别检查两种输入排列的一致性，以及保持一对一分配和拒绝不合理换位。[压力报告](../reports/teacher_pair_swap_stress_079.json)

## 诊断和消融

- 五包因果诊断发现，83.8% 的基线换号发生时原 ID 仍有可匹配的真点。仅删除可疑输出会降低覆盖率和总分。[换号诊断](../reports/identity_diagnostics_train_sample.json)
- 在正确历史的离线诊断中，已训练关联模型的分配阈值取 0.23 时，正确关联召回率为 97.24%，精度为 97.81%。增加竞争优势门限将精度提高到 99.37% 时，召回率降为 71.95%。提升判别需要超出单纯提高门限。[关联置信度诊断](../reports/teacher_oracle_assignment.json)
- 扫描时间和方位残差可用作杂波证据。它改变现有 CV 的代价时有小幅收益，但配合已有新关联模型用于起轨代价的收益接近零；放宽检测阈值仅小幅提高总分并增加换号。[扫描相位消融](../reports/clock_teacher_probe.json) · [输出门限消融](../reports/teacher_link_quality_probe.json)

## 成对分配修正

对已分配的成熟航迹，只有两条航迹交换候选后都仍在关联门内，且至少共用三圈历史时才比较两种分配。模型使用历史相对位置的变化和两种分配的残差。两种输入排列均支持换位才执行，且一圈内每条航迹最多参与一次换位。

在训练包 `022/024/026/028` 进行的离线诊断里，已知两个正确历史和两个真实候选时，17,347 组中独立关联判对 16,853 组，成对判别判对 17,343 组。该结果只衡量两点的方向选择。无真值输入的逐圈回放在 `071/076/082/085` 得分从 85.3168 提高到 86.1293，IDF1 从 78.71% 提高到 81.00%，换号从 1,575 降到 1,255。叠加一击种子模型后的同包总分为 86.0620。[成对诊断](../reports/teacher_pair_swap_oracle.json) · [四包回放](../reports/teacher_pair_swap_sample.json) · [一击种子组合](../reports/teacher_pair_seed_sample.json)

以下命令分别重训和复跑成对模型：

```powershell
python -B experimental/pair_swap_probe.py train --packets 2 5 10 13 19 --teacher experimental/teacher_assoc_wide.txt --model experimental/models/pair_swap_refit.txt
python -B experimental/pair_swap_probe.py tune --packets 71 76 82 85 --teacher experimental/teacher_assoc_wide.txt --model experimental/models/pair_swap_wide.txt --cutoff .85 --jobs 2 --report reports/teacher_pair_swap_replay.json
```

只在已有分配之后使用单侧参照补点，`076` 下降 0.27 分、`085` 增加 0.06 分，暂不并入。下一步可用稳定双目标的相对俯仰提前约束候选分配，覆盖两个目标都没有可靠初始分配的情况。[相对俯仰诊断](../reports/relative_elevation_oracle.json) · [组合消融](../reports/relative_elevation_after_pairs_sample.json)

## 联合分配与补充起轨

联合分配在单轨选择前比较两个稳定目标的观测组合。两条航迹都缺少可靠单独匹配时，也能用过去共同圈的相对俯仰约束同时求解。它比较分配给这两个目标与分配给其他目标的总收益，避免抢走第三条轨的可靠点。

补充起轨保留原起轨流程，为剩余点保留最近五圈中的三点路径。已有成熟轨、本圈暂缓分配和原流程已确认的观察优先占用证据。其余符合几何与幅度条件的路径用 `causal_birth_paths` 模型评分；成对路径还需相对俯仰稳定，双方的历史与当前观测均须互不重叠。较短的历史可以间隔一圈，以恢复连续两击链失效的目标。

| `071–085` | 上轮成对版 | 联合分配 | 联合分配与补充起轨 | 加严格提前确认 |
| --- | ---: | ---: | ---: | ---: |
| 本地代理分 | 85.5240 | 86.1160 | 87.3916 | 87.5219 |
| 首次真点命中口径 | 85.1627 | 85.7735 | 87.1548 | 87.2965 |
| 换号 | 4,502 | 4,129 | 4,212 | 4,199 |
| 拆轨 | 912 | 893 | 895 | 890 |
| 合轨 | 889 | 852 | 848 | 841 |
| 平均航迹覆盖率 | 92.33% | 92.92% | 93.73% | 93.87% |
| IDF1 | 80.17% | 82.03% | 81.94% | 82.01% |

六个训练包的归因表明，702 条真航迹中 336 条在第三个真实观测之后才首次回传；最大的单项原因是第二个真点未接回首个种子（114 条）。全量用三点路径替换原起轨导致四包回落，补充方案的有效增益主要来自时延和覆盖。继续追踪独立的 ID 指标，避免用增加输出掩盖换号。[延迟归因](../reports/teacher_pair_coverage_diagnostic.json) · [联合分配报告](../reports/joint_elevation_pairs_tuning.json) · [补充起轨报告](../reports/joint_paired_birth_tuning.json)

从 `WORK` 读取已有权重复跑：

```powershell
python -B experimental/causal_birth_paths.py tune --packets 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 --clock experimental/models/scan_clock.txt --model experimental/models/causal_birth_paths.txt --mode supplement --joint --paired-birth --two-hit-pairs --floor .86 --jobs 2 --report reports/joint_two_hit_birth_replay.json
```

提前确认仅用于两个独立的高可信观测序列，两圈间相对俯仰变化小于窄门，且方位和距离变化吻合。这 15 个调参包有 264 次此类确认。`079` 单进程回放 P99 为 60.08 ms、最慢圈 439.64 ms，满足压力预算。[提前确认回放](../reports/joint_two_hit_birth_tuning.json) · [压力报告](../reports/joint_two_hit_birth_stress_079.json)

补充模型训练仅用 `001/006/014/022/035/056`，样本和参数见[训练报告](../reports/causal_birth_paths_training.json)。在最高分配置上对未受联合锁定的边加入残差评分，换号进一步降到 3,900，IDF1 升到 83.36%，但拆轨增加 25 条且总分为 87.4644，作为 ID 对照配置保留。[残差组合报告](../reports/joint_two_hit_residual_tuning.json)

近期历史已有错配时，训练集诊断中 96.5% 的正确观测仍在空间门内，但只有 38.7% 通过原关联概率门限。直接放宽门或单次重锚的闭环收益不足。后续优先验证连续几圈的联合证据，以及在含有杂波和缺测的候选中直接识别成对观测归属。[状态诊断](../reports/damaged_gate_audit.json) · [重锚消融](../reports/joint_birth_reanchor_sample.json)

## 相邻回波的幅度差与单侧缺测

训练包中，许多近邻目标在相邻的雷达时间刻度各有一条回波，且跨圈的有符号幅度差稳定到约百万分之一。`paired_returns.py` 用当前圈的两点几何、幅度差和过去的观测确认关系。确认后，两条原航迹先作联合匹配，其他点迹进入普通分配。两次都观测到完整双回波的高置信新候选可在第二次观测起批。已确认双轨最长保留 12 圈；第 8–12 圈只允许通过观测到的成对证据恢复 ID。

训练诊断找到 297 次精确成对关联恢复了上一条普通关联切走的 ID。其中 273 次上一圈只有一侧真回波，24 次两侧都存在但未通过测量筛选。真实单侧回波的归属模型在独立训练包中只有 65.74% 的选边准确率。提高置信门槛至 0.90 时精度为 94.84%，覆盖 5.31%。配置采用严格的单侧回退门限，随后用独立的回传判别恢复高置信真点并筛选杂波，承担一部分覆盖率损失。[缺测诊断](../reports/paired_return_residual_audit_roles.json) · [单侧归属报告](../reports/pair_arm_orientation_train.json)

| `071–085` 同包比较 | 联合分配与补充起轨 | 精确成对 | 成对、回退保护与回传判别 |
| --- | ---: | ---: | ---: |
| 本地代理分 | 87.5219 | 91.5542 | 93.1138 |
| 首次真点命中口径 | 87.2965 | 91.4368 | 93.0712 |
| 换号 | 4,199 | 1,946 | 535 |
| 拆轨 | 890 | 652 | 337 |
| 合轨 | 841 | 533 | 99 |
| IDF1 | 82.01% | 91.53% | 93.09% |
| 平均航迹覆盖率 | 93.87% | 95.05% | 93.42% |
| 虚警比例 | 7.17% | 7.17% | 3.66% |

从 `WORK` 复跑这两个配置：

```powershell
$env:PYTHONPATH = '.;participant_demo/src'
python -B experimental/paired_returns.py tune --packets (71..85) --jobs 2 --report reports/paired_returns_replay.json
python -B experimental/emission_identity.py tune --paired experimental/models/paired_returns.txt --wide-pairs --fallback-floor .96 --packets (71..85) --jobs 2 --report reports/paired_returns_protected_emission_replay.json
```

组合的成绩取 `rescue65_clutter10` 项。此阶段的 `071–085` 总分距 95 分为 1.89 分。精确成对版的 `079` 顺序回放 P99 为 74.60 ms，最慢圈 405.24 ms。[训练报告](../reports/paired_returns_training.json) · [精确成对回放](../reports/paired_returns_tuning.json) · [组合回放](../reports/paired_returns_protected_emission_tuning.json) · [精确成对版 079 压测](../reports/paired_returns_stress_079.json)

## 关联前的点迹过滤

训练包的真点大多紧跟另一个雷达时间刻度。把前后测量的时间间隔、相位、坐标和幅度合在一起，可在关联和起轨前过滤大部分已标注杂波。时间戳大量重叠或整圈相位不可靠时回到原流程，已确认的成对观测直接使用自己的关联证据。模型用 `006/017/024/035/051/063/069` 拟合，`011/026/052` 按整包检查。阈值 0.025 在三个检查包中保留 99.25%–99.71% 的已标注真点和 3.11%–3.82% 的已标注杂波。`-1` 样本未参与拟合。[检测器训练报告](../reports/raw_timing_detection_train.json)

在上一组合上前置过滤，并使用 `rescue_65` 回传策略后，`071–085` 得 **93.5969**，首次真点命中口径 **93.5553**。换号 **470**，拆轨 **303**，合轨 **89**，IDF1 **94.22%**，平均航迹覆盖率 **93.49%**，虚警 **3.08%**。原始 `079` 压力包的单进程 P99 为 **35.67 ms**，8,797 点的第 567 圈耗时 **102.55 ms**，五项分数与全调参回放中同包结果完全一致。同一配置在 Docker 中与原版服务端跑完 `train_001.dat` 的 720 圈，输出和耗时文件通过服务端检查。[全量调参报告](../reports/paired_returns_raw025_emission_tuning.json) · [079 单进程压测报告](../reports/paired_returns_raw025_emission_stress_079.json) · [真实 TCP 联调](../reports/paired_returns_raw025_tcp_001.json)

当原始点时序模型置信度至少 0.96，且航迹归属仍有足够证据时，`rescue65_raw96` 多回传部分真点，使总分达到 **93.8166**、首次真点命中口径 **93.7781**，换号 **490**、拆轨 **312**、IDF1 **94.27%**。`rescue25_raw90` 总分高约 0.0022，另有两次换号和一条拆轨。`rescue65_raw96` 的 `079` 回放与全量五项得分精确一致，P99 **28.20 ms**，最慢圈 **98.50 ms**。[回传策略消融](../reports/paired_returns_raw025_emission_recovery_tuning.json) · [回传策略压测](../reports/paired_returns_raw025_rescue65_raw96_stress_079.json)

```powershell
$env:PYTHONPATH = '.;participant_demo/src'
python -B experimental/emission_identity.py tune --paired experimental/models/paired_returns.txt --wide-pairs --fallback-floor .96 --raw-model experimental/models/raw_timing_detection.txt --raw-floor .025 --packets (71..85) --jobs 2 --report reports/paired_returns_raw025_replay.json
```

前置过滤后降低普通关联门限，让较弱但一致的观测先竞争旧 ID，减少新起轨。已确认双轨的单侧回退门限为 0.96。全量 `071–085` 上，`rescue25_raw90` 的普通关联门限从 0.20 降到 0.12 再降到 0.07，分别得到 **93.8188**、**94.2683**、**94.4755**。0.07 对应 **267** 次换号、**196** 条拆轨、**94.86%** IDF1 和 **93.65%** 平均覆盖率。[0.12 消融](../reports/paired_returns_raw025_assoc012_emission_tuning.json) · [0.07 消融](../reports/paired_returns_raw025_assoc007_emission_tuning.json)

普通关联门限 0.035、回传策略 `rescue25_raw80` 得 **94.4829**，首次真点命中口径 **94.4434**，换号 **260**，拆轨 **194**，合轨 **80**，IDF1 **94.91%**。平均覆盖率 **93.64%**，虚警 **3.13%**。`079` 的单进程 P99 为 **38.15 ms**，最慢圈 **157.85 ms**，五项得分与全量报告相同。验收与泛化数据仍按原定划分保留。[全量调参](../reports/paired_returns_raw025_assoc0035_emission_tuning.json) · [079 单进程压测](../reports/paired_returns_raw025_assoc0035_stress_079.json)

```powershell
$env:PYTHONPATH = '.;participant_demo/src'
python -B experimental/emission_identity.py tune --paired experimental/models/paired_returns.txt --wide-pairs --fallback-floor .96 --raw-model experimental/models/raw_timing_detection.txt --raw-floor .025 --association-floor .035 --packets (71..85) --jobs 2 --report reports/paired_returns_raw025_assoc0035_replay.json
```

## 高速双目标的起批和续接

训练包 `001/006/014/022` 的 63 个迟起批目标中，36 个首两点的角位移超出起轨门，13 个观测间隔超出候选池范围。扩大成对起批门可以提前发布真正的双目标。训练诊断中，18 组在起批后第一圈失去公开 ID，其中 16 组仍有精确配对，位移超过了原来的续接门。[迟起批诊断](../reports/raw_birth_stage_training_sample.json) · [起批后丢失诊断](../reports/fast_pair_persistence_train_sample.json)

组合先执行原来的成对关联。当两个已登记 ID 仍存续、当前两点均未分配，且其有符号幅度差和方向一致时，允许用速度补偿后的中心门恢复原 ID。剩余观测才参与扩大的成对起批，两步都要求候选唯一。`071/076/079/085` 上，原配置、仅扩大起批、仅扩大续接、同时扩大两者的得分分别为 **94.5596**、**94.2707**、**95.5954**、**96.2253**。[四包消融](../reports/registered_pair_recovery_fast_sample.json)

完整 `071–085` 的组合得分为 **96.0047**。换号从 260 降至 150，拆轨从 194 降至 103，平均覆盖率从 93.64% 升至 94.78%，IDF1 从 94.91% 升至 95.21%，合轨点从 1,810 增至 2,214。511 次新增双轨续接的前后对应均匹配同一真实目标，56 次快速成对起批的两条观测归属均正确。部署入口的 15 个逐包回放分数和身份计数与实验版完全一致。[组合调参](../reports/registered_pair_recovery_fast_tuning.json) · [部署入口调参复现](../reports/precision_runtime_tuning.json)
