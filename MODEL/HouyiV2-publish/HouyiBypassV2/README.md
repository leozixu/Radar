# 后羿智能化雷达目标检测系统 - 使用指南

> 版本号：2.0

---

## 一、系统简介

本系统用于对雷达点迹数据进行**智能目标检测**。程序读取点迹，经模型智能化处理后，输出每个点的**目标置信度**（`point_class_probs`）与**点-点相关性**（`point_corr`）。

---

## 二、准备工作

### 2.1 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 或 Linux |
| Python | 3.12 |
| GPU | NVIDIA GPU |

### 2.2 依赖项

- **Python**: 3.12（推荐本版本）
- 依赖安装：在项目根目录执行 `pip install -r requirements.txt`
  - 关键依赖：`torch`、`transformers`、`peft`、`numpy`、`scipy`、`pandas`、`typer`、`click` 等（具体版本见 `requirements.txt`）
  - 特别依赖：`route_tools`，后羿发布工具库，安装包见`route_tools-0.3.1-py3-none-any.whl`

### 2.3 准备模型权重

在 `Model` 目录下放置模型权重（`model_association`训练过程中形成的model.pth、model_config.json以及data_config.json文件）：

```
Model/
├── model.pth            # 基模型权重
├── model_config.json    # 模型结构配置
└── data_config.json     # 数据归一化配置
```

若使用 LoRA 微调：指定 `--lora-weight-path`，指向 LoRA 适配器目录（可选）。LoRA 目录内容与 `Model` 目录形式相同：

```
LoRA/                        # LoRA 适配器目录
├── adapter_config.json      # LoRA 适配器配置（由 PEFT 生成，含 lora_r/alpha 等）
├── adapter_model.safetensors # LoRA 适配器权重（由 PEFT 生成）
└── data_config.json         # 数据归一化配置（使用 LoRA 时由此目录加载）
```

> **说明**：
> - `adapter_config.json` 与 `adapter_model.safetensors`：模型进行LoRA训练时产生的LoRA 适配器文件，`--lora-weight-path` 指向该目录后，程序通过 `PeftModel.from_pretrained` 加载并合并到基模型。
> - `data_config.json`：使用 LoRA 时，程序会**从 LoRA 目录**读取数据归一化配置（`apps.py` 中 `data_config_path` 优先取 `lora_weight_path`），而非 `Model` 目录。
> - `model_config.json`：始终从基模型目录（`Model/`）加载，LoRA 目录无需提供。
> - 仅使用基模型时，只需 `Model` 目录，与 LoRA 目录无关，命令行不加`--lora-weight-path`参数。

### 2.4 准备数据

回放模式需要准备点迹数据（**CSV 文本文件**）：

```
data/                        # 数据目录
├── train_001.csv           # 点迹数据文件
├── train_002.csv
└── ...
```

**数据文件要求**：
- 文件格式：`.csv` 文本，首行为列名，其后每行一条点迹记录
- 文件命名：任意名称，以 `.csv` 结尾

**输入数据列说明**：

| 列名 | 含义 | 单位/取值范围 | 说明 |
|------|------|--------------|------|
| `Pt_RadarTime` | 雷控时间 | ms（日内折算） | 必填，用于时间窗排序与圈号划分 |
| `Pt_Azi` | 方位 | 归一化（0–1） | 必填 |
| `Pt_El` | 俯仰 | 归一化（0–1） | 必填 |
| `Pt_Distance` | 距离 | 归一化（0–1） | 必填 |
| `Pt_Amp` | 幅度 | 归一化（0–1） | 必填 |
| `Pt_SNR` | 信噪比 | 归一化（0–1） | 必填 |
| `Pt_Circle` | 圈号 | 整数 | 必填 |
| `Pt_PtID` | 点序号 | 整数，从 1 起始 | 必填（程序会为缺失点重新编号） |
| `Batch_ID` | 真值轨迹标签 | 整数的批号，`0` 表示非目标点 | 输入真值轨迹标签，用于评估算法检测结果 |

---

## 三、运行方法

### 步骤 1：配置运行参数

编辑 `settings/proc_config.json`：

- `time_range` / `time_step` / `time_delay`：回放时间范围与步进（**单位毫秒**）
- `model_mode`：`base`（加载 `Model/model.pth`）或 `LoRA`（配合 `--lora-weight-path`）
- `cluster_config`：是否开启点迹聚类（`on-off`）及聚类阈值
- `track_config`：航迹关联阈值（`speed_thresh_*`、`acceleration_thresh`、`direc_change_thresh` 等）。目标检测模式已不使用航迹关联，相关阈值可保持不变或按需调整
  - `data_config.json` 会在启动时从模型权重目录（`Model/`）自动加载

> **注**：
>
> - 输入数据时间单位为**毫秒**，对应的`proc_config.json`中`time_range`/`time_step`/`time_delay` 亦为毫秒，而其他与时间相关的配置项则以**秒**为单位。

---

## 三、运行方法

### 步骤 2：运行程序

在 `code` 目录下，使用命令行直接运行：

```bash
python main.py run-playback-tracker data_dir model_weight_path --save-dir <输出目录>
```

示例：

```bash
python main.py run-playback-tracker ../data ../Model/model.pth --num-point-features 72 --log-level INFO --save-dir ../Output
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data_dir` | 回放数据目录（包含 `.csv` 文件） | —（必填） |
| `model_weight_path` | 基模型权重文件路径（`Model/model.pth`） | —（必填） |
| `--lora-weight-path` | LoRA 适配器目录（含 `adapter_config.json`），注入基模型 | 无（可选） |
| `--num-point-features` | 每个点的特征数 | `72` |
| `--log-level` | 日志级别（CRITICAL/FATAL/ERROR/WARNING/INFO/DEBUG/NOTSET） | `DEBUG` |
| `--proc-config-path` | 处理配置路径 | `../settings/proc_config.json` |
| `--save-dir` | 输出目录（自动创建，用于存放运行日志） | 无（可选） |

> 注：程序默认使用 CUDA 设备（`torch.device("cuda")`），需保证 GPU 可用。

### 步骤 3：查看结果

程序运行日志保存在 `--save-dir` 指定的输出目录，文件名 `playback_tracker.log`。
（目标检测结果 `point_class_probs`/`point_corr` 在 `apps.py::inference()` 内计算，可通过Debug断点到该处查看具体结果。）

---

## 四、目标检测输出说明

`apps.py::inference()` 返回前会计算以下模型输出：

| 变量 | 含义 | 形状 |
|------|------|------|
| `point_class_probs` | 每个点的目标分类概率（**目标检测核心输出**），下标 `1` 为目标概率 | `(n_areas, sample_len, 2)` |
| `point_corr` | 点-点相关性矩阵，反映同一扇区内任意两点属于同一目标的程度（原用于航迹关联） | `(n_areas, sample_len, sample_len)` |

> 这两个值在当前版本中存在 `inference()` 局部变量中，方便集成方在 `run()` 循环中按需取用。以下原本用于航迹关联的中间量已在 `apps.py` 中以注释形式保留说明：
>
> - `data_clustered`：预处理+聚类后的当前批点迹
> - `indices_in_hist` / `indices_in_curr`：标记历史点/当前批新点
> - `grouped_corr_mat_all_points` / `grouped_prob_mat_all_points`：汇总后的全量相关性矩阵与分类概率
> - `grouped_corr_mat` / `grouped_indices_in_hist` / `grouped_indices_in_curr`：按「历史-当前」分块的相关性矩阵及索引

---

## 五、常见问题

**Q1：找不到模型文件**

- 检查 `Model/` 目录下是否有 `model.pth`、`model_config.json`、`data_config.json`

**Q2：CUDA out of memory**

- 查看显存是否够用：`sample_len:1000` 至少需 2GB 可用显存；`sample_len:3000` 至少需 8GB；`sample_len:5000` 至少需 24GB
- 将 `settings/proc_config.json` 中 `time_step` 设为 1

**Q3：模型加载报 size mismatch（extra_n_in_features）**

- 模型 `model_config.json` 中 `extra_n_in_features > 0` 时，`code/model.py` 会按 `+ extra_n_in_features` 构建输入维数，并在输入末尾补零。若仍报错，请确认 `Model/model_config.json` 与 `Model/model.pth` 来自同一次训练

**Q4：特征增强报 "must take value in range [0,1]"**

- 若某特征（如高程）归一化后超出 [0,1]，且增广频率为整数时，新版 `sine_augment` 会自动按 `floor` 折回 [0,1)；若频率非整数则仍会报错。此类问题多为 `Model/data_config.json` 的归一化配置与数据量程不匹配，需检查 `elev` / `az` 等 `normalization_factor`

---

**最后更新**: 2026 年 11 月
