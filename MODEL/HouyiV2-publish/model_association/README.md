# 后羿模型训练框架使用说明
本项目为基于神经网络的路由关联（Route Association）发布版本，包含点迹数据预处理、样本生成、模型训练与评估的完整代码。本文档说明该发布版的**样本生成**与**模型训练**两条核心命令的用法。

> **注意**: 本文档中所有运行方式均直接使用 `python` 命令执行脚本，不依赖安装后生成的终端命令（如 `model_asso_gen_dataset_pcc`、`model_asso_train_pcc`）。

---

## 📋 目录

- [功能特性](#-功能特性)
- [安装说明](#-安装说明)
- [使用方法](#-使用方法)
- [配置说明](#-配置说明)
- [示例](#-示例)
- [许可证](#-许可证)

---

## ✨ 功能特性

**主要功能：**
- 从单雷达原始数据仓库采样并生成训练所需的 HDF5 数据集
- 后羿模型训练（支持 LoRA 微调）

**技术亮点：**

- 支持 `from-one-repo` 与 `from-multiple-repos` 两种采样模式
- 支持 LoRA 参数高效微调（`--lora`）

---

## 📦 安装说明

### 系统要求

- **操作系统**: Linux（x86-64 CPU 与 NVIDIA GPU）
- **Python 版本**: 3.10 及以上（发布版建议使用 3.12）
- **内存要求**: ≥ 250 GB（视数据集规模而定）

### 非安装配置（推荐）

若在命令行中直接运行脚本，将源码目录加入 `PYTHONPATH`：

```bash
cd /share/disk2/fyl/model_association_publish
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### 依赖项

**核心依赖：**

```
matplotlib
numba
numpy
pydantic
route_tools（后羿提供安装包route_tools-0.3.1-py3-none-any.whl）
torch
transformers
typer
h5py
peft
```

> 若编译或运行时报缺包，请按 `requirements.txt` 逐项安装。

---

## 🚀 使用方法

> 在命令行中cd进入src子路径下，然后按照以下步骤运行。

### 第一步：数据准备

准备带有目标真值的点迹数据（**CSV 文本文件**）：

```
original_data_dir/      # 数据目录
├── train               # 训练数据集
├──────train_001.csv
├──────train_002.csv
├──────...
├── test                # 测试数据集
└── data_config.json    # 样本生成配置文件
```

**数据文件要求**：

- 文件格式：`.csv` 文本，首行为列名，其后每行一条点迹记录
- 文件命名：任意名称，以 `.csv` 结尾

**输入数据列说明**：

| 列名           | 含义         | 单位/取值范围                | 说明                                   |
| -------------- | ------------ | ---------------------------- | -------------------------------------- |
| `Pt_RadarTime` | 雷控时间     | ms（日内折算）               | 必填，用于时间窗排序与圈号划分         |
| `Pt_Azi`       | 方位         | 归一化（0–1）                | 必填                                   |
| `Pt_El`        | 俯仰         | 归一化（0–1）                | 必填                                   |
| `Pt_Distance`  | 距离         | 归一化（0–1）                | 必填                                   |
| `Pt_Amp`       | 幅度         | 归一化（0–1）                | 必填                                   |
| `Pt_SNR`       | 信噪比       | 归一化（0–1）                | 必填                                   |
| `Pt_Circle`    | 圈号         | 整数                         | 必填                                   |
| `Pt_PtID`      | 点序号       | 整数，从 1 起始              | 必填（程序会为缺失点重新编号）         |
| `Batch_ID`     | 真值轨迹标签 | 整数的批号，`0` 表示非目标点 | 输入真值轨迹标签，用于评估算法检测结果 |

### 第二步：样本生成

从带有真值的原始数据集中采样，写入指定的HDF5样本文件。

```python
python ./src/model_association/point_cc/generate_dataset.py from-one-repo \
    original_data_dir \
    datasets_save_dir \
    --log-level INFO
```

**子命令：**

- `from-one-repo`：单个原始数据仓库采样
- `from-multiple-repos`：同一父目录下多个子目录（每个雷达一个仓库）统一采样

**常用参数：**

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `original_data_dir` | TEXT | 原始数据根目录（须包含 train/test/validate 子目录） | 必填 |
| `datasets_save_dir` | TEXT | 生成的数据集保存目录 | 必填 |
| `--dataset-type` | TEXT | 处理的子集类型（train/test/all） | `all` |
| `--sample-len` | INTEGER | 每个样本的点迹长度 | 5000（来自配置） |
| `--n-features` | INTEGER | 特征维度（不足部分零填充） | 64 |
| `--rewrite` | FLAG | 覆盖已存在的目标文件 | 关闭 |
| `--remove-uncertain-points` | FLAG | 剔除不确定点（标签为 -1 的点） | 关闭 |
| `--reg-the-last-but-one-point` | FLAG | 保留每条航迹倒数第二个点的回归标签 | 开启 |
| `--validate-seed` | INTEGER | 验证集划分随机种子 | 42 |
| `--log-level` | TEXT | 日志级别（INFO/DEBUG…） | INFO |

> 部分命令行 `--*` 选项可覆盖 `<original_data_dir>/data_config.json` 中的配置项。

### 第三步：模型训练

对生成的数据集进行模型训练，支持 LoRA 微调。

```python
python ./src/model_association/point_cc/train_model.py \
    dataset_dir \
    weight_save_dir \
    --model-config /share/disk2/fyl/experiments/model_config.json \
    --weight-ckpt-dir /share/disk3/HouyiV2/HouyiModelV2/release/2.5.17.7-18M \
    --lora \
    --log-level INFO
```

**位置参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset_dir` | TEXT | 处理后数据集目录（须包含 `train.hdf5`、`test.hdf5`、`collected_data_config.json`） |
| `weight_save_dir` | TEXT | 权重保存目录（不存在则自动创建） |

**常用参数：**

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model-config` | TEXT | 模型配置文件路径 | 自动查找 |
| `--weight-ckpt-dir` | TEXT | 预训练权重 checkpoint 目录 | None |
| `--lora` | FLAG | 开启 LoRA 参数高效微调（LoRA 配置在 model_config 的 `lora` 字段中） | 关闭 |
| `--lora-weight-dir` | TEXT | LoRA 适配器权重目录（用于续训，需配合 `--lora`） | None |
| `--only-classify` | FLAG | 仅训练分类头 | 关闭 |
| `--only-regress` | FLAG | 仅训练回归头 | 关闭 |
| `--auto-subdir` | FLAG | 按时间戳自动创建权重子目录 | 开启 |
| `--log-level` | TEXT | 日志级别 | INFO |

> **注意**：`--lora` 为开启 LoRA 微调的关键参数。其 LoRA 配置（rank、alpha、target_modules 等）需在 `--model-config` 指定的 `model_config.json` 内的 `lora` 字段中定义。

---

## ⚙️ 配置说明

### 样本生成配置（data_config.json）

位于原始数据根目录（如 `test_pub/data_config.json`），控制采样与特征归一化参数。关键项：

| 键 | 说明 | 示例 |
|------|------|------|
| `sample_len` | 每个样本的点迹长度 | 5000 |
| `time_step` | 时间步长 | 21 |
| `min_track_len` | 最小航迹长度 | 3 |
| `only_primary_and_fused_points` | 是否仅保留主/融合点迹 | true |
| `normalization_factor` | 各特征归一化系数（time/r/amp/snr/height/elev/az…） | 1 |
| `max_value` | 各特征最大值（越界过滤） | 1 |

### 模型训练配置（model_config.json）

| 键 | 说明 | 示例 |
|------|------|------|
| `input_features` | 输入特征列表（对应 `radar_definition` 索引名） | `["RADAR_TIME","X","Y",…]` |
| `regress_features` | 回归目标特征 | `["RADAR_TIME","X","Y"]` |
| `backbone_type` | 骨干网络类型（bert/qwen2/houyi2） | `bert` |
| `loss_coeffs` | 各损失权重 | `{"classify_coeff":100,"circle_coeff":2000,"regress_coeff":50}` |
| `stop_epoch` | 训练轮数 | 125 |
| `batch_size` | 批大小 | 3 |
| `lora` | LoRA 微调配置（rank/alpha/target_modules…） | 对象 |

---

## 💡 示例

### 完整工作流

```python
# 1. 样本生成（from-one-repo）
python ./src/model_association/point_cc/generate_dataset.py from-one-repo \
    /share/disk2/fyl/Datasets/publish/test_pub \
    /data/raid0/fyl/samples/temp \
    --log-level INFO

# 2. 模型训练（LoRA 微调）
python ./src/model_association/point_cc/train_model.py \
    /data/raid0/fyl/samples/temp \
    /share/disk2/fyl/experiments/test \
    --model-config /share/disk2/fyl/experiments/model_config.json \
    --weight-ckpt-dir /share/disk3/HouyiV2/HouyiModelV2/release/2.5.17.7-18M \
    --lora \
    --log-level INFO
```

---

## 📄 许可证

本项目依据 [Apache License 2.0](LICENSE) 许可。

```text
Copyright 2025-2026 Junxiang Xiao
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 📬 联系方式

- **作者**: Junxiang Xiao (xiaojunxiang@cetc.com.cn)

---

> 若有命令参数变更，请执行对应脚本的 `--help` 查看最新帮助。
