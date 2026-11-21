# 📡 后羿系统使用指南

> **版本号**：2.0

---

## 🚀 快速开始

### 1. 环境准备

- **OS**: Windows >= 10 / Linux / Kylin
- **GPU**: NVIDIA，CUDA >= 11.7 / 12.x（推荐 CUDA 12.8）
- **Python**: 3.12（推荐）

### 2. 依赖安装

两个模块共用同一套依赖。在项目根目录执行：

- **安装 `requirements.txt` 依赖**

  ```bash
  pip install -r HouyiBypassV2/requirements.txt
  ```

  > `model_association` 使用同一份依赖，无需重复安装。

- **单独安装 `route_tools` 本地 wheel**

  `route_tools` 未被收录到公共 PyPI 源，需使用仓库自带的本地 wheel 包 `route_tools-0.3.1-py3-none-any.whl` 安装：

  ```bash
  pip install ./route_tools-0.3.1-py3-none-any.whl
  ```


**主要依赖库：**

- `torch`（含 CUDA 版）、`transformers`、`peft`
- `numpy`、`scipy`、`pandas`
- `typer`、`click`、`h5py`、`numba`
- `route_tools`（本地 wheel 单独安装）

---

## 🏗️ 系统架构总览

本发布版包含 **2 大核心模块**：

```
输入：[带真值的点迹数据 (.csv)]
      ↓
[1. 模型训练]      # model_association
      ↓
[2. 离线推理]      # HouyiBypassV2
      ↓
输出：[目标检测结果 point_class_probs / point_corr]
```

| 模块 | 目录 | 职责 |
|------|------|------|
| 01. 模型训练 | `model_association` | 样本生成（`.csv` → `.hdf5`）、模型训练（支持 LoRA 微调） |
| 02. 离线推理 | `HouyiBypassV2` | 对历史 `.csv` 数据批量离线推理，输出目标检测结果 |

---

## 📑 各模块说明

### 01. 模型训练（model_association）

> **用途**：将带真值的 `.csv` 点迹数据转换为模型可训练的 HDF5 数据集，并进行模型训练（支持 LoRA 参数高效微调）。
> **输入**：（1）带真值的点迹数据（`.csv`）；（2）可选：现有基模型权重。
> **输出**：训练后的模型文件（`model.pth` + `model_config.json` + `data_config.json`）。
> **👉 运行方式**：见 `model_association/README.md` 文档。

### 02. 离线推理（HouyiBypassV2）

> **用途**：对历史 `.csv` 点迹文件进行批量离线推理，输出每个点的目标检测结果（`point_class_probs` / `point_corr`）。
> **输入**：（1）点迹数据（`.csv`）；（2）模型文件（`model.pth` + 训练阶段产生的 `model_config.json` / `data_config.json`）。
> **输出**：目标检测概率与点-点相关性矩阵。
> **👉 运行方式**：见 `HouyiBypassV2/README.md` 文档。

---

> 各模块的具体环境、数据格式与命令参数，请分别参阅对应目录下的 README 文档。
