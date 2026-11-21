#  Copyright 2025 NKLRDS AI team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  SPDX-License-Identifier: Apache-2.0

"""
Apps for target detection (tracking removed).

Author: Junxiang Xiao.

"""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from model_association.point_cc.config import (
    MODEL_CONFIG_FILENAME,
    DATA_CONFIG_FILENAME,
    DataConfig,
)
from model_association.utils import setup_logger

from model_association import radar_definition as rd
from data import DataProcessorBase, DataProcessor
from data.data_queue import (
    DataQueueBase,
    LocalFileDataQueue,
    TimeDelayedLocalFileDataQueue,
    AllLocalFilesVisited,
)
from model import (
    ModelProcessor,
    group_corr_mats_all_points,
    group_corr_mats,
)


class TrackerAppBase(abc.ABC):
    """
    A target detection application (tracking functionality removed).

    """

    def __init__(
        self,
        log_level=logging.INFO,
        log_file_path: Optional[Path] = None,
        *args,
        **kwargs,
    ):
        self.logger = setup_logger(
            self.__class__.__name__, log_level=log_level, log_file_name=log_file_path
        )
        self._process_id = 0
        self._historical_points = np.zeros((0, 0), dtype=np.float32)

    @property
    def historical_points(self) -> np.ndarray:
        """当前缓存的历史点迹（形状=(n_hist_points, n_point_features)）。"""
        return self._historical_points

    def initialize_historical_points(self, n_point_features: int):
        """以指定特征数初始化空的历史点迹缓存。"""
        self._historical_points = np.zeros((0, n_point_features), dtype=np.float32)

    def update_historical_points(self, points: np.ndarray):
        """把新的点迹追加到历史缓存。

        Args:
            points: 形状=(n_points, n_point_features) 的点迹数组。
        """
        if points is None or len(points) == 0:
            return
        points = points.astype(np.float32)
        if self._historical_points.size == 0:
            self._historical_points = points
        else:
            self._historical_points = np.concatenate(
                (self._historical_points, points), axis=0
            )

    def clear_historical_points(self):
        """清空历史点迹缓存（保持特征维数不变）。"""
        if self._historical_points.size > 0:
            self._historical_points = np.zeros(
                (0, self._historical_points.shape[1]), dtype=np.float32
            )

    def trim_historical_points(self, max_cache_time_span: float):
        """按时间跨度裁剪历史点迹，丢弃过旧的点（默认不裁剪，可选调用）。

        Args:
            max_cache_time_span: 允许的最大时间跨度（秒）。
        """
        if self._historical_points.size == 0:
            return
        times = self._historical_points[:, rd.PointFeatureIndex.RADAR_TIME.value]
        time_cutoff = times.max() - max_cache_time_span
        keep_mask = times >= time_cutoff
        if not keep_mask.all():
            self._historical_points = self._historical_points[keep_mask].astype(
                np.float32
            )

    def reset_state(self):
        """重置历史状态（在开始处理新数据文件时调用）。

        重置进程号计数器并清空历史点迹缓存。
        """
        self._process_id = 0
        self.clear_historical_points()

    @property
    @abc.abstractmethod
    def data_queue(self) -> DataQueueBase: ...

    @property
    @abc.abstractmethod
    def data_processor(self) -> DataProcessorBase: ...

    @property
    @abc.abstractmethod
    def model_processor(self) -> ModelProcessor: ...

    @property
    @abc.abstractmethod
    def run(self, *args, **kwargs): ...

    def inference(
        self, raw_data, sample_len=5000, cluster=True, *args, **kwargs
    ) -> bool:
        """对一批点迹数据执行目标检测(不再做航迹关联/跟踪)。

        流程: 原始点迹 -> 预处理 -> (可选)聚类 -> 采样生成 -> 模型推理,
        得到每个点的目标分类概率 ``point_class_probs`` 与点-点相关性 ``point_corr``。

        Returns:
            True 表示本批数据有效并已处理; False 表示输入为空。
        """
        if raw_data is None or len(raw_data) == 0:
            return False
        raw_data[:, rd.PointFeatureIndex.TRACK_ID_ALGO.value] = 0
        raw_data[:, rd.PointFeatureIndex.MODEL_CLASS] = 0
        data_preprocessed = self.data_processor.preprocess(raw_data)

        self._process_id += 1
        data_preprocessed[:, rd.PointFeatureIndex.PROCESS_ID.value] = self._process_id
        if self._historical_points.size == 0 or \
                self._historical_points.shape[1] != data_preprocessed.shape[1]:
            self.initialize_historical_points(data_preprocessed.shape[1])
        historical_points = self.historical_points
        if cluster:
            data_clustered = self.data_processor.point_cluster(
                data_preprocessed, historical_points=historical_points
            )
        else:
            data_clustered = data_preprocessed

        if len(data_clustered) == 0:
            return False

        # ---- 以下变量(原为 track_track 的输入)保持计算, 仅用于理解/调试 ----
        # data_clustered: 预处理+聚类后的当前批点迹, 形状=(n_points, n_point_features)。
        #   track_track 用它作为「当前批目标点」的载体。
        all_points = np.concatenate((historical_points, data_clustered), axis=0)
        # indices_in_hist / indices_in_curr: 分别标记 all_points 中哪些是历史点(已关联)、
        #   哪些是当前批新点(待关联); -1 表示不属于该侧。形状=(n_points,)。
        indices_in_hist = np.concatenate(
            (
                np.arange(len(historical_points)),
                np.full(len(data_clustered), fill_value=-1),
            ),
            axis=0,
        )
        indices_in_curr = np.concatenate(
            (
                np.full(len(historical_points), fill_value=-1),
                np.arange(len(data_clustered)),
            ),
            axis=0,
        )
        # ---- 采样生成: 把点迹按空间扇区/时间窗组织成模型可输入的样本 ----
        data_samples, point_indices, area_ids = self.data_processor.sample_generate(
            all_points, self.model_processor.input_feature_indices
        )
        # point_indices: 采样后的每个样本槽位对应 all_points 中的哪一个点(-1 为填充位)。
        #   shape=(n_areas, sample_len)。
        indices_in_hist = indices_in_hist[point_indices]  # shape=(n_areas, sample_len)
        indices_in_hist[point_indices < 0] = -1
        indices_in_curr = indices_in_curr[point_indices]  # shape=(n_areas, sample_len)
        indices_in_curr[point_indices < 0] = -1

        # ---- 模型推理(目标检测) ----
        point_class_probs, point_corr = self.model_processor.process(
            data_samples, area_ids=area_ids
        )
        grouped_corr_mat_all_points, grouped_prob_mat_all_points, grouped_indices_in_hist2 = \
            group_corr_mats_all_points(point_corr, point_class_probs, indices_in_hist, indices_in_curr, sample_len)
        grouped_corr_mat, grouped_indices_in_hist, grouped_indices_in_curr = \
            group_corr_mats(point_corr, indices_in_hist, indices_in_curr)


        # ---- 以下为模型输出的汇总结果----
        # point_class_probs: 每个点的目标分类概率, 形状=(n_areas, sample_len, 2),
        #   下标 1 为目标概率, 是目标检测的核心输出。
        # point_corr: 点-点相关性矩阵, 形状=(n_areas, sample_len, sample_len),
        #   反映同一扇区内任意两点属于同一目标的程度(航迹关联时才用到)。
        # grouped_corr_mat_all_points: 全部(历史+当前)点的相关性矩阵,
        #     形状=(n_hist+n_curr, n_hist+n_curr), 可用于匈牙利匹配。
        # grouped_prob_mat_all_points: 每个点的目标分类概率, 形状=(n_total, 1)。
        # grouped_indices_in_hist2: 各点在总矩阵中的索引。
        # grouped_corr_mat: 按「历史-当前」分块的相关性矩阵(与匹配算法适配的布局)。
        # grouped_indices_in_hist / grouped_indices_in_curr: 历史点/当前点在分块矩阵中的位置。

        # 更新历史点迹缓存: 将当前批点迹追加到历史, 供下一批做关联/去重。
        # 可选: 用 self.trim_historical_points(max_cache_time_span) 按时间跨度裁剪过旧点。
        self.update_historical_points(data_clustered)

        return True

class PlaybackTrackerApp(TrackerAppBase):
    def __init__(
        self,
        data_dir: Path,
        n_point_features: int,
        model_weight_path: Path,
        device,
        proc_params: dict,
        lora_weight_path: Optional[Path] = None,
        log_level=logging.INFO,
        log_file_path: Optional[Path] = None,
        save_dir: Optional[Path] = None,
    ):
        model_config_path = model_weight_path.parent / MODEL_CONFIG_FILENAME
        with model_config_path.open() as f:
            model_config: dict = json.load(f)

        data_config_path = (
            model_weight_path.parent
            if lora_weight_path is None
            else lora_weight_path
        ) / DATA_CONFIG_FILENAME
        data_config = DataConfig.from_file(Path(data_config_path))
        data_config.fill_in_normalization_factors()
        self.sample_num = data_config.sample_len
        self.cluster = proc_params["cluster_config"]["on-off"]
        self._data_queue = TimeDelayedLocalFileDataQueue(
            data_dir,
            time_step=proc_params["time_step"],
            time_delay=proc_params["time_delay"],
            data_length=n_point_features,
            data_file_name_regex=r".*\.csv",
            start_time=proc_params["time_range"]["start_time"],
            end_time=proc_params["time_range"]["end_time"],
            log_level=log_level,
            log_file_path=log_file_path,
        )

        self._data_preprocessor = DataProcessor(
            n_point_features=n_point_features + 24,
            circle_config=proc_params["circle_config"],
            model_config=model_config,
            data_config=data_config,
            cluster_config=proc_params["cluster_config"],
            preprocess_config=proc_params["preprocess_config"],
            log_level=log_level,
            log_file_path=log_file_path,
        )
        self._model_processor = ModelProcessor(
            model_weight_path=model_weight_path,
            data_config=data_config,
            device=device,
            model_mode="base" if lora_weight_path is None else "LoRA",
            log_level=log_level,
            log_file_path=log_file_path,
            lora_weight_path=lora_weight_path,
            only_regress=proc_params["only_regress"],
            reg_the_last_but_one_point=proc_params["reg_the_last_but_one_point"],
        )
        # 已移除跟踪: 原为 self._track_manager = TrackManager(...),
        # 目标检测模式不再需要航迹管理器, 因此不在此构造。
        # 历史点迹缓存由 TrackerAppBase 管理; 宽度取预处理后的特征数(=输入特征数+24)。
        self.initialize_historical_points(n_point_features + 24)

        self.logger = setup_logger(
            self.__class__.__name__, log_level=log_level, log_file_name=log_file_path
        )

    @property
    def data_queue(self) -> LocalFileDataQueue:
        return self._data_queue

    @property
    def data_processor(self) -> DataProcessor:
        return self._data_preprocessor

    @property
    def model_processor(self) -> ModelProcessor:
        return self._model_processor

    def run(self, *args, **kwargs):
        """
        Run the playback target-detection app.

        """
        while True:
            raw_data, state_reset_args = self.data_queue.get()

            # 文件边界: 开始处理新数据文件时, 重置历史状态(进程号计数 + 历史点迹缓存)。
            if state_reset_args:
                self.reset_state()

            if isinstance(raw_data, AllLocalFilesVisited):
                self.logger.info(
                    "All local data files have been visited. Stop running."
                )
                return

            self.inference(raw_data, self.sample_num, cluster=self.cluster)
