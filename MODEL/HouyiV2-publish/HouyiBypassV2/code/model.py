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
Module for model-based processing.

Author: Junxiang Xiao.

"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, List

import numba
import numpy as np
import torch
from model_association.point_cc.classify import target_prob_from_regression
from model_association.point_cc.config import load_model_config, MODEL_CONFIG_FILENAME, BertCCModelConfig, \
    DataConfig
from model_association.point_cc.feature_augment import FeatureAugmenter
from model_association.point_cc.model import BertCCModel
from model_association.route_point_data_processor import HorizontalCoordPaddedFanSplitter
from model_association.utils import setup_logger, unwrap_state_dict
from transformers import BertConfig

from model_association import radar_definition as rd


class ModelProcessor:
    """
    Process points to be associated based on model.

    """

    def __init__(
        self,
        model_weight_path: Path,
        data_config: DataConfig,
        device: torch.device = torch.device("cuda"),
        model_mode: str = "base",
        log_level=logging.INFO,
        log_file_path: Optional[Path] = None,
        lora_weight_path: Optional[Path] = None,
        only_regress: bool = False,
        r_thresh: float = 1500,  # in meters
        reg_the_last_but_one_point: bool = False,
        *args,
        **kwargs
    ):
        self.logger = setup_logger(
            self.__class__.__name__,
            log_level=log_level,
            log_file_name=log_file_path
        )

        self.model_weight_path = model_weight_path
        self.reload_sign_path = model_weight_path.parent / model_weight_path.name.replace(".pth", "_refresh.txt")
        if not model_weight_path.is_file():
            raise ValueError(
                f"Model weight path does not point to a valid file: "
                f"{model_weight_path}"
            )
        self.device = device

        # -------------------- Load model config --------------------
        model_config_search_paths = [self.model_weight_path.parent / MODEL_CONFIG_FILENAME,
                                     self.model_weight_path.parent.parent / MODEL_CONFIG_FILENAME]
        model_config_obj, model_config_path = load_model_config(
            model_config_search_paths,
            config_type=BertCCModelConfig,
            logger=self.logger
        )
        if model_config_obj is None:
            raise ValueError(f"Failed to load model config (search paths: {model_config_search_paths})")
        self.model_config: BertCCModelConfig = model_config_obj
        self.input_feature_indices: Tuple[rd.PointFeatureIndex, ...] = tuple(self.model_config.input_features)
        self.regress_feature_indices: Tuple[rd.PointFeatureIndex, ...] = tuple(filter(
            lambda f_idx: f_idx in self.input_feature_indices,
            self.model_config.regress_features
        ))
        self.feature_aug_freqs = self.model_config.feature_aug_freqs
        self.regress_feature_aug_freqs = self.model_config.regress_feature_aug_freqs
        self.input_aug_feature_indices: Tuple[rd.PointFeatureIndex] = tuple(filter(
            lambda f_idx: f_idx in self.input_feature_indices,
            self.model_config.input_aug_features
        ))
        self.regress_aug_feature_indices: Tuple[rd.PointFeatureIndex] = tuple(filter(
            lambda f_idx: f_idx in self.regress_feature_indices,
            self.model_config.regress_aug_features
        ))
        if not set(self.input_aug_feature_indices).issubset(set(self.feature_aug_freqs.keys())):
            raise ValueError(
                "Input augmented features are not subset of feature indices "
                "in feature_aug_freqs."
            )
        if not set(self.regress_aug_feature_indices).issubset(set(self.regress_feature_aug_freqs.keys())):
            raise ValueError(
                "Regress augmented features are not subset of feature indices "
                "in regress_features_aug_freqs."
            )
        if len(self.input_aug_feature_indices) > 0:
            self.input_feature_augmenter = FeatureAugmenter(
                feature_indices=self.input_feature_indices,
                augment_freqs=OrderedDict([
                    (f_idx, self.feature_aug_freqs[f_idx])
                    for f_idx in self.input_aug_feature_indices
                ]),
                norm_to_01=self.model_config.regressor_output_sigmoid
            )
        else:
            self.input_feature_augmenter = None

        if len(self.regress_aug_feature_indices) > 0:
            self.regress_feature_augmenter = FeatureAugmenter(
                feature_indices=self.regress_feature_indices,
                augment_freqs=OrderedDict([
                    (f_idx, self.regress_feature_aug_freqs[f_idx])
                    for f_idx in self.regress_aug_feature_indices
                ]),
                norm_to_01=self.model_config.regressor_output_sigmoid
            )
        else:
            self.regress_feature_augmenter = None

        self.only_regress = only_regress
        self.r_thresh = r_thresh
        self.reg_the_last_but_one_point = reg_the_last_but_one_point

        self.xy_indices_in_regress_output = np.array([
            self.regress_feature_indices.index(rd.PointFeatureIndex.X),
            self.regress_feature_indices.index(rd.PointFeatureIndex.Y)
        ])

        # -------------------- Load data config --------------------
        self.data_config = data_config

        self.point_splitter = HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            r_max=self.data_config.max_value.r,
            area_width=np.deg2rad(self.data_config.split_azim_width),
            pad_width=self.data_config.split_pad_width,
            r1=self.data_config.r1
        )

        # -------------------- Create raw model --------------------
        self.extra_n_in_features = getattr(self.model_config, "extra_n_in_features", 0)
        n_in_features = (len(self.input_feature_indices)
                         + sum([len(self.feature_aug_freqs[f_idx])
                                for f_idx in self.input_aug_feature_indices]) * 2) \
                        + self.extra_n_in_features
        n_out_features = (len(self.regress_feature_indices)
                          + sum([len(self.regress_feature_aug_freqs[f_idx])
                                 for f_idx in self.regress_aug_feature_indices]) * 2) \
                         * (3 if self.reg_the_last_but_one_point else 2)

        enc_config = BertConfig(embedding_size=len(self.input_feature_indices),
                                hidden_size=self.model_config.d_model,
                                num_hidden_layers=self.model_config.n_layers,
                                num_attention_heads=self.model_config.n_attn_heads,
                                intermediate_size=self.model_config.d_ff)
        model = BertCCModel(
            n_in_features=n_in_features,
            n_out_features=n_out_features,
            enc_config=enc_config,
            n_hidden_layers_of_classifier=self.model_config.n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor=self.model_config.n_hidden_layers_of_regressor,
            regressor_output_sigmoid=self.model_config.regressor_output_sigmoid,
            use_bert_regressor=self.model_config.use_bert_regressor,
            use_bert_classifier=self.model_config.use_bert_classifier
        )
        self.logger.info(f"Created model using class: %s", type(model).__name__)

        # -------------------- Load base model weight checkpoint --------------------
        loaded = torch.load(
            self.model_weight_path,
            map_location=torch.device("cpu"),
            weights_only=True
        )
        if isinstance(loaded, dict):
            loaded_state_dict = loaded
        else:
            loaded_state_dict = loaded.state_dict()
        model.load_state_dict(unwrap_state_dict(loaded_state_dict), strict=False)
        self.logger.info(f"Loaded base model weight successfully: {self.model_weight_path}")
        model.load_state_dict(torch.load(model_weight_path, map_location=self.device))

        # -------------------- Load LoRA adapter checkpoint --------------------
        # TODO: to be tested
        if model_mode == 'LoRA':
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_weight_path)
            model = model.merge_and_unload()

        self.model = model.to(self.device)
        self.model.eval()

    def update(self):
        if self.reload_sign_path.is_file():
            self.model.load_state_dict(torch.load(self.model_weight_path, map_location=self.device))
            self.reload_sign_path.unlink()
            self.logger.info(
                "model reloaded from %s",
                self.model_weight_path
            )

    def process(
        self,
        points: np.ndarray,
        area_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            points: Processed (e.g. normalized) data of points to be associated,
                shape=(batch_size, sample_len, len(self.input_feature_indices)).
            area_ids: Area IDs of each sample, shape=(batch_size,).

        Return:
            Classification probabilities of points, shape=(batch_size, sample_len) and
            correlations between points, shape=(batch_size, sample_len, sample_len)

        """
        all_probs = []
        all_corr = []
        max_batch_size = 3
        n_base = len(self.input_feature_indices)
        n_extra = self.extra_n_in_features
        for i in range(0, len(points), max_batch_size):
            chunk_points = points[i:i + max_batch_size]
            with torch.no_grad():
                points_tensor = torch.from_numpy(
                    chunk_points
                ).to(device=self.device, dtype=torch.float32)
                # `chunk_points` 末尾可能带有额外输入节点(extra_n_in_features, 如 ACE)。
                # 与训练侧一致: 特征增强只作用于**基础**特征, 额外节点在增强之后追加。
                # 先分离基础/额外节点, 若宽度不足则走原型补零兜底。
                if points_tensor.shape[-1] >= n_base + n_extra:
                    base_tensor = points_tensor[..., :n_base]
                    extra_tensor = points_tensor[..., n_base:n_base + n_extra]
                else:
                    base_tensor = points_tensor
                    extra_tensor = None
                if self.input_feature_augmenter is not None:
                    points_tensor_aug = self.input_feature_augmenter.augment(base_tensor)
                    points_tensor = torch.concatenate(
                        [base_tensor, points_tensor_aug],
                        dim=-1
                    )
                else:
                    points_tensor = base_tensor
                points_tensor[points_tensor.isnan()] = 0.
                if n_extra > 0:
                    if extra_tensor is not None:
                        extra_tensor = torch.nan_to_num(extra_tensor, nan=0.0)
                        points_tensor = torch.concatenate(
                            [points_tensor, extra_tensor], dim=-1
                        )
                    else:
                        points_tensor = torch.nn.functional.pad(
                            points_tensor, pad=(0, n_extra)
                        )
                outputs = self.model(
                    points_tensor
                )
                if not self.only_regress:
                    chunk_probs = outputs[0].detach().cpu().numpy()
                else:
                    regress_output = outputs[1]

                    endpoints_preds_list: List[torch.Tensor] = []
                    n_extra_loc_points = (1 if self.reg_the_last_but_one_point else 0)
                    if self.regress_feature_augmenter is not None:
                        target_preds_aug_in_batch_tn = \
                            regress_output[:, :, len(self.regress_feature_indices) * (2 + n_extra_loc_points):]
                        for i in range(2 + n_extra_loc_points):
                            col_start = i * self.regress_feature_augmenter.n_aug_features
                            col_end = col_start + self.regress_feature_augmenter.n_aug_features
                            _preds = self.regress_feature_augmenter.recover(
                                target_preds_aug_in_batch_tn[:, :, col_start:col_end],
                                indices_of_feat_to_recover=[rd.PointFeatureIndex.X, rd.PointFeatureIndex.Y]
                            )
                            endpoints_preds_list.append(_preds)
                    else:
                        for i in range(2 + n_extra_loc_points):
                            _preds = \
                                regress_output[:, :,
                                self.xy_indices_in_regress_output + i * len(self.regress_feature_indices)]
                            endpoints_preds_list.append(_preds)
                    endpoints_coord_preds = torch.concatenate(endpoints_preds_list, dim=-1)
                    # normalized (x_start, y_start, x_end, y_end, x_loc1, y_loc1, ...)
                    # shape=(batch_size, sample_len, 2 * (2 + n_extra_loc_points))

                    x_min, x_max, y_min, y_max = self.point_splitter.circumrect_corners[area_ids].T
                    area_inverse_transforms = self.point_splitter.areas_inverse_transforms[area_ids]  # shape=(batch_size, 2, 2)
                    x_norm_factor_all = x_max - x_min
                    y_norm_factor_all = y_max - y_min
                    norm_factor_all = torch.tensor(
                        np.repeat(
                            np.array([x_norm_factor_all, y_norm_factor_all]).T,
                            # shape=(batch_size, sample_len, 2)
                            2 + n_extra_loc_points, axis=-1
                        ),
                        device=self.device
                    )  # shape=(batch_size, sample_len, 2 * (2 + n_extra_loc_points))
                    endpoints_coord_preds_denorm = (
                        endpoints_coord_preds.clone()
                        * norm_factor_all
                        + torch.tensor(
                        np.repeat(
                            np.array([x_min, y_min]).T,
                            # shape=(batch_size, sample_len, 2)
                            2 + n_extra_loc_points, axis=-1
                        ), device=self.device
                    )
                    )  # shape=(batch_size, sample_len, 2 * (2 + n_extra_loc_points))

                    target_probs = target_prob_from_regression(
                        points_xy=points[..., [rd.PointFeatureIndex.X, rd.PointFeatureIndex.Y]],
                        points_xy_start=endpoints_coord_preds_denorm[..., :2].cpu().numpy()
                                        @ area_inverse_transforms,
                        points_xy_end=endpoints_coord_preds_denorm[..., 2:4].cpu().numpy()
                                      @ area_inverse_transforms,
                        threshold=self.r_thresh
                    )
                    chunk_probs = np.stack(
                        [1 - target_probs, target_probs], axis=-1
                    )  # shape=(batch_size, sample_len, 2)
                chunk_corr = self.model.get_correlation().detach().cpu().numpy()
                del points_tensor
                del outputs
                torch.cuda.empty_cache()
            all_probs.append(chunk_probs)
            all_corr.append(chunk_corr)
        return np.concatenate(all_probs, axis=0), np.concatenate(all_corr, axis=0)


@numba.njit(cache=True)
def group_corr_mats_all_points(
    corr_mats: np.ndarray,
    point_class_probs: np.ndarray,
    ids_in_historical_points: np.ndarray,
    ids_in_current_points: np.ndarray,
    sample_len: int = 5000,
    max_cach_len: int = 60000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        corr_mats: shape=(batch_size, sample_len, sample_len)
        point_class_probs
        ids_in_historical_points: shape=(batch_size, sample_len), minus value for
            non-historical point.
        ids_in_current_points: shape=(batch_size, sample_len), minus value for
            point not to be associated.
        sample_len
        max_cach_len
    """
    returned_pids_hist = np.unique(ids_in_historical_points)
    returned_pids_hist = returned_pids_hist[returned_pids_hist >= 0]
    returned_pids_curr = np.unique(ids_in_current_points)
    returned_pids_curr = returned_pids_curr[returned_pids_curr >= 0]

    if returned_pids_hist.size + returned_pids_curr.size > sample_len:
        returned_pids_hist = returned_pids_hist[-sample_len + returned_pids_curr.size:]

    n_hist_points = returned_pids_hist.size
    n_curr_points = returned_pids_curr.size
    n_total = n_hist_points + n_curr_points

    grouped_corr_all_points = np.zeros((n_total, n_total), dtype=np.float64)
    grouped_prob_all_points = np.zeros((n_total, 1), dtype=np.float64)

    pid_hist_to_idx_map_dict = np.zeros((ids_in_historical_points.shape[0], max_cach_len)) - 1
    for i_batch in range(ids_in_historical_points.shape[0]):
        for i_point in range(ids_in_historical_points.shape[1]):
            if ids_in_historical_points[i_batch, i_point] >= 0:
                pid_hist_to_idx_map_dict[i_batch, int(ids_in_historical_points[i_batch, i_point])] = i_point

    pid_curr_to_idx_map_dict = np.zeros((ids_in_current_points.shape[0], max_cach_len)) - 1
    for i_batch in range(ids_in_current_points.shape[0]):
        for i_point in range(ids_in_current_points.shape[1]):
            if ids_in_current_points[i_batch, i_point] >= 0:
                pid_curr_to_idx_map_dict[i_batch, int(ids_in_current_points[i_batch, i_point])] = i_point

    num_batch = ids_in_current_points.shape[0]

    for i, i_point in enumerate(returned_pids_hist):

        for i_batch in range(num_batch):
            for j in range(i, n_hist_points):

                j_point = returned_pids_hist[j]
                j_idx = int(pid_hist_to_idx_map_dict[i_batch, j_point])
                i_idx = int(pid_hist_to_idx_map_dict[i_batch, i_point])

                if i_idx < 0:
                    continue
                _prob_val = point_class_probs[i_batch, i_idx, 1]
                if _prob_val > grouped_prob_all_points[i, 0]:
                    grouped_prob_all_points[i, 0] = _prob_val

                if j_idx < 0:
                    continue
                _corr_val = corr_mats[i_batch, i_idx, j_idx]
                grouped_corr_all_points[i, j] = _corr_val
                grouped_corr_all_points[j, i] = _corr_val

            for j in range(0, n_curr_points):
                j_point = returned_pids_curr[j]
                j_idx = int(pid_curr_to_idx_map_dict[i_batch, j_point])
                i_idx = int(pid_hist_to_idx_map_dict[i_batch, i_point])
                if j_idx < 0 or i_idx < 0:
                    continue
                _corr_val = corr_mats[i_batch, i_idx, j_idx]
                grouped_corr_all_points[i, j + n_hist_points] = _corr_val
                grouped_corr_all_points[j + n_hist_points, i] = _corr_val

    for i, i_point in enumerate(returned_pids_curr):

        for i_batch in range(num_batch):
            for j in range(i, n_curr_points):
                j_point = returned_pids_curr[j]
                j_idx = int(pid_curr_to_idx_map_dict[i_batch, j_point])
                i_idx = int(pid_curr_to_idx_map_dict[i_batch, i_point])

                if i_idx < 0:
                    continue
                _prob_val = point_class_probs[i_batch, i_idx, 1]
                if _prob_val > grouped_prob_all_points[i + n_hist_points, 0]:
                    grouped_prob_all_points[i + n_hist_points, 0] = _prob_val

                if j_idx < 0:
                    continue
                _corr_val = corr_mats[i_batch, i_idx, j_idx]
                grouped_corr_all_points[i + n_hist_points, j + n_hist_points] = _corr_val
                grouped_corr_all_points[j + n_hist_points, i + n_hist_points] = _corr_val


    return grouped_corr_all_points, grouped_prob_all_points, returned_pids_hist


@numba.njit(cache=True)
def group_corr_mats(
    corr_mats: np.ndarray,
    ids_in_historical_points: np.ndarray,
    ids_in_current_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        corr_mats: shape=(batch_size, sample_len, sample_len)
        ids_in_historical_points: shape=(batch_size, sample_len), minus value for
            non-historical point.
        ids_in_current_points: shape=(batch_size, sample_len), minus value for
            point not to be associated.

    Return:
        - grouped correlation matrix, shape=(n_hist_points, n_curr_points),
          where n_curr_points = count of unique valid point indices (i.e.
          non-negative indices) in :attr:`indices_in_current_points`, and
          n_hist_points = count of unique valid point indices (i.e.
          non-negative indices) in :attr:`indices_in_historical_points`;

        - indices of point in historical points (no negative values), shape=
          (n_hist_points,).

        - indices of point in points to be associated (no negative values),
          shape=(n_curr_points,).

    """
    returned_pids_hist = np.unique(ids_in_historical_points)
    returned_pids_hist = returned_pids_hist[returned_pids_hist >= 0]
    returned_pids_curr = np.unique(ids_in_current_points)
    returned_pids_curr = returned_pids_curr[returned_pids_curr >= 0]

    n_hist_points = returned_pids_hist.size
    n_curr_points = returned_pids_curr.size

    grouped_corr = np.zeros((n_hist_points, n_curr_points), dtype=np.float64)

    pid_hist_to_idx_map = dict(zip(returned_pids_hist, np.arange(n_hist_points)))
    # mapping from point ID to index in `returned_pids_hist`
    pid_curr_to_idx_map = dict(zip(returned_pids_curr, np.arange(n_curr_points)))
    # mapping from point ID to index in `returned_pids_curr`

    for i_batch in range(corr_mats.shape[0]):
        for i_point in range(corr_mats.shape[1]):
            i_pid_curr = ids_in_current_points[i_batch, i_point]
            i_pid_hist = ids_in_historical_points[i_batch, i_point]

            if i_pid_curr < 0 and i_pid_hist < 0:  # i_point refers to a padded point
                continue

            for j_point in range(i_point + 1, corr_mats.shape[2]):
                j_pid_curr = ids_in_current_points[i_batch, j_point]
                j_pid_hist = ids_in_historical_points[i_batch, j_point]

                _corr_val = corr_mats[i_batch, i_point, j_point]

                if i_pid_curr >= 0:
                    if j_pid_hist >= 0:
                        hist_pid = j_pid_hist
                        curr_pid = i_pid_curr
                    else:
                        continue
                else:  # identical to: `elif i_pid_hist >= 0:`
                    if j_pid_curr >= 0:
                        hist_pid = i_pid_hist
                        curr_pid = j_pid_curr
                    else:
                        continue

                grouped_corr[pid_hist_to_idx_map[hist_pid], pid_curr_to_idx_map[curr_pid]] = _corr_val

    argsort_pids_hist = np.argsort(returned_pids_hist)
    argsort_pids_curr = np.argsort(returned_pids_curr)
    grouped_corr = grouped_corr[argsort_pids_hist]
    grouped_corr = grouped_corr[:, argsort_pids_curr]

    return grouped_corr, returned_pids_hist[argsort_pids_hist], returned_pids_curr[argsort_pids_curr]
