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
Module for data processing.

Authors: Junxiang Xiao, Yuanli Feng.

"""

import abc

import numba
import numpy as np

from model_association.point_cc.config import DataConfig, NormalizationFactor
from model_association.point_cc.data import (
    data_preprocess, normalize_samples, split_area_samples,
    append_legacy_features, denormalize_samples, az_enc_to_rad
)
from model_association.route_point_data_processor import HorizontalCoordPaddedFanSplitter
from model_association.utils import dict_to_numba
from model_association.utils import setup_logger
from numpy import typing as npt

from model_association import radar_definition as rd


class DataProcessorBase(abc.ABC):

    @abc.abstractmethod
    def preprocess(self, original_data: npt.NDArray[float], *args, **kwargs):
        """
        Preprocess original raw data, including clipping out-of-bound values,
        filling features to be calculated like (x,y), converting quantities to
        SI units, etc.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_generate(self, data: npt.NDArray[float], *args, **kwargs):
        """
        Generate samples for model input from raw data (potentially preprocessed).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def point_cluster(
        self,
        points: np.ndarray[tuple[int, int], float],
        historical_points,
        *args, **kwargs
    ):
        """
        Perform clustering on points (potentially preprocessed). Hyper-parameters
        are supposed to be passed in as positional arguments and key-word arguments.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def middle_process(
        self,
        points: np.ndarray[float],
        *args, **kwargs
    ):
        """
        Post-process point data.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(
        self,
        points: np.ndarray[float],
        *args, **kwargs
    ):
        """
        Post-process point data.

        """
        raise NotImplementedError


class DataProcessor(DataProcessorBase):

    def __init__(
        self,
        n_point_features,
        circle_config,
        model_config,
        data_config: DataConfig,
        cluster_config,
        preprocess_config,
        log_level,
        log_file_path,
    ):
        # Hyper-parameters for preprocess assignment
        self.preprocess_config = preprocess_config

        # Hyper-parameters for circle assignment
        self.circle_time = circle_config['circle_time']
        self.circle_length = circle_config['circle_length']
        self.circle_num = circle_config['circle_num']

        # Hyper-parameters for point clustering
        self.cluster_time_thresh = cluster_config['time_thresh']  # s (RADAR_TIME in seconds)
        self.cluster_r_thresh = cluster_config['r_thresh']  # 原始距离单位(米/500e3)
        self.cluster_az_thresh = cluster_config['az_thresh']  # 原始方位单位(度/120)

        # Hyper-parameters for data preprocessing
        self.data_config = data_config

        # Hyper-parameters for sub-area splitting
        self.padded_point_splitter = HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            r_max=self.data_config.max_value.r,
            area_width=np.deg2rad(self.data_config.split_azim_width),
            pad_width=self.data_config.split_pad_width,
            r1=self.data_config.r1
        )
        self.nopad_point_splitter = HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            r_max=self.max_values.r,
            area_width=np.deg2rad(self.data_config.split_azim_width),
            pad_width=0,
            r1=0
        )

        # Hyper-parameters for sample generating
        self.sample_len = self.data_config.sample_len
        self.time_len = circle_config['time_len']
        self.n_point_features = n_point_features

        self.az_min = -np.pi
        self.az_range = 2 * np.pi

        # Hyper-parameters for day spanning
        self.last_time = 0

        self.logger = setup_logger(
            self.__class__.__name__,
            log_level=log_level,
            log_file_name=log_file_path
        )

    @property
    def max_values(self):
        return self.data_config.max_value

    def point_cluster(
        self,
        points: np.ndarray[tuple[int, int], float],
        historical_points,
        *args, **kwargs
    ) -> np.ndarray:
        return classical_point_cluster(
            points[points[:, rd.PointFeatureIndex.SNR.value].argsort()],
            self.cluster_r_thresh,
            self.cluster_az_thresh,
            self.cluster_time_thresh,
            historical_points
        )

    def circle_process(self, original_data: npt.NDArray[float], *args, **kwargs) -> np.ndarray:
        if len(original_data) > 0:
            if self.circle_time == 0:
                self.circle_time = np.min(original_data[:, rd.PointFeatureIndex.RADAR_TIME.value])
            while True:
                idx = ((original_data[:, rd.PointFeatureIndex.RADAR_TIME.value] >= self.circle_time) &
                       (original_data[
                            :, rd.PointFeatureIndex.RADAR_TIME.value] < self.circle_time + self.circle_length))
                original_data[idx, rd.PointFeatureIndex.CIRCLE_ID] = self.circle_num
                if self.circle_time + self.circle_length < np.max(
                    original_data[:, rd.PointFeatureIndex.RADAR_TIME.value]):
                    self.circle_time = self.circle_time + self.circle_length
                    self.circle_num = self.circle_num + 1
                else:
                    break
        return original_data

    def preprocess(self, original_data: npt.NDArray[float], *args, **kwargs) -> np.ndarray:
        value_bounds_dict = {
            rd.PointFeatureIndex.R.value: (0, self.max_values.r),
            rd.PointFeatureIndex.AMP.value: (-np.inf, self.max_values.amp),
            rd.PointFeatureIndex.SNR.value: (-np.inf, self.max_values.snr),
            rd.PointFeatureIndex.HEIGHT.value: (0, self.max_values.height),  # TODO: min value?
            rd.PointFeatureIndex.EL.value: (np.deg2rad(self.max_values.elev.min),
                                            np.deg2rad(self.max_values.elev.max)),
            rd.PointFeatureIndex.RADIAL_SPEED: (-self.max_values.speed, self.max_values.speed)
        }
        out_of_bounds_filters_dict = {
            rd.PointFeatureIndex.R.value: self.preprocess_config['distance_filter'],
            rd.PointFeatureIndex.EL.value: self.preprocess_config['elevation_filter']
        }
        if self.preprocess_config['channel_filter'] >= 0:
            value_bounds_dict[rd.PointFeatureIndex.CHANNEL.value] = (
                self.preprocess_config['channel_filter'] - 0.1,
                self.preprocess_config['channel_filter'] + 0.1
            )
            out_of_bounds_filters_dict[rd.PointFeatureIndex.CHANNEL.value] = True
        value_bounds = dict_to_numba(
            value_bounds_dict,
            key_type=numba.int64,
            value_type=numba.types.UniTuple(numba.float64, 2)
        )
        out_of_bounds_filters = dict_to_numba(
            out_of_bounds_filters_dict,
            key_type=numba.int64,
            value_type=numba.bool_
        )
        features_to_si_coeffs = dict_to_numba(
            rd.POINT_FEATURES_TO_SI_UNIT_COEFF_MAPPINGS,
            key_type=numba.int64,
            value_type=numba.float64
        )

        data = data_preprocess(
            original_data,
            value_bounds=value_bounds,
            out_of_bounds_filters=out_of_bounds_filters,
            to_si_unit_coeff_mapping=features_to_si_coeffs,
            relabel_track_id_manual=False,
            nan_filter=self.preprocess_config['nan_filter'],
            inf_filter=True,
            only_certain_points_filter=self.preprocess_config['manual_label_filter'],
            only_primary_and_fused_points=self.preprocess_config['only_primary_and_fused_points'],
            calculate_xy=True,
            az_unit_to_rad=2.0 * np.pi / self.az_range
        )[0]
        if len(data) == 0:
            return data
        else:
            data = self.circle_process(data)
            curr_area_mask = self.nopad_point_splitter.split(
                data[:, rd.PointFeatureIndex.R.value],
                az_enc_to_rad(
                    data[:, rd.PointFeatureIndex.AZ.value],
                    az_range=self.az_range
                )
            )
            data_preprocessed = np.zeros((data.shape[0], self.n_point_features + curr_area_mask.shape[1]))
            min_cols = min(data.shape[1], self.n_point_features)
            data_preprocessed[:, :min_cols] = data[:, :min_cols]
            data_preprocessed[:, -curr_area_mask.shape[1]:] = curr_area_mask
            return data_preprocessed


    def sample_generate(self, data: npt.NDArray[float], input_feature_indices, *args, **kwargs) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        if data[:, 1].max() - data[:, 1].min() > 100:
            print(1)
        intime_indices = \
            np.where(
                data[:, rd.PointFeatureIndex.RADAR_TIME.value]
                >= data[:, rd.PointFeatureIndex.RADAR_TIME.value].max() - self.time_len
            )[0]
        data = data[intime_indices, :]
        focus = data[:, rd.PointFeatureIndex.PACK_ID] == data[:, rd.PointFeatureIndex.PACK_ID].max()
        samples, area_ids, point_indices, _ = split_area_samples(
            data, self.padded_point_splitter.n_split_areas, self.sample_len
        )
        if focus is not None:
            focusing = focus[point_indices.astype(int)]
            focusing[point_indices < 0] = False
            area_focus = np.max(focusing, axis=1)
            area_focus_inds = np.where(area_focus)[0]
            samples = samples[area_focus_inds, :]
            area_ids = area_ids[area_focus_inds]
            point_indices = point_indices[area_focus_inds, :]
        samples_extended = np.concatenate(
            (samples, np.zeros((*samples.shape[:2], 6))), axis=-1
        )
        use_legacy_features = any(map(lambda i: i.is_legacy_feature, input_feature_indices))
        _splitter = self.padded_point_splitter._circumrect_splitter
        area_borders = np.zeros((len(area_ids), 4), dtype=float)
        area_inverse_transforms = np.zeros((len(area_ids), 2, 2), dtype=float)
        for i, area_id in enumerate(area_ids):
            area_borders[i] = np.array(
                [-_splitter.rect_widths[area_id] / 2,
                 _splitter.rect_widths[area_id] / 2,
                 -_splitter.rect_depths[area_id],
                 _splitter.rect_lengths[area_id] - _splitter.rect_depths[area_id]]
            )
            area_inverse_transforms[i] = _splitter.areas_inverse_transforms[area_id]

        normalization_output = normalize_samples(
            samples_extended,
            area_borders,
            area_inverse_transforms,
            time_normalization_factor=self.data_config.normalization_factor.time,
            r_normalization_factor=self.data_config.normalization_factor.r,
            amp_normalization_factor=self.data_config.normalization_factor.amp,
            snr_normalization_factor=self.data_config.normalization_factor.snr,
            h_normalization_factor=self.data_config.normalization_factor.height,
            elev_normalization_factor=np.deg2rad(self.data_config.normalization_factor.elev.range),
            elev_min=np.deg2rad(self.data_config.normalization_factor.elev.min),
            az_min=self.az_min,
            az_range=self.az_range,
            half_speed_normalization_factor=self.data_config.normalization_factor.speed_half,
            return_legacy_features=use_legacy_features
        )
        samples_normalized = normalization_output[0]
        if not use_legacy_features:
            input_data = samples_normalized[:, :, input_feature_indices]
        else:
            legacy_features = normalization_output[2]
            input_data = append_legacy_features(samples_normalized,
                                                legacy_features,
                                                input_feature_indices)

        intime_indices = np.concatenate([intime_indices, [-1]])
        return input_data, intime_indices[point_indices.astype(np.int64)], area_ids

    def middle_process(
        self,
        points: np.ndarray[float],
        *args, **kwargs
    ):
        points[:, -self.padded_point_splitter.n_split_areas:] = self.padded_point_splitter.split(
            points_r=points[:, rd.PointFeatureIndex.R],
            points_azim=az_enc_to_rad(
                points[:, rd.PointFeatureIndex.AZ],
                az_range=self.az_range
            )
        )
        return points

    def post_process(
        self,
        points: np.ndarray[float],
        *args, **kwargs
    ):
        # 数值反归一化: 将模型输出的归一化值恢复为显控/上报协议单位(度/米/dB 等)。
        # 归一化因子来自 config (normalization_factor), 与 normalize_samples 对偶。
        rad2deg = 180.0 / np.pi
        az_deg = (points[:, rd.PointFeatureIndex.AZ] * rad2deg) % 360.0
        el_deg = ((points[:, rd.PointFeatureIndex.EL] * rad2deg + 180.0) % 360.0) - 180.0
        points = denormalize_samples(
            points,
            az_range=self.az_range,
            elev_range=self.data_config.normalization_factor.elev.range,
            r_nf=self.data_config.normalization_factor.r,
            amp_nf=self.data_config.normalization_factor.amp,
            snr_nf=self.data_config.normalization_factor.snr,
            h_nf=self.data_config.normalization_factor.height,
            half_speed_nf=self.data_config.normalization_factor.speed_half,
        )
        points[:, rd.PointFeatureIndex.AZ] = az_deg
        points[:, rd.PointFeatureIndex.EL] = el_deg
        return points


@numba.njit(cache=True)
def classical_point_cluster(
    points: np.ndarray,
    r_thresh: float,
    azim_thresh: float,
    time_thresh: float,
    historical_points: np.ndarray
) -> np.ndarray:
    """
    Args:
        points:
        r_thresh:
        azim_thresh:
        time_thresh:
        historical_points: Historical points of shape (n_hist_points, n_point_features),
            which MUST be ordered by time in ascending order.

    Return:
        Filtered points.

    """
    n_points = points.shape[0]

    # 方位(原始单位, 量程 3)环绕差: 折叠到 [-1.5, 1.5) 后作普通差
    az_wrapped = points[:, rd.PointFeatureIndex.AZ.value] \
        - 3.0 * np.floor((points[:, rd.PointFeatureIndex.AZ.value] + 1.5) / 3.0)
    hist_az_wrapped = historical_points[:, rd.PointFeatureIndex.AZ.value] \
        - 3.0 * np.floor((historical_points[:, rd.PointFeatureIndex.AZ.value] + 1.5) / 3.0)

    kept_index = []
    for i_point in range(n_points):
        p_i = points[i_point]

        _to_kept = True
        for j_point in range(i_point + 1, n_points):
            p_j = points[j_point]

            if (
                np.abs(p_i[rd.PointFeatureIndex.R.value] - p_j[rd.PointFeatureIndex.R.value]) <= r_thresh
                and
                np.abs(az_wrapped[i_point] - az_wrapped[j_point]) <= azim_thresh
                and
                np.abs(p_i[rd.PointFeatureIndex.RADAR_TIME.value] - p_j[
                    rd.PointFeatureIndex.RADAR_TIME.value]) <= time_thresh
                and
                p_i[rd.PointFeatureIndex.SNR.value] <= p_j[rd.PointFeatureIndex.SNR.value]
            ):
                _to_kept = False
                break

        if _to_kept:
            for j_point_history in range(len(historical_points) - 1, -1, -1):
                p_j = historical_points[j_point_history]

                if np.abs(p_i[rd.PointFeatureIndex.RADAR_TIME.value] - p_j[
                    rd.PointFeatureIndex.RADAR_TIME.value]) > time_thresh:
                    # Previous historical points can all be skipped since
                    # they are sorted by time in ascending order
                    break

                if (
                    np.abs(p_i[rd.PointFeatureIndex.R.value] - p_j[rd.PointFeatureIndex.R.value]) <= r_thresh
                    and
                    np.abs(az_wrapped[i_point] - hist_az_wrapped[j_point_history]) <= azim_thresh
                ):
                    _to_kept = False
                    break

        if _to_kept:
            kept_index.append(i_point)
    points_out = points[np.array(kept_index)]
    return points_out[points_out[:, rd.PointFeatureIndex.RADAR_TIME.value].argsort()]
