#  Copyright 2025-2026 Junxiang Xiao
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
Definitions of radar types and data fields.

"""

import enum
import numpy as np

_START_IDX_OF_LEGACY_FEAT = 10000


class PointFeatureIndex(enum.IntEnum):
    RADAR_TIME = 0  # ms
    AZ = 1  # 度（0-360）
    EL = 2  # 度
    R = 3  # 米
    AMP = 4  # 幅度（20*log）
    SNR = 5  # 信噪比（20*log）
    CIRCLE_ID = 6  # 圈号
    POINT_ID = 7  # 点序号
    TRACK_ID_MANUAL = 8  # 手工航迹标签（批号）
    TRACK_ID_ALGO = 9  # 算法航迹标签（批号）
    TAS_FLAG = 11
    DATA_RATE = 16
    PLOT_TYPE = 22
    PROCESS_CODE = 23
    X = 24  # X（处理按需生成）
    Y = 25  # Y（处理按需生成）
    MODEL_CLASS = 26  # 模型判断是否为目标0/1
    MODEL_PROB = 27  # 模型判断目标概率（0-1）
    MODEL_CORR = 28  # 模型输出特征系数（相似性关联度）
    TRACK_ID_INTELLI_FUSED = 30  # 灵活融合标签
    CONFIRM_LABEL = 31
    CHANNEL = 32
    PACK_ID = 33  # ID of incoming data pack
    Z = 34  # Y（处理按需生成）
    R_FILT = 35
    AZ_FILT = 36
    EL_FILT = 37
    X_FILT = 38
    Y_FILT = 39
    LOST_FLAG = 40
    HEIGHT = 41  # 米
    RADIAL_SPEED = 42
    LAT = 47
    LON = 48
    ROUTE_NEXT_POINT_R = 49  # R of the route's next point (for models that regress R)
    ROUTE_NEXT_POINT_T = 53
    ROUTE_NEXT_POINT_X = 54
    ROUTE_NEXT_POINT_Y = 55
    FOCUSING_POINT_FLAG = 52
    ROUTE_LOC1_POINT_IDX = 56
    ROUTE_LOC2_POINT_IDX = 57
    ROUTE_LOC3_POINT_IDX = 58
    ROUTE_START_POINT_IDX = 59
    ROUTE_END_POINT_IDX = 60
    RADAR_TYPE = 63

    PROCESS_ID = 72
    TARGET_FIND_PROCESS_ID = 73
    ROUTE_CORR = 74
    ROUTE_SHOW_FLAG = 75
    First_report = 76
    REPORT_PROCESS_ID = 77
    REPORT_TIME = 78

    X_NORM_DEC1_SCALED_SIN = _START_IDX_OF_LEGACY_FEAT
    X_NORM_DEC1_SCALED_COS = _START_IDX_OF_LEGACY_FEAT + 1
    Y_NORM_DEC1_SCALED_SIN = _START_IDX_OF_LEGACY_FEAT + 2
    Y_NORM_DEC1_SCALED_COS = _START_IDX_OF_LEGACY_FEAT + 3
    R_NORM_DEC1_SCALED_SIN = _START_IDX_OF_LEGACY_FEAT + 4
    R_NORM_DEC1_SCALED_COS = _START_IDX_OF_LEGACY_FEAT + 5

    @property
    def is_legacy_feature(self):
        return self.value >= _START_IDX_OF_LEGACY_FEAT

    @classmethod
    @property
    def required_columns(cls) -> int:
        return max(p for p in PointFeatureIndex if not p.is_legacy_feature).value + 1


# replica with azimuth correctly recovered
POINT_FEATURES_TO_SI_UNIT_COEFF_MAPPINGS = {
    PointFeatureIndex.RADAR_TIME: 1e-3,
    PointFeatureIndex.AZ: np.pi / 1.5,
    PointFeatureIndex.EL: 1
}