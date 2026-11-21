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

""" Parameters for processing data from different radars. """

from __future__ import annotations

from typing import Dict, Literal, Union

import numba
import numpy as np

from . import radar_definition as rd

# @numba.njit(cache=True)
def data_preprocess(
    data,
    *,
    to_si_unit_coeff_mapping: numba.typed.Dict = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.float64
    ),
    relabel_track_id_manual: bool = False,
    nan_filter: bool = True,
    inf_filter: bool = True,
    only_certain_points_filter: bool = False,
    only_primary_and_fused_points: bool = False,
    value_bounds: numba.typed.Dict = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.types.UniTuple(numba.float64, 2)
    ),
    out_of_bounds_filters: numba.typed.Dict = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.bool_
    ),
    calculate_xy: bool = True,
    az_unit_to_rad: float = 2.0 * np.pi / 3.0
):
    kept_flags = np.ones(data.shape[0], dtype=np.bool_)

    n_points_filtered: Dict[
        Literal["nan"] | Literal["inf"] | Literal["uncertain"] | Literal["not_primary_and_fused_point"],
        int
    ] = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type,
        value_type=numba.int64
    )
    if nan_filter:
        n_points_filtered["nan"] = 0
    if inf_filter:
        n_points_filtered["inf"] = 0
    if only_certain_points_filter:
        n_points_filtered["uncertain"] = 0
    if only_primary_and_fused_points:
        n_points_filtered["not_primary_and_fused_point"] = 0

    n_points_out_of_bounds = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.int64
    )
    for k in out_of_bounds_filters.keys():
        # # Comment following lines to allow field to be listed in
        # # `out_of_bounds_filters` but not actually unbounded:
        # if k not in value_bounds:
        #     raise ValueError(
        #         f"Key {k} in `out_of_bounds_filters` is not found in "
        #         f"`value_bounds`"
        #     )
        n_points_out_of_bounds[k] = 0

    for i_point in numba.prange(len(data)):
        # filter out points whose features contain NaN
        if np.any(np.isnan(data[i_point])):
            if nan_filter:
                kept_flags[i_point] = False
                n_points_filtered["nan"] += 1
                continue
            else:
                data[i_point, np.isnan(data[i_point, :])] = 0

        # filter out points whose features contain inf
        if inf_filter and np.any(np.isinf(data[i_point])):
            kept_flags[i_point] = False
            n_points_filtered["inf"] += 1
            continue

        if data[i_point, rd.PointFeatureIndex.TRACK_ID_MANUAL.value] < 0:
            kept_flags[i_point] = False
            n_points_filtered["uncertain"] += 1
            continue

        # convert quantities to SI units
        for field, coeff in (
            to_si_unit_coeff_mapping.items()
        ):
            data[i_point, field] *= coeff

        # filter out points whose feature is out of bound, or clip the feature value
        for idx, (v_min, v_max) in value_bounds.items():
            v = data[i_point, idx]
            if v_min <= v <= v_max:
                continue

            _filter = out_of_bounds_filters.get(idx, False)

            if v < v_min:
                if _filter:
                    kept_flags[i_point] = False
                    n_points_out_of_bounds[idx] += 1
                    break
                else:
                    data[i_point, idx] = v_min

            if v_max < v:
                if _filter:
                    kept_flags[i_point] = False
                    n_points_out_of_bounds[idx] += 1
                    break
                else:
                    data[i_point, idx] = v_max

        # For filtered points, skip remaining procedure
        if not kept_flags[i_point]:
            continue

        # calculate (x,y) coordinates
        if calculate_xy:
            r = data[i_point, rd.PointFeatureIndex.R.value]
            azim = data[i_point, rd.PointFeatureIndex.AZ.value] * az_unit_to_rad
            data[i_point, rd.PointFeatureIndex.X.value] = r * np.sin(azim)
            data[i_point, rd.PointFeatureIndex.Y.value] = r * np.cos(azim)

    return data[kept_flags], n_points_filtered, n_points_out_of_bounds
