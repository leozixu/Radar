#  Copyright 2025-2026 Junxiang Xiao
#
#  SPDX-License-Identifier: Apache-2.0
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


import warnings
from typing import Tuple, Optional

import numpy as np
import torch


def eval_dynamic_vals(
    ts: torch.Tensor,
    xys: torch.Tensor
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor]
]:
    """
    Evaluate dynamic values for routes.

    Args:
        ts: Times of points, shape=(n_routes, n_points_per_route). For each
            route, times must be organized in ascending order (for example,
            tailing padded NaNs are OK; but tailing padded zeros are not),
            otherwise some dynamic quantities may take unexpected values.
        xys: Coordinates of points, shape=(n_routes, n_points_per_route, 2),
            with each point represented by (x, y) coordinate.

    Returns:
        A Tuple of:
        - Moving distances between adjacent points, shape=``(n_routes,
          n_points_per_route-1)`` if ``n_points_per_route >= 1``, or
          ``None`` otherwise.
        - Average speeds between adjacent points, shape=``(n_routes,
          n_points_per_route-1)`` if ``n_points_per_route >= 1``, or
          ``None`` otherwise.
        - Velocity directions, shape=``(n_routes, n_points_per_route-1)`` if
          ``n_points_per_route >= 1``, or ``None`` otherwise.
        - The magnitude of acceleration between two adjacent intervals,
          shape=``(n_routes, n_points_per_route-2)`` if ``n_points_per_route
          >= 2``, or ``None`` otherwise.
        - The change rate of velocity directions between two adjacent intervals,
          shape=``(n_routes, n_points_per_route-2)`` if ``n_points_per_route
          >= 2``, or ``None`` otherwise.

    """
    n_routes, n_points_per_route, _ = xys.shape

    if n_points_per_route <= 0:
        return None, None, None, None, None

    dts = ts[:, 1:] - ts[:, :-1]  # shape=(n_routes, n_points_per_route-1)
    if torch.any(dts < 0):
        warnings.warn(
            "Input time sequences are not be organized in ascending order! "
            "This function may give out unexpected results."
        )

    # Detect distance anomaly
    dxys = xys[:, 1:] - xys[:, :-1]
    # shape=(n_routes, n_points_per_route-1, 2)
    dists = torch.norm(dxys, dim=-1)
    # shape=(n_routes, n_points_per_route-1)

    velocities = torch.zeros_like(dxys, dtype=dxys.dtype, device=dxys.device)
    nonzero_dt_mask = dts != 0
    velocities[nonzero_dt_mask] = dxys[nonzero_dt_mask] / dts[nonzero_dt_mask].unsqueeze(1)
    speeds = torch.norm(velocities, dim=-1)
    # shape=(n_routes, n_points_per_route-1)
    v_direc = torch.atan2(velocities[:, :, 1], velocities[:, :, 0])
    # shape=(n_routes, n_points_per_route-1)

    # Detect speed anomaly
    if n_points_per_route > 2:
        # Common values for calculating 2nd order differences
        nonzero_dt_mask_2nd_ord = nonzero_dt_mask[:, 1:]
        nonzero_dts_2nd_ord = dts[:, 1:][nonzero_dt_mask_2nd_ord]

        # Acceleration magnitude:
        vel_diffs = torch.abs(velocities[:, 1:] - velocities[:, :-1])
        # shape=(n_routes, n_points_per_route-2, 2)
        vel_diff_norms = torch.norm(vel_diffs, dim=-1)
        # shape=(n_routes, n_points_per_route-2)
        accelerations = torch.zeros_like(
            vel_diff_norms,
            dtype=vel_diff_norms.dtype,
            device=vel_diff_norms.device
        )
        accelerations[nonzero_dt_mask_2nd_ord] = \
            vel_diff_norms[nonzero_dt_mask_2nd_ord] / nonzero_dts_2nd_ord

        # Change rate of velocity direction:
        # shape=(n_routes, n_points_per_route-1), value range: [-pi, pi]
        v_direc_diff = v_direc[:, 1:] - v_direc[:, :-1]
        # shape=(n_routes, n_points_per_route-2), value range: [-2*pi, 2*pi]
        v_direc_diff = torch.remainder(v_direc_diff + np.pi, 2 * np.pi) - np.pi
        # value range: [-pi, pi]

        v_direc_diff_rate = torch.zeros_like(
            v_direc_diff,
            dtype=v_direc_diff.dtype,
            device=v_direc_diff.device
        )
        v_direc_diff_rate[nonzero_dt_mask_2nd_ord] = \
            v_direc_diff[nonzero_dt_mask_2nd_ord] / nonzero_dts_2nd_ord

    else:
        accelerations = None
        v_direc_diff_rate = None

    return dists, speeds, v_direc, accelerations, v_direc_diff_rate


def detect_anomalies(
    ts: torch.Tensor,
    xys: torch.Tensor,
    speed_thresh: float,
    acceleration_thresh: float,
    direc_change_thresh: float
) -> torch.Tensor:
    """
    Detect anomalies in routes according to dynamics properties.

    Args:
        ts: Times of points, shape=(n_routes, n_points_per_route).
        xys: Coordinates of points, shape=(n_routes, n_points_per_route, 2),
            with each point represented by (x, y) coordinate.
        speed_thresh: When the speed in one interval (between two adjacent points)
            exceeds this value, an anomaly is detected.
        acceleration_thresh: When the magnitude of acceleration between two
            adjacent intervals exceeds this value, an anomaly is detected.
        direc_change_thresh: When the change rate of velocity directions between
            two adjacent intervals exceeds this value, an anomaly is detected.

    Return:
        A boolean mask specifying whether a point is an anomaly, shape=
        (n_routes, n_points_per_route).

    """
    n_routes, n_points_per_route, _ = xys.shape
    is_anomalous = torch.zeros(
        [n_routes, n_points_per_route],
        dtype=torch.bool,
        device=xys.device
    )

    dists, speeds, v_direc, accelerations, v_direc_diff_rate = \
        eval_dynamic_vals(ts, xys)

    if n_points_per_route > 1:
        # Detect distance anomaly
        is_anomalous[:, 1:] |= speeds > speed_thresh

        # Detect speed anomaly
        if n_points_per_route > 2:
            # Acceleration magnitude:
            is_anomalous[:, 2:] |= accelerations > acceleration_thresh

            # Change rate of velocity direction:
            is_anomalous[:, 2:] |= torch.abs(v_direc_diff_rate) > direc_change_thresh

    return is_anomalous
