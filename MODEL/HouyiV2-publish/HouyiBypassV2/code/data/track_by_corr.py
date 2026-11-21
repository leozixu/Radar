#  Copyright 2025 Junxiang Xiao
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

"""
This module provides several functions for tracking routes according to
correlation matrix between points.

"""

from typing import Tuple
import numba
import numpy as np

@numba.njit(cache=True)
def _call_scoring_reduce_fn(
    a: np.ndarray,
    reduce_fn_name: str
):
    if reduce_fn_name == "mean":
        return np.mean(a)
    if reduce_fn_name == "max":
        return np.max(a)
    if reduce_fn_name == "min":
        return np.min(a)

    raise ValueError(f"Currently {reduce_fn_name} is not supported.")

@numba.njit(cache=True, parallel=True)
def score_point_routes(
    corr: np.ndarray,
    prev_point_rids: np.ndarray,
    reduce_fn_name: str = "mean",
    score_decrement_for_1p_route: float = 0.
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate point-route scores.

    Args:
        corr: shape=(n_current_circ_points, n_lookback_points)
        prev_point_rids: shape=(n_lookback_points,)
        reduce_fn_name: Function to evaluate point-route score from multi-circle
            historical correlation values of each route. Available options are
            "mean", "max", "min".
        score_decrement_for_1p_route: For routes which only contain 1 point in
            previous circles, all its point-route scores will be decremented by
            this value.

    Returns:
        - scores, shape=(n_current_circ_points, n_lookback_routes)
        - route_id_set, shape=(n_lookback_routes,), each element corresponds
            to a row of `scores`

    Note:
        If :attr:`score_decrement_for_1p_route` is set, elements in returned
        :attr:`scores` may be negative rather than stay in range [0, 1].

    """
    route_id_set = np.sort(np.unique(prev_point_rids))
    scores = np.zeros((corr.shape[0], len(route_id_set)), dtype=np.float64)

    for i_point in numba.prange(corr.shape[0]):
        point_corrs = corr[i_point]
        for i_route in numba.prange(len(route_id_set)):
            route_id = route_id_set[i_route]

            route_prev_points_mask = prev_point_rids == route_id
            pr_corrs = point_corrs[route_prev_points_mask]
            scores[i_point, i_route] = _call_scoring_reduce_fn(pr_corrs, reduce_fn_name)

            if np.sum(route_prev_points_mask) == 1:
                scores[i_point, i_route] -= score_decrement_for_1p_route

    return scores, route_id_set
