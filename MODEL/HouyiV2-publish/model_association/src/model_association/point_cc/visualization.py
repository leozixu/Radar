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

from typing import Optional, List, Tuple

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from model_association.point_cc.data import FeatureColumnIndices
from model_association.utils import normalize
from model_association.visualization import plot_metric_curve

CMAPS = tuple(
    cm.get_cmap(name) for name in
    ["Oranges", "Blues", "Greens", "Purples",
     "spring", "summer", "autumn", "winter", "ocean"]
)


def visualize_route_points(
    points_data: np.ndarray,
    feature_indices: FeatureColumnIndices,
    pred_route_points: Optional[List[List[int]]],
    x_normalization_factor: float = 1.,
    y_normalization_factor: float = 1.,
    x_shift: float = 0,
    y_shift: float = 0,
    title: str = ""
) -> Tuple[Figure, List[int]]:
    """
    Plot route-points data.

    Args:
        points_data: Data of points, whose (x, y) coordinates must be normalized,
            shape=(n_points, n_features)
        feature_indices: Feature column indices.
        pred_route_points: List of point indices of each tracked route.
        x_normalization_factor: In plotted figures, all x coordinates will
            be multiplied with this value.
        y_normalization_factor: In plotted figures, all y coordinates will
            be multiplied with this value.
        x_shift: In plotted figures, all x coordinates will be shifted with
            this value.
        y_shift: In plotted figures, all y coordinates will be shifted with
            this value.
        title: Figure title.

    Returns:
        Plotted figure and indices of plotted routes.

    """
    fig: Figure = plt.figure()  # (figsize=(8, 5), dpi=200)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.93)
    gs = GridSpec(1, 1,  # 2, 2,
                  fig,
                  # width_ratios=[1, 1], height_ratios=[1, 1],
                  wspace=0.3, hspace=0.3)
    axes_xy: Axes = fig.add_subplot(gs[0, 0])
    # axes_ra: Axes = fig.add_subplot(gs[0, 1])
    # axes_tr: Axes = fig.add_subplot(gs[1, 0])
    # axes_ta: Axes = fig.add_subplot(gs[1, 1])

    plotted_route_ids = []

    true_route_points = []
    true_route_ids = points_data[:, feature_indices.route_id]
    true_route_id_set = np.sort(np.unique(true_route_ids))
    for route_id in true_route_id_set:
        true_route_points.append(np.where(true_route_ids == route_id)[0])

    for i_route, (_true_rid, _true_route_pids) in enumerate(
        zip(true_route_id_set, true_route_points)
    ):
        _route = points_data[_true_route_pids]
        route_ts = _route[:, feature_indices.time]  # normalized
        # route_rs = _route[:, feature_indices.r_noisy]
        # route_azims = _route[:, feature_indices.azim_noisy]  # degrees
        route_xs = (_route[:, feature_indices.x_noisy] * x_normalization_factor + x_shift)
        route_ys = (_route[:, feature_indices.y_noisy] * y_normalization_factor + y_shift)

        route_not_nan_xs = route_xs[~np.isnan(route_xs)]
        route_not_nan_ys = route_ys[~np.isnan(route_ys)]

        _cmap = CMAPS[i_route % len(CMAPS)]

        for (
            _axes,
            _route_hori_feature,
            _route_vert_feature
        ) in (
            [axes_xy, route_xs, route_ys],
            # [axes_ra, route_rs, route_azims],
            # [axes_tr, route_ts, route_rs],
            # [axes_ta, route_ts, route_azims]
        ):
            # plot route line
            if _true_rid > 0:
                route_plot = _axes.plot(
                    _route_hori_feature, _route_vert_feature,
                    alpha=0.3,
                    label=f"route {_true_rid}",
                    color=_cmap(0.5),
                    markersize=3,
                    linewidth=2
                )[0]
            # plot all points in the route
            _axes.scatter(
                _route_hori_feature,
                _route_vert_feature,
                c=_cmap((route_ts - route_ts[0])
                        / (route_ts[-1] - route_ts[0]) * 0.6 + 0.4),
                marker="o",
                s=8,
                alpha=0.6
            )

        for _axes, _hori_label, _vert_label in (
            [axes_xy, '$x$ / m', '$y$ / m'],
            # [axes_ra, '$r$ / km', r'Azimuth / $\degree$'],
            # [axes_tr, '$t$ / (normalized unit)', '$r$ / km'],
            # [axes_ta, '$t$ /  (normalized unit)', r'Azimuth / $\degree$']
        ):
            _axes.set_xlabel(_hori_label)
            _axes.set_ylabel(_vert_label)
            # _axes.legend()

        plotted_route_ids.append(_true_rid)

    for _pred_route_pids in pred_route_points:
        _route = points_data[_pred_route_pids]
        route_ts = _route[:, feature_indices.time]  # normalized
        route_rs = _route[:, feature_indices.r_noisy]
        route_azims = _route[:, feature_indices.azim_noisy]  # degrees
        route_xs = (_route[:, feature_indices.x_noisy] * x_normalization_factor + x_shift)
        route_ys = (_route[:, feature_indices.y_noisy] * y_normalization_factor + y_shift)

        x_max = x_shift + x_normalization_factor
        x_min = x_shift
        y_max = y_shift + y_normalization_factor
        y_min = y_shift

        for (
            _axes,
            _route_hori_feature,
            _route_vert_feature
        ) in (
            [axes_xy, route_xs, route_ys],
            # [axes_ra, route_rs, route_azims],
            # [axes_tr, route_ts, route_rs],
            # [axes_ta, route_ts, route_azims]
        ):
            # plot tracked route line
            route_plot = _axes.plot(
                _route_hori_feature, _route_vert_feature,
                alpha=0.5,
                label=f"route {_true_rid}",
                linestyle="--"
            )[0]

    if len(plotted_route_ids):
        # In this case all limits are not NaN or Inf
        x_mid = (x_max + x_min) / 2
        x_range = (x_max - x_min) * 1.2
        y_mid = (y_max + y_min) / 2
        y_range = (y_max - y_min) * 1.2
        d_range = max(x_range, y_range)

        axes_xy.set_xlim(x_mid - d_range / 2, x_mid + d_range / 2)
        axes_xy.set_ylim(y_mid - d_range / 2, y_mid + d_range / 2)
        axes_xy.set_aspect(1)

    if title:
        fig.suptitle(title)

    return fig, plotted_route_ids


def visualize_route_points_xys(
    points_data: np.ndarray,
    feature_indices: FeatureColumnIndices,
    pred_route_points: Optional[List[List[int]]],
    x_normalization_factor: float = 1.,
    y_normalization_factor: float = 1.,
    x_shift: float = 0,
    y_shift: float = 0,
    title: str = "",
    axes_xy: Optional[Axes] = None
) -> List[int]:
    """
    Plot route-points data.

    Args:
        points_data: Data of points, whose (t,x,y) coordinates can be normalized
            or not, shape=(n_points, n_features)
        feature_indices: Feature column indices.
        pred_route_points: List of point indices of each tracked route.
        x_normalization_factor: In plotted figures, all x coordinates will
            be multiplied with this value.
        y_normalization_factor: In plotted figures, all y coordinates will
            be multiplied with this value.
        x_shift: In plotted figures, all x coordinates will be shifted with
            this value.
        y_shift: In plotted figures, all y coordinates will be shifted with
            this value.
        title: Figure title.
        axes_xy: Axes to plot on. If being None, axes will be created inside
            this function.

    Returns:
        Plotted figure and indices of plotted routes.

    """
    if axes_xy is None:
        fig: Figure
        fig, axes_xy = plt.subplots()

    # Filter out NaN points
    valid_points_mask = ~np.any(
        np.isnan(
            points_data[:, [feature_indices.time,
                            feature_indices.x_noisy,
                            feature_indices.y_noisy,
                            feature_indices.route_id]]
        ), axis=-1
    )
    points_data = points_data[valid_points_mask]

    plotted_route_ids = []

    true_route_points = []
    true_route_ids = points_data[:, feature_indices.route_id]
    true_route_id_set = np.sort(np.unique(true_route_ids))
    for route_id in true_route_id_set:
        if np.isnan(route_id):
            continue
        true_route_points.append(np.where(true_route_ids == route_id)[0])

    # De-normalize (x,y)
    points_data[:, feature_indices.x_noisy] = \
        points_data[:, feature_indices.x_noisy] * x_normalization_factor + x_shift
    points_data[:, feature_indices.y_noisy] = \
        points_data[:, feature_indices.y_noisy] * y_normalization_factor + y_shift
    x_min = np.min(points_data[:, feature_indices.x_noisy])
    x_max = np.max(points_data[:, feature_indices.x_noisy])
    y_min = np.min(points_data[:, feature_indices.y_noisy])
    y_max = np.max(points_data[:, feature_indices.y_noisy])

    for i_route, (_true_rid, _true_route_pids) in enumerate(
        zip(true_route_id_set, true_route_points)
    ):
        _route = points_data[_true_route_pids]
        route_ts = _route[:, feature_indices.time]
        route_ts = normalize(route_ts)  # normalized
        route_xs = _route[:, feature_indices.x_noisy]
        route_ys = _route[:, feature_indices.y_noisy]

        _cmap = CMAPS[i_route % len(CMAPS)]

        for (
            _axes,
            _route_hori_feature,
            _route_vert_feature
        ) in (
            [axes_xy, route_xs, route_ys],
        ):
            # plot route line
            if _true_rid > 0:
                _axes.plot(
                    _route_hori_feature, _route_vert_feature,
                    alpha=0.3,
                    label=f"route {_true_rid}",
                    color=_cmap(0.5),
                    markersize=3,
                    linewidth=2
                )
            # plot all points in the route
            _axes.scatter(
                _route_hori_feature,
                _route_vert_feature,
                c=_cmap((route_ts - route_ts[0])
                        / (route_ts[-1] - route_ts[0]) * 0.6 + 0.4),
                marker="o",
                s=8,
                alpha=0.6
            )

        for _axes, _hori_label, _vert_label in (
            [axes_xy, '$x$ / m', '$y$ / m'],
        ):
            _axes.set_xlabel(_hori_label)
            _axes.set_ylabel(_vert_label)
            # _axes.legend()

        plotted_route_ids.append(_true_rid)

    for _pred_route_pids in pred_route_points:
        _route = points_data[_pred_route_pids]
        route_xs = _route[:, feature_indices.x_noisy]
        route_ys = _route[:, feature_indices.y_noisy]

        for (
            _axes,
            _route_hori_feature,
            _route_vert_feature
        ) in (
            [axes_xy, route_xs, route_ys],
        ):
            # plot tracked route line
            _axes.plot(
                _route_hori_feature, _route_vert_feature,
                alpha=0.6,
                label=f"route {_true_rid}",
                linestyle="--"
            )

    if len(plotted_route_ids):
        # In this case all limits are not NaN or Inf
        x_mid = (x_max + x_min) / 2
        x_range = (x_max - x_min) * 1.2
        y_mid = (y_max + y_min) / 2
        y_range = (y_max - y_min) * 1.2
        d_range = max(x_range, y_range)

        axes_xy.set_xlim(x_mid - d_range / 2, x_mid + d_range / 2)
        axes_xy.set_ylim(y_mid - d_range / 2, y_mid + d_range / 2)
        axes_xy.set_aspect(1)

    axes_xy.set_title(title)
    return plotted_route_ids
