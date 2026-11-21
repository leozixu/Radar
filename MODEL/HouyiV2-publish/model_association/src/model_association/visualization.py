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


from pathlib import Path
from typing import Optional, List, Tuple, Sequence, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from model_association.route_point_data_processor import FeaturesColumnIndices


def visualize_route_points(
    route_data: np.ndarray,
    nearest_points_data: np.ndarray,
    feature_indices: FeaturesColumnIndices,
    labels_true: np.ndarray,
    labels_pred: Optional[np.ndarray] = None,
    pred_pos_probs: Optional[np.ndarray] = None,
    range_normalization_factor: float = 1.,
    only_plot_falsies: bool = False,
    title: str = ""
) -> Tuple[Figure, List[int]]:
    """
    Plot route-points data.

    Args:
        route_data: shape=(n_routes, route_len, n_features)
        nearest_points_data: shape=(n_routes, n_nearest_points, n_features)
        feature_indices: Feature column indices.
        labels_true: 0/1 true labels of shape=(n_routes, n_nearest_points).
        labels_pred: 0/1 labels of shape=(n_routes, n_nearest_points), output
            from model.
        pred_pos_probs: Positive probabilities of each (route, point) pair,
            output from model, shape=(n_routes, n_nearest_points)
        range_normalization_factor: In plotted figures, all distances will
            be multiplied with this value.
        only_plot_falsies: Whether only to plot falsely associated routes.
            Only effective if :attr:`labels_pred` and :attr:`pred_pos_probs`
            are provided; otherwise it will be overwritten to False (i.e.
            plot all routes).
        title: Figure title.

    Returns:
        Plotted figure and indices of plotted routes.

    """
    r_max = range_normalization_factor

    if labels_pred is None or pred_pos_probs is None:
        only_plot_falsies = False
        has_pred = False
    else:
        has_pred = True

    fig: Figure = plt.figure(figsize=(8, 5), dpi=200)
    fig.subplots_adjust(0.1, 0.1, 0.95, 0.93)
    gs = GridSpec(2, 2, fig,
                  width_ratios=[1, 1], height_ratios=[1, 1],
                  wspace=0.3, hspace=0.3)
    axes_xy: Axes = fig.add_subplot(gs[0, 0])
    axes_ra: Axes = fig.add_subplot(gs[0, 1])
    axes_tr: Axes = fig.add_subplot(gs[1, 0])
    axes_ta: Axes = fig.add_subplot(gs[1, 1])

    alpha = 0.5

    plotted_route_indices = []
    x_max = -np.inf
    x_min = np.inf
    y_max = -np.inf
    y_min = np.inf

    for i_route, (_route, _nearest_points, _y_true) in enumerate(
        zip(route_data, nearest_points_data, labels_true)
    ):
        # Evaluate true index of associated point
        _point_idx_true = np.where(_y_true)[0]
        if len(_point_idx_true) == 0:
            _point_idx_true = -1  # no point is associated
        else:
            _point_idx_true = _point_idx_true[0]

        # Evaluate predicted index of associated point (if provided)
        if has_pred:
            _y_pred = labels_pred[i_route]
            _point_idx_pred = np.where(_y_pred)[0]
            if len(_point_idx_pred) == 0:
                _point_idx_pred = -1  # no point is associated
            else:
                _point_idx_pred = _point_idx_pred[0]

            _correct = _point_idx_true == _point_idx_pred

            if only_plot_falsies and _correct:
                continue

        plotted_route_indices.append(i_route)

        route_real_len = np.sum(
            ~np.all(np.isnan(_route), axis=-1), axis=-1
        )

        _route = _route[:route_real_len]
        route_ts = _route[:, feature_indices.time]  # normalized
        route_rs = _route[:, feature_indices.range] * r_max
        route_azims = _route[:, feature_indices.azim] * 360  # degrees
        route_xs = (_route[:, feature_indices.x] * 2 * r_max - r_max)
        route_ys = (_route[:, feature_indices.y] * 2 * r_max - r_max)

        points_real_count = np.sum(
            ~np.all(np.isnan(_nearest_points), axis=-1), axis=-1
        )

        _nearest_points = _nearest_points[:points_real_count]
        points_ts = _nearest_points[:, feature_indices.time]  # normalized
        points_rs = _nearest_points[:, feature_indices.range] * r_max
        points_azims = _nearest_points[:, feature_indices.azim] * 360  # degrees
        points_xs = (_nearest_points[:, feature_indices.x] * 2 * r_max - r_max)
        points_ys = (_nearest_points[:, feature_indices.y] * 2 * r_max - r_max)

        route_not_nan_xs = route_xs[~np.isnan(route_xs)]
        route_not_nan_ys = route_ys[~np.isnan(route_ys)]
        points_not_nan_xs = points_xs[~np.isnan(points_xs)]
        points_not_nan_ys = points_ys[~np.isnan(points_ys)]
        x_max = max(x_max, route_not_nan_xs.max(), points_not_nan_xs.max())
        x_min = min(x_min, route_not_nan_xs.min(), points_not_nan_xs.min())
        y_max = max(y_max, route_not_nan_ys.max(), points_not_nan_ys.max())
        y_min = min(y_min, route_not_nan_ys.min(), points_not_nan_ys.min())

        for (
            _axes,
            _route_hori_feature,
            _route_vert_feature,
            _points_hori_feature,
            _points_vert_feature
        ) in (
            [axes_xy, route_xs, route_ys, points_xs, points_ys],
            [axes_ra, route_rs, route_azims, points_rs, points_azims],
            [axes_tr, route_ts, route_rs, points_ts, points_rs],
            [axes_ta, route_ts, route_azims, points_ts, points_azims]
        ):
            # plot route line
            route_plot = _axes.plot(
                _route_hori_feature, _route_vert_feature,
                alpha=alpha,
                label=f"route {i_route}"
            )[0]
            # plot lines connecting route to nearby points
            for i_point, (_point_hori, _point_vert) in enumerate(
                zip(_points_hori_feature, _points_vert_feature)
            ):
                _axes.plot(
                    [_route_hori_feature[-1], _point_hori],
                    [_route_vert_feature[-1], _point_vert],
                    linestyle="--" if _y_true[i_point] else ":",
                    color=route_plot.get_color(),
                    alpha=alpha
                )
            # plot all points in the route
            _axes.scatter(
                _route_hori_feature,
                _route_vert_feature,
                s=16,
                c=route_plot.get_color(),
                alpha=alpha
            )
            # plot all nearby points
            _axes.scatter(_points_hori_feature, _points_vert_feature,
                          marker="v",
                          s=25,
                          c=route_plot.get_color(),
                          alpha=alpha)
            # circle out the association result of the model (if provided)
            if has_pred:
                for i_point in range(len(_points_hori_feature)):
                    _axes.scatter(
                        _points_hori_feature[i_point],
                        _points_vert_feature[i_point],
                        marker="o" if i_point != _point_idx_pred else "v",
                        s=40,
                        c=("g" if _correct else "r"),
                        alpha=(pred_pos_probs[i_route, i_point]) / 3
                    )

        for _axes, _hori_label, _vert_label in (
            [axes_xy, '$x$ / km', '$y$ / km'],
            [axes_ra, '$r$ / km', r'Azimuth / $\degree$'],
            [axes_tr, '$t$ / (normalized unit)', '$r$ / km'],
            [axes_ta, '$t$ /  (normalized unit)', r'Azimuth / $\degree$']
        ):
            _axes.set_xlabel(_hori_label)
            _axes.set_ylabel(_vert_label)
            # _axes.legend()

    if len(plotted_route_indices):
        # In this case all limits are not NaN or Inf
        x_mid = (x_max + x_min) / 2
        x_range = (x_max - x_min) * 1.2
        y_mid = (y_max + y_min) / 2
        y_range = (y_max - y_min) * 1.2
        d_range = max(x_range, y_range)

        axes_xy.set_xlim(x_mid - d_range / 2, x_mid + d_range / 2)
        axes_xy.set_ylim(y_mid - d_range / 2, y_mid + d_range / 2)
        axes_xy.set_aspect(1)
        axes_xy.text(
            x=0.05, y=0.05,
            s=f"true point: {_point_idx_true}\n"
              f"pred point: {_point_idx_pred} " +
              (f'(prob = {pred_pos_probs[i_route, _point_idx_pred]:.2f})'
               if _point_idx_pred != -1 else
               f'(max prob = {np.max(pred_pos_probs[i_route]):.2f})'),
            fontsize=6,
            ha="left",
            va="bottom",
            bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=0.5"),
            transform=axes_xy.transAxes,
            alpha=0.6
        )

    if title:
        fig.suptitle(title)

    return fig, plotted_route_indices


def plot_metric_curve(
    metrics: List[Union[np.ndarray, Sequence[float]]],
    legends: List[str],
    title,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot curves of different metrics.

    Args:
        metrics: List of metric records, each of which contains values of the
            metric recorded in every epoch. They can have different lengths.
        legends: Legend labels of each metric.
        title: Title of the figure.
        save_path: Saving path of the plot. If being None, the plot won't be
            saved.

    """
    font_size = 13

    fig: Figure
    axes: Axes
    fig, axes = plt.subplots()

    for metric, legend in zip(metrics, legends):
        x = np.arange(1, len(metric) + 1, 1)
        axes.plot(x, metric, label=legend)

    axes.set_xlabel("Epoch")
    axes.set_title(title, fontsize=font_size + 5)
    axes.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    plt.close(fig)
