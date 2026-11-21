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

from __future__ import annotations

import copy
import importlib
import numbers
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Sequence, Union, Dict, DefaultDict

import h5py
import numba
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from model_association import radar_definition as rd
from model_association.anomaly_detection import eval_dynamic_vals
from model_association.data import data_preprocess  # noqa
from model_association.point_cc.config import CollectedDataConfig
from model_association.point_cc.feature_augment import FeatureAugmenter
from model_association.route_point_data_processor import HorizontalCoordPaddedFanSplitter, HorizontalCoordSplitter


@dataclass
class FeatureColumnIndices:
    time: Optional[int] = None
    x_true: Optional[int] = None
    y_true: Optional[int] = None
    r_true: Optional[int] = None
    azim_true: Optional[int] = None
    x_noisy: Optional[int] = None
    y_noisy: Optional[int] = None
    r_noisy: Optional[int] = None
    azim_noisy: Optional[int] = None
    route_id: Optional[int] = None
    circle_id: Optional[int] = None
    vx_noisy: Optional[int] = None
    vy_noisy: Optional[int] = None
    vx_true: Optional[int] = None
    vy_true: Optional[int] = None

    @staticmethod
    def from_dict(name_val_map) -> "FeatureColumnIndices":
        return FeatureColumnIndices(
            time=name_val_map.get("time"),
            x_true=name_val_map.get("x_true"),
            y_true=name_val_map.get("y_true"),
            r_true=name_val_map.get("r_true"),
            azim_true=name_val_map.get("azim_true"),
            x_noisy=name_val_map.get("x_noisy"),
            y_noisy=name_val_map.get("y_noisy"),
            r_noisy=name_val_map.get("r_noisy"),
            azim_noisy=name_val_map.get("azim_noisy"),
            route_id=name_val_map.get("route_id"),
            circle_id=name_val_map.get("circle_id"),
            vx_noisy=name_val_map.get("vx_noisy"),
            vy_noisy=name_val_map.get("vy_noisy"),
            vx_true=name_val_map.get("vx_true"),
            vy_true=name_val_map.get("vy_true")
        )


default_feature_columns = FeatureColumnIndices(
    time=1,
    x_true=24,
    y_true=25,
    r_true=4,
    azim_true=2,
    x_noisy=22,
    y_noisy=23,
    r_noisy=4,
    azim_noisy=8,
    route_id=17
)


@dataclass
class RealRadarRouteClusterFeatureColumnIndices:
    time: Optional[int] = None
    x: Optional[int] = None
    y: Optional[int] = None
    r: Optional[int] = None
    azim: Optional[int] = None
    route_id: Optional[int] = None
    circle_id: Optional[int] = None
    area_masks: Optional[List[int]] = None
    focusing_point_flag: Optional[int] = None

    @staticmethod
    def from_dict(name_val_map) -> "RealRadarRouteClusterFeatureColumnIndices":
        return RealRadarRouteClusterFeatureColumnIndices(
            time=name_val_map.get("time", None),
            x=name_val_map.get("x", None),
            y=name_val_map.get("y", None),
            r=name_val_map.get("r", None),
            azim=name_val_map.get("azim", None),
            route_id=name_val_map.get("route_id", None),
            circle_id=name_val_map.get("circle_id", None),
            area_masks=name_val_map.get("area_masks", None),
            focusing_point_flag=name_val_map.get("focusing_point_flag", None)
        )


class PointsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        data_file_pattern: Optional[str] = None,
        n_points_per_sample: int = 120,
        feature_columns: FeatureColumnIndices = default_feature_columns
    ):
        self.data_dir = Path(data_dir)
        self.data_file_paths = [
            path for path in self.data_dir.iterdir()
            if not data_file_pattern or re.match(data_file_pattern, path.name)
        ]
        self.n_points_per_sample = n_points_per_sample
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.data_file_paths)

    def __getitem__(self, index):
        data_file = self.data_file_paths[index]
        mat = loadmat(data_file)

        # Filter out false points
        data_label = mat["data_label"]  # shape=(n_points, n_features)
        data_label = data_label[data_label[:, self.feature_columns.route_id] > 0]

        # Sort points by time in ascending order
        sorted_point_indices = np.argsort(data_label[:, self.feature_columns.time])
        data_label = data_label[sorted_point_indices]

        return data_label[:self.n_points_per_sample]


def circular_area_mask(
    points: np.ndarray,
    coords_columns: Tuple[int, ...],
    radius: float,
    center: np.ndarray
) -> np.ndarray:
    center = np.squeeze(center)
    assert center.shape == (len(coords_columns),)
    return np.linalg.norm(
        points[:, coords_columns] - center,
        axis=-1
    ) < radius


def clip_route(
    route: np.ndarray,
    point_filter: Callable[[np.ndarray], np.ndarray],
    min_route_len: int = -1
):
    point_reserved_mask = point_filter(route)
    sub_routes_points_indices = []

    to_start_new_route = True
    for i_point, to_reserve in enumerate(point_reserved_mask):
        if to_reserve:
            if to_start_new_route:
                sub_routes_points_indices.append([])
                to_start_new_route = False
            sub_routes_points_indices[-1].append(i_point)
        else:
            to_start_new_route = True

    clipped_routes = [
        route[sub_route_points_indices]
        for sub_route_points_indices in sub_routes_points_indices
        if len(sub_route_points_indices) >= min_route_len
    ]

    return clipped_routes


class NoisyRouteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        speed_min,
        speed_max,
        range_max,
        n_routes_per_sample,
        ts,
        coord_noise_scale,
        azim_noise_scale: float = 0,
        pos_cross: Optional[np.ndarray] = None,
        pos_cross_range_max: Optional[float] = None,
        t_cross: Optional[float] = 0,
        normalize: bool = False,
        n_samples: int = 9999,
        feature_columns: FeatureColumnIndices = default_feature_columns
    ):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.range_max = range_max
        self.n_routes_per_sample = n_routes_per_sample
        self.ts = np.sort(ts)
        self.coord_noise_scale = coord_noise_scale
        self.azim_noise_scale = azim_noise_scale
        self.pos_cross = pos_cross
        self.pos_cross_range_max = pos_cross_range_max or self.range_max
        self.t_cross = t_cross
        self.normalize = normalize
        self.n_samples = n_samples

        self.feature_columns = feature_columns

    def generate_route(self, n_routes, ts, coord_noise_scale=None):
        dr_at_cross = np.random.rand(n_routes) * self.pos_cross_range_max
        dazim_at_cross = np.random.rand(n_routes) * 2 * np.pi - np.pi

        if self.pos_cross is None:
            _pos_cross_azim = np.random.rand() * 2 * np.pi - np.pi
            pos_cross = np.random.rand() * (self.range_max / 2.) * \
                        np.array([np.cos(_pos_cross_azim), np.sin(_pos_cross_azim)])
        else:
            pos_cross = self.pos_cross
        pos_cross = pos_cross + np.array(
            [dr_at_cross * np.cos(dazim_at_cross),
             dr_at_cross * np.sin(dazim_at_cross)]
        ).T
        # shape=(n_routes, 2)

        speed0 = (
                     np.random.rand(n_routes) * (self.speed_max - self.speed_min)
                 ) + self.speed_min
        v_direc = np.random.rand(n_routes) * np.pi * 2
        v0 = np.array([speed0 * np.cos(v_direc), speed0 * np.sin(v_direc)]).T
        # shape=(n_routes, 2)

        t_cross = (self.t_cross if self.t_cross is not None
                   else np.random.choice(ts))
        pos = pos_cross[:, None] + \
              np.einsum("ij,k->ikj", v0, (ts - t_cross))
        # shape=(n_routes, len(ts), 2)

        if coord_noise_scale:
            pos += np.random.normal(loc=0, scale=coord_noise_scale, size=pos.shape)

        ts_tiled = np.tile(ts.reshape(1, -1, 1),
                           [n_routes, 1, 1])
        return np.concatenate([ts_tiled, pos], axis=-1)

    def __len__(self):
        return self.n_samples

    @property
    def t_norm_factor(self):
        return self.ts[-1] - self.ts[0]

    def __getitem__(self, item):
        pos_true = self.generate_route(
            self.n_routes_per_sample,
            self.ts, coord_noise_scale=None
        )

        # Clip routes
        routes = []
        for _route in pos_true:
            _clipped_routes = clip_route(
                _route,
                lambda points: circular_area_mask(
                    points,
                    coords_columns=(1, 2),
                    radius=self.range_max,
                    center=np.zeros(2)
                ),
                min_route_len=5
            )
            routes += _clipped_routes

        points: np.ndarray = np.ones(
            (self.n_routes_per_sample, len(self.ts), 15)
        ) * np.nan
        for i_route, sub_route in enumerate(routes):
            # Write (t, x_true, y_true)
            points[
            i_route, :len(sub_route),
            [self.feature_columns.time,
             self.feature_columns.x_true,
             self.feature_columns.y_true]
            ] = sub_route.T
            points[i_route, :len(sub_route), self.feature_columns.route_id] = i_route

            # Evaluate and write (r_true, azim_true)
            xs_true, ys_true = sub_route.T[1:3]
            points_rs_true = np.sqrt(xs_true ** 2 + ys_true ** 2)
            points_azims_true = np.arctan2(ys_true, xs_true)
            points[i_route, :len(sub_route), self.feature_columns.r_true] = points_rs_true
            points[i_route, :len(sub_route), self.feature_columns.azim_true] = points_azims_true

            # Simulate noise on x and y
            if self.azim_noise_scale == 0:
                points_xs_noisy, points_ys_noisy = (sub_route[:, 1:3] + np.random.normal(
                    loc=0, scale=self.coord_noise_scale,
                    size=(len(sub_route), 2)
                )).T
                points_rs_noisy = np.sqrt(points_xs_noisy ** 2 + points_ys_noisy ** 2)
                points_azims_noisy = np.arctan2(points_ys_noisy, points_xs_noisy)
            else:
                points_azims_noisy = points_azims_true + np.random.normal(
                    loc=0, scale=self.azim_noise_scale,
                    size=points_azims_true.shape
                )
                points_rs_noisy = points_rs_true
                points_xs_noisy = points_rs_noisy * np.cos(points_azims_noisy)
                points_ys_noisy = points_rs_noisy * np.sin(points_azims_noisy)

            points[i_route, :len(sub_route), self.feature_columns.r_noisy] = points_rs_noisy
            points[i_route, :len(sub_route), self.feature_columns.azim_noisy] = points_azims_noisy
            points[i_route, :len(sub_route), self.feature_columns.x_noisy] = points_xs_noisy
            points[i_route, :len(sub_route), self.feature_columns.y_noisy] = points_ys_noisy

            # Write circle IDs
            points[i_route, :len(sub_route), self.feature_columns.circle_id] = \
                np.sum(
                    self.ts[None]
                    <= sub_route[:, self.feature_columns.time][:, None],
                    # shape=(len(sub_route), len(ts))
                    axis=-1
                ) - 1

            # Evaluate and write velocities
            route_points_tensor = torch.from_numpy(
                points[i_route, :len(sub_route)]
            )
            _, speed_noisy, v_direc_noisy, _, _ = eval_dynamic_vals(
                route_points_tensor[:, self.feature_columns.time].unsqueeze(0),
                route_points_tensor[:, [self.feature_columns.x_noisy,
                                        self.feature_columns.y_noisy]].unsqueeze(0)
            )
            _, speed_true, v_direc_true, _, _ = eval_dynamic_vals(
                route_points_tensor[:, self.feature_columns.time].unsqueeze(0),
                route_points_tensor[:, [self.feature_columns.x_true,
                                        self.feature_columns.y_true]].unsqueeze(0)
            )
            speed_true = speed_true[0].cpu().numpy()
            v_direc_true = v_direc_true[0].cpu().numpy()
            speed_noisy = speed_noisy[0].cpu().numpy()
            v_direc_noisy = v_direc_noisy[0].cpu().numpy()
            points[i_route, :len(sub_route) - 1, self.feature_columns.vx_noisy] = \
                speed_noisy * np.cos(v_direc_noisy)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vy_noisy] = \
                speed_noisy * np.sin(v_direc_noisy)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vx_true] = \
                speed_true * np.cos(v_direc_true)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vy_true] = \
                speed_true * np.sin(v_direc_true)
            points[i_route, len(sub_route) - 1, [
                self.feature_columns.vx_noisy,
                self.feature_columns.vy_noisy,
                self.feature_columns.vx_true,
                self.feature_columns.vy_true
            ]] = points[i_route, len(sub_route) - 2, [
                self.feature_columns.vx_noisy,
                self.feature_columns.vy_noisy,
                self.feature_columns.vx_true,
                self.feature_columns.vy_true
            ]]

        if self.normalize:
            points[:, :, self.feature_columns.time] /= self.ts[-1]
            points[:, :, [self.feature_columns.x_true,
                          self.feature_columns.y_true,
                          self.feature_columns.x_noisy,
                          self.feature_columns.y_noisy]] = \
                (points[:, :, [self.feature_columns.x_true,
                               self.feature_columns.y_true,
                               self.feature_columns.x_noisy,
                               self.feature_columns.y_noisy]] + self.range_max) \
                / (self.range_max * 2)
            points[:, :, [self.feature_columns.vx_noisy,
                          self.feature_columns.vy_noisy,
                          self.feature_columns.vx_true,
                          self.feature_columns.vy_true]] *= (self.ts[-1] / (self.range_max * 2))

        points = points.reshape(-1, points.shape[-1])
        # shape=(n_routes*len(self.ts), n_features)
        points = points[np.random.permutation(points.shape[0])]
        points = points[np.argsort(points[:, self.feature_columns.time])]

        return points


class NearlyParallelRoutesDataset(NoisyRouteDataset):
    def __init__(
        self,
        # speed_min,
        # speed_max,
        range_max,
        n_routes_per_sample,
        ts,
        coord_noise_scale,
        target_dist_max: float,
        azim_noise_scale: float = 0,
        n_samples: int = 9999,
        normalize: bool = False,
        feature_columns: FeatureColumnIndices = default_feature_columns
    ):
        super().__init__(
            speed_min=np.nan,
            speed_max=np.nan,
            range_max=range_max,
            n_routes_per_sample=n_routes_per_sample,
            ts=ts,
            coord_noise_scale=coord_noise_scale,
            azim_noise_scale=azim_noise_scale,
            pos_cross_range_max=np.nan,
            normalize=normalize,
            n_samples=n_samples,
            feature_columns=feature_columns
        )

        self.target_dist_max = target_dist_max

    def generate_route(self, n_routes, ts, coord_noise_scale=None):
        r_start, r_end = np.random.rand(2) * (self.range_max - self.target_dist_max)
        azim_start, azim_end = np.random.rand(2) * 2 * np.pi - np.pi
        pos_c_start = np.array([r_start * np.cos(azim_start),
                                r_start * np.sin(azim_start)])
        pos_c_end = np.array([r_end * np.cos(azim_end),
                              r_end * np.sin(azim_end)])

        rel_r_0 = np.random.rand(n_routes) * self.target_dist_max
        rel_azim_0 = np.random.rand(n_routes) * 2 * np.pi - np.pi
        rel_pos_0 = np.array([rel_r_0 * np.cos(rel_azim_0),
                              rel_r_0 * np.sin(rel_azim_0)]).T
        pos_0 = pos_c_start + rel_pos_0
        # shape=(n_routes, 2)

        rel_r_t = np.random.rand(n_routes) * self.target_dist_max
        rel_azim_t = np.random.rand(n_routes) * 2 * np.pi - np.pi
        rel_pos_t = np.array([rel_r_t * np.cos(rel_azim_t),
                              rel_r_t * np.sin(rel_azim_t)]).T
        pos_t = pos_c_end + rel_pos_t
        # shape=(n_routes, 2)

        pos = np.linspace(pos_0, pos_t, len(ts)).swapaxes(0, 1)
        # shape=(n_routes, len(ts), 2)

        if coord_noise_scale:
            pos += np.random.normal(loc=0, scale=coord_noise_scale, size=pos.shape)

        ts_tiled = np.tile(ts.reshape(1, -1, 1),
                           [n_routes, 1, 1])
        return np.concatenate([ts_tiled, pos], axis=-1)


class TwoRoutesCrossDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        speed_min,
        speed_max,
        range_max,
        ts,
        coord_noise_scale,
        azim_noise_scale: float = 0,
        pos_cross: Optional[np.ndarray] = None,
        pos_cross_range_max: Optional[float] = None,
        t_cross: Optional[float] = 0,
        normalize: bool = False,
        n_samples: int = 9999,
        feature_columns: FeatureColumnIndices = default_feature_columns
    ):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.range_max = range_max
        self.ts = np.sort(ts)
        self.coord_noise_scale = coord_noise_scale
        self.azim_noise_scale = azim_noise_scale
        self.pos_cross = pos_cross
        self.pos_cross_range_max = pos_cross_range_max or self.range_max
        self.t_cross = t_cross
        self.normalize = normalize
        self.n_samples = n_samples

        self.feature_columns = feature_columns

    def generate_route(self, ts, coord_noise_scale=None):
        speed0 = (
                     np.random.rand(2) * (self.speed_max - self.speed_min)
                 ) + self.speed_min
        v_direc_0 = np.random.rand() * np.pi * 2
        v_direc_1 = v_direc_0 + np.random.rand() * np.deg2rad(60) + np.deg2rad(60)
        v_direc = np.array([v_direc_0, v_direc_1])
        v0 = np.array([speed0 * np.cos(v_direc), speed0 * np.sin(v_direc)]).T
        # shape=(n_routes==2, 2)

        t_cross = (self.t_cross if self.t_cross is not None
                   else np.random.choice(ts))

        if self.pos_cross is None:
            _pos_cross_azim = np.random.rand() * 2 * np.pi - np.pi
            pos_cross = np.random.rand() * (self.range_max / 2.) * \
                        np.array([np.cos(_pos_cross_azim), np.sin(_pos_cross_azim)])
        else:
            pos_cross = self.pos_cross
        # shape=(2,)

        dt = ts[1] - ts[0]
        pos_cross = (pos_cross[None, :]  # shape=(1, 2)
                     + v0 * dt / 2)  # shape=(n_routes==2, 2)

        pos = pos_cross[:, None] + \
              np.einsum("ij,k->ikj", v0, (ts - t_cross))
        # shape=(n_routes, len(ts), 2)

        if coord_noise_scale:
            pos += np.random.normal(loc=0, scale=coord_noise_scale, size=pos.shape)

        ts_tiled = np.tile(ts.reshape(1, -1, 1),
                           [2, 1, 1])
        return np.concatenate([ts_tiled, pos], axis=-1)

    def __len__(self):
        return self.n_samples

    @property
    def t_norm_factor(self):
        return self.ts[-1] - self.ts[0]

    def __getitem__(self, item):
        pos_true = self.generate_route(
            self.ts, coord_noise_scale=None
        )

        # Clip routes
        routes = []
        for _route in pos_true:
            _clipped_routes = clip_route(
                _route,
                lambda points: circular_area_mask(
                    points,
                    coords_columns=(1, 2),
                    radius=self.range_max,
                    center=np.zeros(2)
                ),
                min_route_len=5
            )
            routes += _clipped_routes

        points: np.ndarray = np.ones(
            (2, len(self.ts), 15)
        ) * np.nan
        for i_route, sub_route in enumerate(routes):
            # Write (t, x_true, y_true)
            points[
            i_route, :len(sub_route),
            [self.feature_columns.time,
             self.feature_columns.x_true,
             self.feature_columns.y_true]
            ] = sub_route.T
            points[i_route, :len(sub_route), self.feature_columns.route_id] = i_route

            # Evaluate and write (r_true, azim_true)
            xs_true, ys_true = sub_route.T[1:3]
            points_rs_true = np.sqrt(xs_true ** 2 + ys_true ** 2)
            points_azims_true = np.arctan2(ys_true, xs_true)
            points[i_route, :len(sub_route), self.feature_columns.r_true] = points_rs_true
            points[i_route, :len(sub_route), self.feature_columns.azim_true] = points_azims_true

            # Simulate noise on x and y
            if self.azim_noise_scale == 0:
                points_xs_noisy, points_ys_noisy = (sub_route[:, 1:3] + np.random.normal(
                    loc=0, scale=self.coord_noise_scale,
                    size=(len(sub_route), 2)
                )).T
                points_rs_noisy = np.sqrt(points_xs_noisy ** 2 + points_ys_noisy ** 2)
                points_azims_noisy = np.arctan2(points_ys_noisy, points_xs_noisy)
            else:
                points_azims_noisy = points_azims_true + np.random.normal(
                    loc=0, scale=self.azim_noise_scale,
                    size=points_azims_true.shape
                )
                points_rs_noisy = points_rs_true
                points_xs_noisy = points_rs_noisy * np.cos(points_azims_noisy)
                points_ys_noisy = points_rs_noisy * np.sin(points_azims_noisy)

            points[i_route, :len(sub_route), self.feature_columns.r_noisy] = points_rs_noisy
            points[i_route, :len(sub_route), self.feature_columns.azim_noisy] = points_azims_noisy
            points[i_route, :len(sub_route), self.feature_columns.x_noisy] = points_xs_noisy
            points[i_route, :len(sub_route), self.feature_columns.y_noisy] = points_ys_noisy

            # Write circle IDs
            points[i_route, :len(sub_route), self.feature_columns.circle_id] = \
                np.sum(
                    self.ts[None]
                    <= sub_route[:, self.feature_columns.time][:, None],
                    # shape=(len(sub_route), len(ts))
                    axis=-1
                ) - 1

            # Evaluate and write velocities
            route_points_tensor = torch.from_numpy(
                points[i_route, :len(sub_route)]
            )
            _, speed_noisy, v_direc_noisy, _, _ = eval_dynamic_vals(
                route_points_tensor[:, self.feature_columns.time].unsqueeze(0),
                route_points_tensor[:, [self.feature_columns.x_noisy,
                                        self.feature_columns.y_noisy]].unsqueeze(0)
            )
            _, speed_true, v_direc_true, _, _ = eval_dynamic_vals(
                route_points_tensor[:, self.feature_columns.time].unsqueeze(0),
                route_points_tensor[:, [self.feature_columns.x_true,
                                        self.feature_columns.y_true]].unsqueeze(0)
            )
            speed_true = speed_true[0].cpu().numpy()
            v_direc_true = v_direc_true[0].cpu().numpy()
            speed_noisy = speed_noisy[0].cpu().numpy()
            v_direc_noisy = v_direc_noisy[0].cpu().numpy()
            points[i_route, :len(sub_route) - 1, self.feature_columns.vx_noisy] = \
                speed_noisy * np.cos(v_direc_noisy)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vy_noisy] = \
                speed_noisy * np.sin(v_direc_noisy)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vx_true] = \
                speed_true * np.cos(v_direc_true)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vy_true] = \
                speed_true * np.sin(v_direc_true)
            points[i_route, len(sub_route) - 1, [
                self.feature_columns.vx_noisy,
                self.feature_columns.vy_noisy,
                self.feature_columns.vx_true,
                self.feature_columns.vy_true
            ]] = points[i_route, len(sub_route) - 2, [
                self.feature_columns.vx_noisy,
                self.feature_columns.vy_noisy,
                self.feature_columns.vx_true,
                self.feature_columns.vy_true
            ]]

        if self.normalize:
            points[:, :, self.feature_columns.time] /= self.ts[-1]
            points[:, :, [self.feature_columns.x_true,
                          self.feature_columns.y_true,
                          self.feature_columns.x_noisy,
                          self.feature_columns.y_noisy]] = \
                (points[:, :, [self.feature_columns.x_true,
                               self.feature_columns.y_true,
                               self.feature_columns.x_noisy,
                               self.feature_columns.y_noisy]] + self.range_max) \
                / (self.range_max * 2)
            points[:, :, [self.feature_columns.vx_noisy,
                          self.feature_columns.vy_noisy,
                          self.feature_columns.vx_true,
                          self.feature_columns.vy_true]] *= (self.ts[-1] / (self.range_max * 2))

        points = points.reshape(-1, points.shape[-1])
        # shape=(n_routes*len(self.ts), n_features)
        points = points[np.random.permutation(points.shape[0])]
        points = points[np.argsort(points[:, self.feature_columns.time])]

        return points


class TwoRoutesCrossWithFakePointsDataset(TwoRoutesCrossDataset):
    def __init__(
        self,
        speed_min,
        speed_max,
        range_max,
        ts,
        coord_noise_scale,
        azim_noise_scale: float = 0,
        pos_cross: Optional[np.ndarray] = None,
        pos_cross_range_max: Optional[float] = None,
        t_cross: Optional[float] = 0,
        normalize: bool = False,
        n_samples: int = 9999,
        fake_points_ratio: float = 1.,
        feature_columns: FeatureColumnIndices = default_feature_columns
    ):
        super().__init__(
            speed_min,
            speed_max,
            range_max,
            ts,
            coord_noise_scale,
            azim_noise_scale,
            pos_cross,
            pos_cross_range_max,
            t_cross,
            normalize,
            n_samples,
            feature_columns
        )
        self.fake_points_ratio = fake_points_ratio

    def __getitem__(self, item):
        pos_true = self.generate_route(
            self.ts, coord_noise_scale=None
        )

        # Clip routes
        routes = []
        for _route in pos_true:
            _clipped_routes = clip_route(
                _route,
                lambda points: circular_area_mask(
                    points,
                    coords_columns=(1, 2),
                    radius=self.range_max,
                    center=np.zeros(2)
                ),
                min_route_len=5
            )
            routes += _clipped_routes

        points: np.ndarray = np.ones(
            (2, len(self.ts) * int(1 + self.fake_points_ratio), 15)
        ) * np.nan
        for i_route, sub_route in enumerate(routes):
            # Write (t, x_true, y_true)
            points[
            i_route, :len(sub_route),
            [self.feature_columns.time,
             self.feature_columns.x_true,
             self.feature_columns.y_true]
            ] = sub_route.T
            points[i_route, :len(sub_route), self.feature_columns.route_id] = i_route

            # Evaluate and write (r_true, azim_true)
            xs_true, ys_true = sub_route.T[1:3]
            points_rs_true = np.sqrt(xs_true ** 2 + ys_true ** 2)
            points_azims_true = np.arctan2(ys_true, xs_true)
            points[i_route, :len(sub_route), self.feature_columns.r_true] = points_rs_true
            points[i_route, :len(sub_route), self.feature_columns.azim_true] = points_azims_true

            # Simulate noise on x and y
            if self.azim_noise_scale == 0:
                points_xs_noisy, points_ys_noisy = (sub_route[:, 1:3] + np.random.normal(
                    loc=0, scale=self.coord_noise_scale,
                    size=(len(sub_route), 2)
                )).T
                points_rs_noisy = np.sqrt(points_xs_noisy ** 2 + points_ys_noisy ** 2)
                points_azims_noisy = np.arctan2(points_ys_noisy, points_xs_noisy)
            else:
                points_azims_noisy = points_azims_true + np.random.normal(
                    loc=0, scale=self.azim_noise_scale,
                    size=points_azims_true.shape
                )
                points_rs_noisy = points_rs_true
                points_xs_noisy = points_rs_noisy * np.cos(points_azims_noisy)
                points_ys_noisy = points_rs_noisy * np.sin(points_azims_noisy)

            points[i_route, :len(sub_route), self.feature_columns.r_noisy] = points_rs_noisy
            points[i_route, :len(sub_route), self.feature_columns.azim_noisy] = points_azims_noisy
            points[i_route, :len(sub_route), self.feature_columns.x_noisy] = points_xs_noisy
            points[i_route, :len(sub_route), self.feature_columns.y_noisy] = points_ys_noisy

            points_t = points[i_route, :len(sub_route), self.feature_columns.time]

            # add fake points
            points_fake_x = np.random.uniform(min(points_xs_noisy), max(points_xs_noisy),
                                              size=len(points[i_route]) - len(points_xs_noisy))
            points_fake_y = np.random.uniform(min(points_ys_noisy), max(points_ys_noisy),
                                              size=len(points[i_route]) - len(points_ys_noisy))
            points_fake_r = np.sqrt(points_fake_x ** 2 + points_fake_y ** 2)
            points_fake_az = np.arctan2(points_fake_y, points_fake_x)
            points[i_route, len(sub_route):, self.feature_columns.route_id] = -1
            points[i_route, len(sub_route):, self.feature_columns.r_noisy] = points_fake_r
            points[i_route, len(sub_route):, self.feature_columns.azim_noisy] = points_fake_az
            points[i_route, len(sub_route):, self.feature_columns.x_noisy] = points_fake_x
            points[i_route, len(sub_route):, self.feature_columns.y_noisy] = points_fake_y

            points[i_route, len(sub_route):, self.feature_columns.r_true] = points_fake_r
            points[i_route, len(sub_route):, self.feature_columns.azim_true] = points_fake_az
            points[i_route, len(sub_route):, self.feature_columns.x_true] = points_fake_x
            points[i_route, len(sub_route):, self.feature_columns.y_true] = points_fake_y

            points[i_route, len(sub_route):, self.feature_columns.time] = points_t

            # Write circle IDs
            points[i_route, :, self.feature_columns.circle_id] = np.array(
                points[i_route, :, self.feature_columns.time] / 10, dtype=np.int8)

            # Evaluate and write velocities
            route_points_tensor = torch.from_numpy(
                points[i_route, :len(sub_route)]
            )
            _, speed_noisy, v_direc_noisy, _, _ = eval_dynamic_vals(
                route_points_tensor[:, self.feature_columns.time].unsqueeze(0),
                route_points_tensor[:, [self.feature_columns.x_noisy,
                                        self.feature_columns.y_noisy]].unsqueeze(0)
            )
            _, speed_true, v_direc_true, _, _ = eval_dynamic_vals(
                route_points_tensor[:, self.feature_columns.time].unsqueeze(0),
                route_points_tensor[:, [self.feature_columns.x_true,
                                        self.feature_columns.y_true]].unsqueeze(0)
            )
            speed_true = speed_true[0].cpu().numpy()
            v_direc_true = v_direc_true[0].cpu().numpy()
            speed_noisy = speed_noisy[0].cpu().numpy()
            v_direc_noisy = v_direc_noisy[0].cpu().numpy()
            points[i_route, :len(sub_route) - 1, self.feature_columns.vx_noisy] = \
                speed_noisy * np.cos(v_direc_noisy)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vy_noisy] = \
                speed_noisy * np.sin(v_direc_noisy)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vx_true] = \
                speed_true * np.cos(v_direc_true)
            points[i_route, :len(sub_route) - 1, self.feature_columns.vy_true] = \
                speed_true * np.sin(v_direc_true)
            points[i_route, len(sub_route) - 1, [
                self.feature_columns.vx_noisy,
                self.feature_columns.vy_noisy,
                self.feature_columns.vx_true,
                self.feature_columns.vy_true
            ]] = points[i_route, len(sub_route) - 2, [
                self.feature_columns.vx_noisy,
                self.feature_columns.vy_noisy,
                self.feature_columns.vx_true,
                self.feature_columns.vy_true
            ]]

        if self.normalize:
            points[:, :, self.feature_columns.time] /= self.ts[-1]
            points[:, :, [self.feature_columns.x_true,
                          self.feature_columns.y_true,
                          self.feature_columns.x_noisy,
                          self.feature_columns.y_noisy]] = \
                (points[:, :, [self.feature_columns.x_true,
                               self.feature_columns.y_true,
                               self.feature_columns.x_noisy,
                               self.feature_columns.y_noisy]] + self.range_max) \
                / (self.range_max * 2)
            points[:, :, [self.feature_columns.vx_noisy,
                          self.feature_columns.vy_noisy,
                          self.feature_columns.vx_true,
                          self.feature_columns.vy_true]] *= (self.ts[-1] / (self.range_max * 2))

        points = points.reshape(-1, points.shape[-1])
        # shape=(n_routes*len(self.ts), n_features)
        points = points[np.random.permutation(points.shape[0])]
        points = points[np.argsort(points[:, self.feature_columns.time])]

        return points


def create_point_splitter(point_splitter_config: dict) -> HorizontalCoordSplitter:
    """
    Create a `HorizontalCoordSplitter` object from configuration dict.

    Args:
        point_splitter_config: Configuration of the splitter, which is organized as:

        .. code-block:

            {
                "type": <type name of splitter>,
                "module": <module of the splitter type>,
                "generate_params": {
                    // parameters of <type name of splitter>.generate_fully_covering_splitter
                    "r_max": 500e3,
                    "area_width_deg": 30,
                    "pad_width": 50e3,
                    ...
                }
            }

    Returns:
        Created splitter instance.

    """
    point_splitter_config = copy.deepcopy(point_splitter_config)
    point_splitter_module = importlib.import_module(point_splitter_config["module"])
    point_splitter_type = getattr(point_splitter_module, point_splitter_config["type"])

    if not issubclass(point_splitter_type, HorizontalCoordSplitter):
        warnings.warn(
            f"{point_splitter_type} is not a subclass of {HorizontalCoordSplitter}."
        )

    if not hasattr(point_splitter_type, "generate_fully_covering_splitter"):
        raise AttributeError(
            f"{point_splitter_type} does not implement `generate_fully_covering_splitter`."
        )

    point_splitter_params: dict = point_splitter_config["generate_params"]
    point_splitter_params["area_width"] = np.deg2rad(
        point_splitter_params.pop("area_width_deg")
    )

    point_splitter = point_splitter_type.generate_fully_covering_splitter(
        **point_splitter_config["generate_params"]
    )
    return point_splitter


def sample_points_filter(
    samples: np.ndarray,  # shape=(batch_size, sample_len, n_features)
    remove_the_last_track_points: bool,
    points_drop_window_length: float | np.ndarray = 0,  # shape=() | (batch_size,)
    fa_lookback_window_length: float | np.ndarray = np.inf,  # shape=() | (batch_size,)
    pad_token_value: float = np.nan,
    return_mask: bool = True
) -> np.ndarray | bool:
    """ This function must be called on not-clipped samples.

    Points in each sample are assumed to be sorted by ascending time, with
    all-NaN padding rows at the tail. If ``return_mask`` is true, the mask
    indicating which points to remove is returned. Otherwise the mask is
    applied: removed points are stably dropped (the remaining points move
    forward while keeping their original order), the sample length is
    unchanged, and the holes at the tail (including pre-existing all-NaN pad
    rows) are filled with ``pad_token_value``.
    """
    mask_for_removing: np.ndarray | bool = False
    if remove_the_last_track_points:
        point_indices = np.tile(
            np.arange(samples.shape[1])[None],
            [samples.shape[0], 1]
        )
        last_point_mask = samples[..., rd.PointFeatureIndex.ROUTE_END_POINT_IDX] == point_indices
        last_point_mask &= (samples[..., rd.PointFeatureIndex.TRACK_ID_MANUAL] > 0)
        # this rule only applies to certain targets

        mask_for_removing |= last_point_mask

    # t_mins = samples[
    #     np.arange(samples.shape[0]),
    #     0,
    #     rd.PointFeatureIndex.RADAR_TIME
    # ]  # shape=(batch_size,)
    t_maxes = np.max(
        np.nan_to_num(samples[..., rd.PointFeatureIndex.RADAR_TIME], nan=-999),
        axis=-1
    )  # shape=(batch_size,)

    # `points_drop_window_length` / `fa_lookback_window_length` may be a scalar
    # or of shape (batch_size,). Their time unit is the same as the RADAR_TIME
    # column. A window value of 0 (drop window) or inf (FA lookback) disables
    # the corresponding rule for that sample.
    points_drop_window_length = np.asarray(points_drop_window_length, dtype=float)
    if np.any(points_drop_window_length > 0):
        # The points (targets & false alarms) satisfying
        # (t_max - points_drop_window_length <= t <= t_max) will
        # not be input to the model

        # Points are already sorted by t in ascending order in each sample:
        points_in_last_window_mask = (
            (samples[:, :, rd.PointFeatureIndex.RADAR_TIME] >=
             (t_maxes - points_drop_window_length)[..., None])
            &
            (points_drop_window_length > 0)[..., None]
        )  # shape=(batch_size, sample_len)

        mask_for_removing |= points_in_last_window_mask

    fa_lookback_window_length = np.asarray(fa_lookback_window_length, dtype=float)
    if np.any(np.isfinite(fa_lookback_window_length)):
        # False alarms in t <= t_max - fa_lookback_window_length will not
        # be input to the model
        fa_to_drop_mask = (
            (samples[:, :, rd.PointFeatureIndex.RADAR_TIME] <=
             (t_maxes - fa_lookback_window_length)[..., None])
            &
            (samples[:, :, rd.PointFeatureIndex.TRACK_ID_MANUAL] == 0)
        )

        mask_for_removing |= fa_to_drop_mask

    if return_mask:
        return mask_for_removing

    if isinstance(mask_for_removing, bool):
        mask_for_removing = np.full(samples.shape[:2], mask_for_removing, dtype=bool)

    # Pre-existing all-NaN pad rows are treated as holes too, so that the tail
    # of the output is filled with a single uniform pad value.
    holes_mask = mask_for_removing | np.isnan(samples).all(axis=-1)
    order = np.argsort(holes_mask, axis=-1, kind="stable")  # kept rows first, original order
    filtered = samples[np.arange(samples.shape[0])[:, None], order]
    holes_mask = np.take_along_axis(holes_mask, order, axis=-1)

    filtered[holes_mask] = pad_token_value
    return filtered


@numba.njit(cache=True)
def split_area_samples(
    data,
    n_split_areas,
    radial_split_borders: Optional[tuple[
        tuple[float, float],
        ...
    ]] = ((-1, -1),)
):
    """
    Split point sequence into several samples corresponding to different areas.
    Note that this function WILL NOT clip the samples into certain length.

    Args:
        data: Points data with area masks appended, shape=(n_points, n_features +
            n_split_areas).
        n_split_areas: Number of split areas.
        radial_split_borders (experimental): If provided, each area will be further
            split into several radial regions, whose borders are items of this
            argument (for example, radial_split_borders=((0,100e3), (80e3,500e3))
            will make points in each area split into regions of 0<=r<=100e3 and
            80e3<=r<=50e3). Border with negative value means there is no limit
            of r on that direction. By default, areas will not be split.
            Note that this does not affect the area masks of each point.

    Returns:
        Tuple of:
            - Split area samples, shape=(n_nonempty_split_areas, sample_len,
              n_features + n_split_areas);

            - Area IDs (indices) of each sample, shape=(n_nonempty_split_areas,);

            - Indices of points in :attr:`data` of each sample, shape=
              (n_nonempty_split_areas, sample_len). -1 is filled in for padded
              NaN points;

            - The numbers of points in areas, shape=(n_nonempty_split_areas,).

    """
    indices_sorted_by_time = np.argsort(data[:, rd.PointFeatureIndex.RADAR_TIME.value])
    data = data[indices_sorted_by_time]

    n_point_features = data.shape[-1]  # NOTE: area masks are included

    area_points_counts = np.sum(data[:, -n_split_areas:], axis=0).astype(np.int64)
    # shape=(n_split_areas,)
    nonempty_area_mask = area_points_counts > 0
    max_points = int(area_points_counts.max())

    next_indices_to_write = np.zeros(
        n_split_areas, dtype=np.int64
    )

    samples = np.full((n_split_areas, max_points, n_point_features),
                      fill_value=np.nan, dtype=data.dtype)
    point_indices = np.full((n_split_areas, max_points), fill_value=-1, dtype=np.int64)
    for i_point in range(data.shape[0]):
        for i_area in range(n_split_areas):
            if not data[i_point, -n_split_areas + i_area]:  # this point is not in the area
                continue

            next_index_to_write = next_indices_to_write[i_area]
            samples[i_area, next_index_to_write] = data[i_point]
            point_indices[i_area, next_index_to_write] = indices_sorted_by_time[i_point]

            next_indices_to_write[i_area] += 1

    area_ids = np.arange(n_split_areas)[nonempty_area_mask]
    samples = samples[nonempty_area_mask]
    point_indices = point_indices[nonempty_area_mask]
    n_points_in_areas = area_points_counts[nonempty_area_mask]

    if radial_split_borders == ((-1, -1),) or radial_split_borders is None:
        return samples, area_ids, point_indices, n_points_in_areas

    # Split samples in the radial direction
    n_samples = len(samples)
    samples_r_split = np.full(
        (n_samples * len(radial_split_borders), *samples.shape[1:]),
        fill_value=np.nan,
        dtype=samples.dtype
    )
    area_ids_r_split = np.full(
        (n_samples * len(radial_split_borders),),
        fill_value=-1,
        dtype=area_ids.dtype
    )
    point_indices_r_split = np.full(
        (n_samples * len(radial_split_borders), *point_indices.shape[1:]),
        fill_value=-1,
        dtype=point_indices.dtype
    )
    n_points_in_areas_r_split = np.full(
        (n_samples * len(radial_split_borders),),
        fill_value=0,
        dtype=n_points_in_areas.dtype
    )
    n_sample_r_split = 0
    for i_sample, _sample in enumerate(samples):
        for r_min, r_max in radial_split_borders:
            in_region_mask = np.ones(len(_sample), dtype=np.bool_)
            if r_min >= 0:
                in_region_mask &= r_min <= _sample[:, rd.PointFeatureIndex.R.value]
            if r_max >= 0:
                in_region_mask &= _sample[:, rd.PointFeatureIndex.R.value] <= r_max

            _sample_in_region = _sample[in_region_mask]
            _point_count_in_region = len(_sample_in_region)
            if _point_count_in_region == 0:
                continue

            _sample_in_region = np.concatenate(
                (_sample_in_region,
                 np.full((max_points - _point_count_in_region, n_point_features),
                         fill_value=np.nan,
                         dtype=_sample_in_region.dtype)),
                axis=0
            )
            _point_indices_in_region = np.concatenate(
                (point_indices[i_sample][in_region_mask],
                 np.full((max_points - _point_count_in_region,),
                         fill_value=-1,
                         dtype=point_indices.dtype))
            )

            samples_r_split[n_sample_r_split] = _sample_in_region
            area_ids_r_split[n_sample_r_split] = area_ids[i_sample]
            point_indices_r_split[n_sample_r_split] = _point_indices_in_region
            n_points_in_areas_r_split[n_sample_r_split] = (
                np.isnan(_sample_in_region).sum(axis=-1)
                < _sample_in_region.shape[-1]
            ).sum()
            n_sample_r_split += 1

    return (samples_r_split[:n_sample_r_split],
            area_ids_r_split[:n_sample_r_split],
            point_indices_r_split[:n_sample_r_split],
            n_points_in_areas_r_split[:n_sample_r_split])


def _select_overlen_keep_indices(
    sample: np.ndarray,  # shape=(i_len, n_features), valid points only
    target_len: int,
    select_by_index: int,
    fa_track_id_index: int | None,
    fa_track_id_value: int,
    r_index: int,
    r_clutter_region: float,
) -> np.ndarray:
    """Choose which points to keep from an over-length sample.

    The sample has ``i_len > target_len`` valid points. The input is NOT assumed
    to be sorted by the ``select_by_index`` column. Points are dropped until
    ``target_len`` remain, ordered by increasing ``select_by_index`` value (the
    points with the smallest ``select_by_index`` values are dropped first):

    1. If ``fa_track_id_index`` is given, first drop the false alarms
       (``track_id == fa_track_id_value``) that lie inside the clutter region
       (``r <= r_clutter_region``), ordered by increasing ``select_by_index``.
       This prioritises dropping clutter-region false alarms over targets/other
       points regardless of their ``select_by_index`` value.
    2. Then drop the remaining points (targets and false alarms alike) with the
       smallest ``select_by_index`` values, until ``target_len`` points remain.

    If ``fa_track_id_index`` is ``None``, step 1 is skipped and only step 2 is
    applied (i.e. the points with the smallest ``select_by_index`` value are
    dropped, targets and false alarms treated identically).

    The returned indices are in ascending (original) index order, so the kept
    points preserve their original ordering regardless of how ``select_by_index``
    is ordered.
    """
    n = sample.shape[0]
    n_to_drop = n - target_len
    dropped = np.zeros(n, dtype=bool)

    if n_to_drop <= 0:
        return np.arange(n)

    # NaN is treated as the smallest value so that points with a missing
    # ``select_by_index`` column are dropped first.
    sort_col = np.nan_to_num(sample[:, select_by_index], nan=-np.inf)

    if fa_track_id_index is not None:
        is_fa = sample[:, fa_track_id_index] == fa_track_id_value
        in_clutter = sample[:, r_index] <= r_clutter_region  # NaN -> False
        fa_candidates = np.where(is_fa & in_clutter)[0]
        if fa_candidates.size:
            order = fa_candidates[np.argsort(sort_col[fa_candidates], kind="stable")]
            n_fa_drop = min(n_to_drop, order.size)
            dropped[order[:n_fa_drop]] = True
            n_to_drop -= n_fa_drop

    if n_to_drop > 0:
        remaining = np.where(~dropped)[0]
        order = remaining[np.argsort(sort_col[remaining], kind="stable")]
        dropped[order[:n_to_drop]] = True

    keep = np.where(~dropped)[0]
    return keep


def clip_samples(
    samples,  # shape=(B, ..., seq_len, n_features)
    pad_token_value: float | Callable[[np.ndarray | torch.Tensor], np.ndarray[bool]] = np.nan,
    max_sample_len: int | None = None,
    *,
    select_by_index: int | None = None,
    fa_track_id_index: int | None = None,
    fa_track_id_value: int = 0,
    r_index: int | None = None,
    r_clutter_region: float = np.inf,
) -> np.ndarray | torch.Tensor:
    """Clip/pad a batch of samples to a uniform length.

    Points in each sample are assumed to be sorted by ascending time, with the
    empty (padded) tokens at the tail. If ``max_sample_len`` is larger than the
    number of valid points, the sample is padded at the tail with
    ``pad_token_value``; if it is smaller, the sample is clipped.

    When ``select_by_index`` is ``None``, an over-length sample keeps its
    ``max_sample_len`` newest points (the oldest ones are dropped).

    When ``select_by_index`` is given (a column index), an over-length sample is
    culled by dropping the points with the smallest ``select_by_index`` values
    until ``max_sample_len`` points remain. If ``fa_track_id_index`` is also
    given, the false alarms (``track_id == fa_track_id_value``) inside the clutter
    region (``r <= r_clutter_region``) are dropped first (ordered by increasing
    ``select_by_index``), and only then the remaining points are dropped. The kept
    points preserve their original ordering (the input is not assumed to be sorted
    by ``select_by_index``).
    """
    if isinstance(samples, torch.Tensor):
        data = samples
        is_tensor = True
    else:
        data = np.asarray(samples)
        is_tensor = False
    if data.ndim == 2:
        data = data[None, ...]
    elif data.ndim < 2:
        raise ValueError("`samples` must at least have 2 dimensions.")

    batch_shape = data.shape[:-2]
    seq_len = data.shape[-2]
    n_features = data.shape[-1]
    data_flat = data.reshape(-1, seq_len, n_features)  # (M, seq_len, n_features)
    n_batches = data_flat.shape[0]

    data_np = data_flat.cpu().numpy() if is_tensor else data_flat

    if isinstance(pad_token_value, numbers.Number):
        if np.isnan(pad_token_value):
            def _pad_check(_d):
                return np.isnan(_d).all(-1)
        else:
            def _pad_check(_d):
                return np.all(_d == pad_token_value, axis=-1)
    else:
        _pad_check = pad_token_value

    pad_token_flag = _pad_check(data_np)  # shape=(M, seq_len)
    assert (pad_token_flag[..., :-1] <= pad_token_flag[..., 1:]).all(), \
        "There are padded tokens in the middle of the sequence."
    sample_lengths = (~pad_token_flag).sum(-1)  # shape=(M,)

    target_len = int(sample_lengths.max()) if max_sample_len is None else int(max_sample_len)

    if select_by_index is not None:
        if not (0 <= select_by_index < n_features):
            raise ValueError(
                f"`select_by_index` ({select_by_index}) is out of range "
                f"[0, {n_features})."
            )
        # ``fa_track_id_index`` is optional: when it is None, the over-length
        # culling drops points purely by ``select_by_index`` (no FA priority).
        # When it is given, the clutter-region FA priority is enabled and the R
        # column must be resolvable (defaulting to ``Rd.R`` if not provided).
        if fa_track_id_index is not None:
            if not (0 <= fa_track_id_index < n_features):
                raise ValueError(
                    f"`fa_track_id_index` ({fa_track_id_index}) is out of range "
                    f"[0, {n_features})."
                )
            if r_index is None:
                r_index = rd.PointFeatureIndex.R.value
            if not (0 <= r_index < n_features):
                raise ValueError(
                    f"`r_index` ({r_index}) is out of range [0, {n_features})."
                )

    if isinstance(pad_token_value, numbers.Number):
        pad_fill = pad_token_value
    else:
        pad_fill = np.nan
    output = np.full(
        (n_batches, target_len, n_features),
        fill_value=pad_fill,
        dtype=data_np.dtype,
    )

    for i_batch in range(n_batches):
        i_len = int(sample_lengths[i_batch])
        if i_len <= 0:
            continue
        if i_len >= target_len:
            if select_by_index is not None:
                keep_indices = _select_overlen_keep_indices(
                    data_np[i_batch, :i_len],
                    target_len,
                    select_by_index,
                    fa_track_id_index,
                    fa_track_id_value,
                    r_index,
                    r_clutter_region,
                )
                output[i_batch] = data_np[i_batch, keep_indices]
            else:
                output[i_batch] = data_np[i_batch, i_len - target_len:i_len]
        else:
            output[i_batch, :i_len] = data_np[i_batch, :i_len]

    output = output.reshape(batch_shape + (target_len, n_features))
    if is_tensor:
        return torch.as_tensor(output, dtype=samples.dtype, device=samples.device)
    return output


@numba.njit(parallel=True, cache=True)
def _normalize_samples_txys_impl(
    points_in_areas: np.ndarray,
    area_borders: np.ndarray,
    time_normalization_factors: np.ndarray,
    areas_inverse_transforms: np.ndarray,
    t_idx: int,
    x_idx: int,
    y_idx: int,
    next_t_idx: int = -1,
    next_x_idx: int = -1,
    next_y_idx: int = -1
):
    data_normalized = points_in_areas.copy()

    min_x, max_x, min_y, max_y = area_borders.T

    x_norm_factor = max_x - min_x  # shape=(n_areas,)
    y_norm_factor = max_y - min_y  # shape=(n_areas,)

    n_areas = data_normalized.shape[0]
    # Whether the next-point columns are present (need to be normalized).
    normalize_next = next_t_idx >= 0

    for i_area in numba.prange(n_areas):
        t_nf = time_normalization_factors[i_area]
        min_t_i = np.nanmin(data_normalized[i_area, :, t_idx])
        min_x_i = min_x[i_area]
        min_y_i = min_y[i_area]
        x_norm_factor_i = x_norm_factor[i_area]
        y_norm_factor_i = y_norm_factor[i_area]

        _inv_transform_mat = areas_inverse_transforms[i_area]
        for i_point in numba.prange(data_normalized.shape[1]):
            data_normalized[i_area, i_point, t_idx] = \
                (data_normalized[i_area, i_point, t_idx] - min_t_i) \
                / t_nf

            _x = data_normalized[i_area, i_point, x_idx]
            _y = data_normalized[i_area, i_point, y_idx]
            data_normalized[i_area, i_point, x_idx] = \
                (_inv_transform_mat[0, 0] * _x
                 + _inv_transform_mat[0, 1] * _y
                 - min_x_i) / x_norm_factor_i
            data_normalized[i_area, i_point, y_idx] = \
                (_inv_transform_mat[1, 0] * _x
                 + _inv_transform_mat[1, 1] * _y
                 - min_y_i) / y_norm_factor_i

            # The coords of the route's next point (stored in dedicated columns
            # in the same raw (radar) frame as :attr:`x_idx` / :attr:`y_idx`)
            # are normalized with the SAME per-area transform so that they
            # remain in the same space as the (normalized) start/end points.
            if normalize_next:
                data_normalized[i_area, i_point, next_t_idx] = \
                    (data_normalized[i_area, i_point, next_t_idx] - min_t_i) \
                    / t_nf

                _nx = data_normalized[i_area, i_point, next_x_idx]
                _ny = data_normalized[i_area, i_point, next_y_idx]
                data_normalized[i_area, i_point, next_x_idx] = \
                    (_inv_transform_mat[0, 0] * _nx
                     + _inv_transform_mat[0, 1] * _ny
                     - min_x_i) / x_norm_factor_i
                data_normalized[i_area, i_point, next_y_idx] = \
                    (_inv_transform_mat[1, 0] * _nx
                     + _inv_transform_mat[1, 1] * _ny
                     - min_y_i) / y_norm_factor_i

    return data_normalized


def normalize_samples_v2(
    points_in_areas: np.ndarray,
    area_borders: np.ndarray,
    area_inverse_transforms: np.ndarray,
    time_normalization_factor: Union[float, np.ndarray],
    r_normalization_factor: Union[float, np.ndarray] = 500e3,
    amp_normalization_factor: Union[float, np.ndarray] = 150,
    snr_normalization_factor: Union[float, np.ndarray] = 100,
    h_normalization_factor: Union[float, np.ndarray] = 20000,
    elev_normalization_factor: Union[float, np.ndarray] = np.pi,
    elev_min: Union[float, np.ndarray] = -np.pi / 2,
    half_speed_normalization_factor: Union[float, np.ndarray, None] = None,
    return_legacy_features: bool = False
):
    """
    Normalize generated samples.

    Args:
        points_in_areas: shape=(batch_size=num_areas, sample_len, n_features)
        area_borders: shape=(batch_size=num_areas, 4 [x_min, x_max, y_min, y_max])
        area_inverse_transforms: shape=(batch_size=num_areas, 2, 2)
        time_normalization_factor: shape=(batch_size=num_areas,)
        r_normalization_factor: shape=(batch_size=num_areas,)
        amp_normalization_factor: shape=(batch_size=num_areas,)
        snr_normalization_factor: shape=(batch_size=num_areas,)
        h_normalization_factor: shape=(batch_size=num_areas,)
        elev_normalization_factor: shape=(batch_size=num_areas,)
        elev_min: shape=(batch_size=num_areas,)
        half_speed_normalization_factor: shape=(batch_size=num_areas,)
        return_legacy_features: Whether to return legacy normalized features.

    Returns:
        A tuple of (data_normalized, batch_labels) or (data_normalized,
        batch_labels, legacy_features). The shapes are:
        - data_normalized.shape=(batch_size=num_areas, sample_len, n_features)
        - [DEPRECATED] None.  # batch_labels.shape=(batch_size=num_areas, sample_len, 6)
        - legacy_features.len=n_legacy_features,
          legacy_features.key.shape=(batch_size=num_areas, sample_len)

    """
    time_normalization_factor = np.broadcast_to(time_normalization_factor, (len(points_in_areas),))
    r_normalization_factor = np.broadcast_to(r_normalization_factor, (len(points_in_areas),))
    amp_normalization_factor = np.broadcast_to(amp_normalization_factor, (len(points_in_areas),))
    snr_normalization_factor = np.broadcast_to(snr_normalization_factor, (len(points_in_areas),))
    h_normalization_factor = np.broadcast_to(h_normalization_factor, (len(points_in_areas),))
    elev_normalization_factor = np.broadcast_to(elev_normalization_factor, (len(points_in_areas),))
    elev_min = np.broadcast_to(elev_min, (len(points_in_areas),))
    data_normalized = _normalize_samples_txys_impl(
        points_in_areas,
        area_borders,
        time_normalization_factor,
        area_inverse_transforms,
        rd.PointFeatureIndex.RADAR_TIME.value,
        rd.PointFeatureIndex.X.value,
        rd.PointFeatureIndex.Y.value,
        next_t_idx=rd.PointFeatureIndex.ROUTE_NEXT_POINT_T.value,
        next_x_idx=rd.PointFeatureIndex.ROUTE_NEXT_POINT_X.value,
        next_y_idx=rd.PointFeatureIndex.ROUTE_NEXT_POINT_Y.value
    )

    data_normalized[:, :, rd.PointFeatureIndex.R.value] /= r_normalization_factor[:, None]
    data_normalized[:, :, rd.PointFeatureIndex.ROUTE_NEXT_POINT_R.value] /= \
        r_normalization_factor[:, None]
    data_normalized[:, :, rd.PointFeatureIndex.AZ.value] = \
        (data_normalized[:, :, rd.PointFeatureIndex.AZ.value] + np.pi) / (2 * np.pi)
    data_normalized[:, :, rd.PointFeatureIndex.EL.value] = \
        (data_normalized[:, :, rd.PointFeatureIndex.EL.value] - elev_min[:, None]) \
        / elev_normalization_factor[:, None]
    data_normalized[:, :, rd.PointFeatureIndex.AMP.value] /= amp_normalization_factor[:, None]
    data_normalized[:, :, rd.PointFeatureIndex.SNR.value] /= snr_normalization_factor[:, None]
    data_normalized[:, :, rd.PointFeatureIndex.HEIGHT.value] /= h_normalization_factor[:, None]

    if half_speed_normalization_factor is not None:
        half_speed_normalization_factor = np.broadcast_to(half_speed_normalization_factor, (len(points_in_areas),))
        # data_normalized[:, :, rd.PointFeatureIndex.RADIAL_SPEED.value] /= half_speed_normalization_factor[:, None]
        data_normalized[:, :, rd.PointFeatureIndex.RADIAL_SPEED.value] = (
            (data_normalized[:, :, rd.PointFeatureIndex.RADIAL_SPEED.value] + half_speed_normalization_factor[:, None])
            / (2 * half_speed_normalization_factor[:, None])
        )

    # Following code are only for compatibility with legacy models.
    # In future releases they may be removed.
    legacy_features = {}
    if return_legacy_features:
        legacy_features = {
            rd.PointFeatureIndex.R_NORM_DEC1_SCALED_SIN:
                (np.sin((data_normalized[:, :, rd.PointFeatureIndex.R.value] % 0.1) * 10 * 2 * np.pi) + 1) / 2,
            rd.PointFeatureIndex.R_NORM_DEC1_SCALED_COS:
                (np.cos((data_normalized[:, :, rd.PointFeatureIndex.R.value] % 0.1) * 10 * 2 * np.pi) + 1) / 2,
            rd.PointFeatureIndex.X_NORM_DEC1_SCALED_SIN:
                (np.sin((data_normalized[:, :, rd.PointFeatureIndex.X.value] % 0.1) * 10 * 2 * np.pi) + 1) / 2,
            rd.PointFeatureIndex.X_NORM_DEC1_SCALED_COS:
                (np.cos((data_normalized[:, :, rd.PointFeatureIndex.X.value] % 0.1) * 10 * 2 * np.pi) + 1) / 2,
            rd.PointFeatureIndex.Y_NORM_DEC1_SCALED_SIN:
                (np.sin((data_normalized[:, :, rd.PointFeatureIndex.Y.value] % 0.1) * 10 * 2 * np.pi) + 1) / 2,
            rd.PointFeatureIndex.Y_NORM_DEC1_SCALED_COS:
                (np.cos((data_normalized[:, :, rd.PointFeatureIndex.Y.value] % 0.1) * 10 * 2 * np.pi) + 1) / 2,
        }

    if return_legacy_features:
        return data_normalized, None, legacy_features
    return data_normalized, None


normalize_samples = normalize_samples_v2


def gather_splitter_properties(
    point_splitters: Dict[int, HorizontalCoordPaddedFanSplitter],
    radar_type_ids: np.ndarray,
    area_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(radar_type_ids)
    area_borders = np.zeros((n_samples, 4), dtype=float)
    area_inverse_transforms = np.zeros((n_samples, 2, 2), dtype=float)
    for i, (rd_type_id, area_id) in enumerate(zip(radar_type_ids, area_ids)):
        _splitter = point_splitters[int(rd_type_id)]._circumrect_splitter
        area_borders[i] = np.array(
            [-_splitter.rect_widths[area_id] / 2,
             _splitter.rect_widths[area_id] / 2,
             -_splitter.rect_depths[area_id],
             _splitter.rect_lengths[area_id] - _splitter.rect_depths[area_id]]
        )
        area_inverse_transforms[i] = _splitter.areas_inverse_transforms[area_id]

    return area_borders, area_inverse_transforms


def split_intervals(
    vals: np.ndarray,
    val_range: float,
    val_step: float
) -> np.ndarray:
    """
    For a sorted value series :attr:`vals`, returns array of (i_start, i_end)
    indices of values where

        indices[i] == (i_start, i_end)
        vals[0] + i * val_step <= vals[i_start:i_end] <= vals[0] + i * val_step + val_range.

    Returns:
        Indices of start and end values in :attr:`vals` of different intervals.

    """
    indices = split_intervals_impl(vals, val_range, val_step)
    indices_set = {tuple(bounds) for bounds in indices}
    return np.array(sorted(indices_set, key=lambda x: x[0]))


@numba.njit(cache=True)
def split_intervals_impl(
    vals: np.ndarray,
    val_range: float,
    val_step: float
) -> np.ndarray:
    indices = np.full(
        (int(np.ceil((vals[-1] - vals[0]) / val_step)), 2),
        fill_value=-1,
        dtype=np.int64
    )
    v0 = vals[0]
    n_vals = len(vals)

    i = -1
    j = -1
    for i_v in range(len(vals)):
        v = vals[i_v]

        while v >= v0 + (i + 1) * val_step and i < len(indices) - 1:
            i += 1
            indices[i, 0] = i_v

        while v > v0 + (j + 1) * val_step + val_range and j < len(indices) - 1:
            j += 1
            indices[j, 1] = i_v

    for i in range(len(indices)):
        if indices[i, 0] == -1:
            break

        if indices[i, 1] == -1:
            indices[i, 1] = n_vals

    return indices[
        np.sum(indices == -1, axis=-1) != indices.shape[-1]
        ]


class HDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_path: Path,
        dset_names: Sequence[str],
    ):
        self.h5_path = hdf5_path
        self.h5_file = None

        self.open()

        self.dset_names = dset_names
        assert len(set(
            self.h5_file[dset_name].shape[0] for dset_name in dset_names
        )) == 1, "All datasets' lengths should be the same!"

        self._n_samples = self.h5_file[dset_names[0]].shape[0]

        self.close()

    def open(self):
        self.h5_file = h5py.File(self.h5_path, "r")

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __len__(self):
        return self._n_samples

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        if self.h5_file is None:
            raise RuntimeError(
                "h5_file is not set. Maybe you forget to call worker_init_fn "
                "to open the file in distributed training?"
            )

        samples = []
        for dset_name in self.dset_names:
            arr = np.array(self.h5_file[dset_name][index])
            if arr.ndim == 0:
                arr = arr.reshape(-1)

            samples.append(torch.from_numpy(arr))

        return samples


class IndexedDatasetWrapper:
    def __init__(self, dataset: HDF5Dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def open(self):
        self.dataset.open()

    def close(self):
        self.dataset.close()


def _extract_next_point_labels(
    points: np.ndarray,
    feature_indices: Sequence[int],
    nan_points_mask: np.ndarray
) -> np.ndarray:
    """Read the route's NEXT point features.

    Unlike the start/end points (whose positions are recorded as indices into
    the sample), the NEXT point lies *after* the sample end and is therefore
    not a row of ``points``. Instead its (t, x, y, r) is stored - already
    normalized to the same space as :attr:`RADAR_TIME` / :attr:`X` /
    :attr:`Y` / :attr:`R` - in the ``ROUTE_NEXT_POINT_{T,X,Y,R}`` columns.
    Only features among RADAR_TIME, X, Y and R can be returned; anything else
    raises.
    """
    next_t = np.take(points, indices=rd.PointFeatureIndex.ROUTE_NEXT_POINT_T.value, axis=-1)
    next_x = np.take(points, indices=rd.PointFeatureIndex.ROUTE_NEXT_POINT_X.value, axis=-1)
    next_y = np.take(points, indices=rd.PointFeatureIndex.ROUTE_NEXT_POINT_Y.value, axis=-1)
    next_r = np.take(points, indices=rd.PointFeatureIndex.ROUTE_NEXT_POINT_R.value, axis=-1)
    time_values = (rd.PointFeatureIndex.RADAR_TIME.value,)

    vec = []
    for feat in feature_indices:
        feat_val = int(feat)
        if feat_val in time_values:
            vec.append(next_t)
        elif feat_val == rd.PointFeatureIndex.X.value:
            vec.append(next_x)
        elif feat_val == rd.PointFeatureIndex.Y.value:
            vec.append(next_y)
        elif feat_val == rd.PointFeatureIndex.R.value:
            vec.append(next_r)
        else:
            raise ValueError(
                f"NEXT point labels only support regressing RADAR_TIME, X, Y, R, "
                f"but `feature_indices` contains {feat}."
            )

    next_label = np.stack(vec, axis=-1)
    next_label[nan_points_mask] = np.nan
    return next_label


def extract_route_ends_features(
    points: np.ndarray,
    feature_indices: Optional[Sequence[int]] = None,
    n_extra_loc_points: Optional[int] = 0
) -> List[np.ndarray]:
    """
    Extract features of the route start, end and next point of each point.

    Args:
        points: input points data, shape=(sample_len, n_points_features)
            or (n_samples, sample_len, n_points_features). For each point,
            the `rd.PointFeatureIndex.ROUTE_START_POINT_IDX`-th column and
            the `rd.PointFeatureIndex.ROUTE_END_POINT_IDX`-th column record
            the indices of its route's start and end point in `points`.
            The route's next point is stored directly in the
            `rd.PointFeatureIndex.ROUTE_NEXT_POINT_{T,X,Y}` columns.
        feature_indices: Indices of features to be extracted. If not given
            (being None), all features will be returned. Must be given when
            `n_extra_loc_points` >= 1 because the NEXT point features are
            read from dedicated coordinate columns.
        n_extra_loc_points: The number of extra location points in the route
            to be extracted. When it is >= 1, the 1st extra location point is
            the route's NEXT point (the point immediately after the END point
            in the original data - see :mod:`generate_dataset`); it is read
            directly from the ROUTE_NEXT_POINT_* columns rather than as an
            index into ``points``.

    Returns:
        (2+N)-tuple of extracted features of route start / end / next
        points for each point, each of which is of shape=
        (sample_len, len(feature_indices)) or (n_samples, sample_len,
        len(feature_indices)). Here N is n_extra_loc_points.

    Notes:
        For padded NaN points, returned features of the route points will be
        all-NaN. For (valid) points whose route has no NEXT point, the NEXT
        label is all-NaN too.

    """
    max_extra_loc_points = (1  # NEXT
                            + 2)  # legacy LOC2/LOC3
    if n_extra_loc_points > max_extra_loc_points:
        raise ValueError(
            f"Currently at most {max_extra_loc_points} extra locations "
            f"are supported. "
            f"But `n_extra_loc_points` is set to {n_extra_loc_points}."
        )

    nan_points_mask = np.all(np.isnan(points), axis=-1)
    # Points with all-NaN features are padded points.

    if points.ndim not in (2, 3):
        raise ValueError(
            "`points` must be of 2-D or 3-D, but its dimension is "
            f"{points.shape}"
        )

    if points.ndim == 3:
        area_indices = np.expand_dims(np.arange(points.shape[0]), -1)
    else:
        area_indices = None

    def _lookup_by_index(point_idx_col: int) -> np.ndarray:
        _end_indices = np.nan_to_num(np.take(
            points,
            indices=point_idx_col,
            axis=-1
        ), nan=0).astype(int)
        if points.ndim == 2:
            _end_points = points[_end_indices]
        else:
            _end_points = points[area_indices, _end_indices]

        _end_points = _end_points.copy()
        _end_points[nan_points_mask] = np.nan

        if feature_indices is not None:
            _end_points = np.take(_end_points, indices=feature_indices, axis=-1)

        return _end_points

    ends_points = [
        _lookup_by_index(rd.PointFeatureIndex.ROUTE_START_POINT_IDX.value),
        _lookup_by_index(rd.PointFeatureIndex.ROUTE_END_POINT_IDX.value),
    ]

    # 1st extra location point: the route's NEXT point, read directly from the
    # ROUTE_NEXT_POINT_* columns.
    if n_extra_loc_points >= 1:
        if feature_indices is None:
            raise ValueError(
                "`feature_indices` must be given to extract the NEXT point "
                "features."
            )
        ends_points.append(_extract_next_point_labels(
            points, feature_indices, nan_points_mask
        ))

    # 2nd / 3rd extra location points: legacy LOC2 / LOC3 (index-based).
    for i_extra, loc_col in enumerate(
        [rd.PointFeatureIndex.ROUTE_LOC2_POINT_IDX.value,
         rd.PointFeatureIndex.ROUTE_LOC3_POINT_IDX.value],
        start=1
    ):
        if i_extra < n_extra_loc_points:
            ends_points.append(_lookup_by_index(loc_col))

    return ends_points


def append_legacy_features(
    batch_data_normalized: Union[np.ndarray, torch.Tensor],
    legacy_features: Dict[int, np.ndarray],
    input_feature_indices: Sequence[int]
) -> Union[np.ndarray, torch.Tensor]:
    idx_mapping = {}  # mapping: idx in batch_data_normalized -> idx in input_data
    idx_mapping_non_legacy = {}
    idx_mapping_legacy = {}
    for _idx_input, _idx_norm in enumerate(input_feature_indices):
        idx_mapping[_idx_norm] = _idx_input
        if rd.PointFeatureIndex(_idx_norm).is_legacy_feature:
            idx_mapping_legacy[_idx_norm] = _idx_input
        else:
            idx_mapping_non_legacy[_idx_norm] = _idx_input

    if isinstance(batch_data_normalized, np.ndarray):
        input_data = np.zeros(shape=batch_data_normalized.shape[:2] + (len(input_feature_indices),),
                              dtype=batch_data_normalized.dtype)
    elif isinstance(batch_data_normalized, torch.Tensor):
        input_data = torch.zeros(size=batch_data_normalized.shape[:2] + (len(input_feature_indices),),
                                 dtype=batch_data_normalized.dtype,
                                 device=batch_data_normalized.device)
        legacy_features = {
            idx: torch.from_numpy(legacy_features[idx]).to(
                device=batch_data_normalized.device,
                dtype=batch_data_normalized.dtype
            )
            for idx in legacy_features
        }
    else:
        raise ValueError(f"Unsupported type of batch_data_normalized: "
                         f"{type(batch_data_normalized)}")
    input_data[:, :, list(idx_mapping_non_legacy.values())] \
        = batch_data_normalized[:, :, list(idx_mapping_non_legacy.keys())]
    for _idx_norm, _idx_input in idx_mapping_legacy.items():
        input_data[:, :, _idx_input] = legacy_features[_idx_norm]
    return input_data


def process_model_data(
    batch_samples: torch.Tensor,  # shape=(batch_size, sample_len, n_features)
    batch_area_ids: torch.Tensor,  # shape=(batch_size, 1)
    input_feature_indices: Tuple[int, ...],
    regress_feature_indices: Tuple[int, ...],
    extra_n_in_features: int,
    input_feature_augmenter: Optional[FeatureAugmenter],
    regress_feature_augmenter: Optional[FeatureAugmenter],
    collected_data_config: CollectedDataConfig,
    point_splitters: DefaultDict[int, HorizontalCoordPaddedFanSplitter],
    reg_the_last_but_one_point: bool,
    device,
    output_labels: bool = True,
    loss_on_track_without_next: bool = True
):
    # NOTE: all point-removal (last track point, last circle, old false alarms)
    # is done at dataset generation time, before clipping, so that the samples
    # stored in the dataset are exactly what the model sees as input.
    from model_association.radar_definition import PointFeatureIndex

    batch_samples = batch_samples.to(device)
    batch_area_ids = batch_area_ids.squeeze(-1)  # (n_samples, 1) -> (n_samples,)

    rd_type_ids = batch_samples[:, 0, PointFeatureIndex.RADAR_TYPE].cpu().long().numpy()
    area_borders, area_inverse_transforms = gather_splitter_properties(
        point_splitters,
        rd_type_ids,
        batch_area_ids.cpu().numpy()
    )
    time_normalization_factors = np.array(
        [collected_data_config.get_config(id).normalization_factor.time for id in rd_type_ids]
    )
    r_normalization_factors = np.array(
        [collected_data_config.get_config(id).normalization_factor.r for id in rd_type_ids]
    )
    amp_normalization_factors = np.array(
        [collected_data_config.get_config(id).normalization_factor.amp for id in rd_type_ids]
    )
    snr_normalization_factors = np.array(
        [collected_data_config.get_config(id).normalization_factor.snr for id in rd_type_ids]
    )
    h_normalization_factors = np.array(
        [collected_data_config.get_config(id).normalization_factor.height for id in rd_type_ids]
    )
    elev_mins = np.deg2rad(np.array(
        [collected_data_config.get_config(id).normalization_factor.elev.min for id in rd_type_ids]
    ))
    elev_normalization_factors = np.deg2rad(np.array(
        [collected_data_config.get_config(id).normalization_factor.elev.range for id in rd_type_ids]
    ))
    half_speed_normalization_factors = np.array(
        [(collected_data_config.get_config(id).normalization_factor.speed_half
          if collected_data_config.get_config(id).normalization_factor.speed_half is not None
          else 1)
         for id in rd_type_ids]
    )

    # Patch: clip out-of-bound values
    # TODO: this part should be handled in dataset generation, not here.
    speed_max = torch.tensor(np.array(
        [(collected_data_config.get_config(id).max_value.speed
          if collected_data_config.get_config(id).max_value.speed is not None
          else np.inf)
         for id in rd_type_ids]
    ), device=device).reshape([-1, 1])
    torch.clip_(batch_samples[..., PointFeatureIndex.RADIAL_SPEED], min=-speed_max, max=speed_max)
    torch.clip_(batch_samples[..., PointFeatureIndex.AMP], min=0)
    torch.clip_(batch_samples[..., PointFeatureIndex.SNR], min=0)

    use_legacy_features = any(map(lambda i: i.is_legacy_feature, input_feature_indices))
    normalization_output = normalize_samples(
        batch_samples.cpu().numpy(),
        area_borders,
        area_inverse_transforms,
        time_normalization_factor=time_normalization_factors,
        r_normalization_factor=r_normalization_factors,
        amp_normalization_factor=amp_normalization_factors,
        snr_normalization_factor=snr_normalization_factors,
        h_normalization_factor=h_normalization_factors,
        elev_normalization_factor=elev_normalization_factors,
        elev_min=elev_mins,
        half_speed_normalization_factor=half_speed_normalization_factors,
        return_legacy_features=use_legacy_features
    )
    batch_data_normalized = normalization_output[0]
    legacy_features = normalization_output[2] if use_legacy_features else None
    batch_labels_ends = extract_route_ends_features(
        batch_data_normalized,
        feature_indices=regress_feature_indices,
        n_extra_loc_points=1 if reg_the_last_but_one_point else 0
    )  # (2+n)-tuple, each element is of shape=(n_samples, sample_len, n_features).
    batch_data_normalized = torch.from_numpy(
        batch_data_normalized
    ).to(device, dtype=torch.float32)
    batch_labels_ends = [
        torch.from_numpy(_label_ends).to(device, dtype=torch.float32)
        for _label_ends in batch_labels_ends
    ]

    if not use_legacy_features:
        input_data = batch_data_normalized[:, :, input_feature_indices]
    else:
        input_data = append_legacy_features(batch_data_normalized,
                                            legacy_features,
                                            input_feature_indices)

    valid_point_mask = (~input_data.isnan().any(-1))
    # for points whose input features contain no NaN, the mask is True
    true_target_labels = (
        valid_point_mask
        & (batch_data_normalized[:, :, PointFeatureIndex.TRACK_ID_MANUAL] > 0)
    )
    # for true targets, the mask is True
    non_nan_points_mask = valid_point_mask
    # for true targets and false points, the mask is True

    # Augment input features if needed
    if input_feature_augmenter is not None:
        input_data_aug = input_feature_augmenter.augment(input_data)
        input_data = torch.concatenate(
            [input_data, input_data_aug],
            dim=-1
        )
    input_data[~non_nan_points_mask] = 0

    # Augment label features if needed
    if output_labels:
        if regress_feature_augmenter is not None:
            batch_labels_ends_aug = []
            for _end_label in batch_labels_ends:
                _end_label_aug = regress_feature_augmenter.augment(_end_label)
                batch_labels_ends_aug.append(_end_label_aug)
            batch_labels_expanded = torch.concatenate(
                batch_labels_ends + batch_labels_ends_aug,
                dim=-1
            )
        else:
            batch_labels_expanded = torch.concatenate(
                batch_labels_ends,
                dim=-1
            )
    else:
        batch_labels_expanded = None

    # The route's NEXT point may be absent (END is the last point of the route
    # in the original data, or the gap to the next point is too large). In that
    # case its label coords are NaN (masked out per position). When
    # ``loss_on_track_without_next`` is false, the points of such whole routes
    # are instead excluded from the regression objective entirely by setting
    # their whole label row to NaN (the classification / circle / association
    # objectives of those points are unaffected).
    if (batch_labels_expanded is not None
        and not loss_on_track_without_next
        and reg_the_last_but_one_point):
        _track_id = batch_data_normalized[
            ..., PointFeatureIndex.TRACK_ID_MANUAL.value]
        _has_next = torch.isfinite(
            batch_data_normalized[..., PointFeatureIndex.ROUTE_NEXT_POINT_T.value])
        _rows_without_next = (_track_id > 0) & (~_has_next) & non_nan_points_mask
        if _rows_without_next.any():
            batch_labels_expanded[_rows_without_next] = float("nan")

    if extra_n_in_features > 0:
        input_data = torch.nn.functional.pad(input_data, pad=(0, extra_n_in_features))

    return {
        "batch_samples": batch_samples,
        "model_input": input_data,
        "data_normalized": batch_data_normalized,
        "true_target_labels": true_target_labels,
        "normalization_factors": {
            "r": r_normalization_factors,
            "time": time_normalization_factors,
            "amp": amp_normalization_factors,
            "snr": snr_normalization_factors,
            "h": h_normalization_factors,
            "elev": elev_normalization_factors,
            "half_speed": half_speed_normalization_factors,
        },
        "regress_labels": batch_labels_expanded,
        "non_nan_points_mask": non_nan_points_mask,
        "rd_type_ids": rd_type_ids,
        "area_ids": batch_area_ids,
        "area_borders": area_borders,
        "area_inverse_transforms": area_inverse_transforms
    }
