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

import abc
import functools
import itertools
import numbers
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence

import numba.experimental
import numpy as np
import numpy.typing as npt


class RoutePointDataProcessor(object):

    def __init__(
        self,
        route_length: int,
        sample_length: Optional[int],
        dist_thresh: float,
        min_route_length: Optional[int],
        circle_id_idx: int,
        route_id_idx: int,
        x_idx: int,
        y_idx: int
    ):
        self.route_length = route_length
        self.sample_length: int = (
            sample_length
            if isinstance(sample_length, int) and sample_length > 0
            else -1
        )
        self.dist_thresh = dist_thresh
        self.min_route_length = (
            min_route_length
            if isinstance(min_route_length, int) and min_route_length > 0
            else -1
        )

        self.circle_id_idx = circle_id_idx
        self.route_id_idx = route_id_idx
        self.x_idx = x_idx
        self.y_idx = y_idx

        # Caches
        self._n_features = None
        self._route_data = None
        self._route_lengths = None
        self._last_circ_data = None

    def collect_route_data(self, data: np.ndarray):
        """
        Collect route data of previous circles and point data in the last circle
        separately.

        Args:
            data: Input data of shape=(n_points, n_features), usually returned
                from :func:`GetDataWithMask.__getitem__`.

        Return:
            Tuple of (route_data, route_lengths, last_circ_data), whose shapes are:

            .. code-block::

                route_data.shape=(num_routes, self.route_length, n_features)
                route_lengths.shape=(num_routes,)
                last_circ_data.shape=(n_points_in_last_circ, n_features)

            These tensors will also be cached.

        """
        circle_ids = data[:, self.circle_id_idx]
        circle_id_set = np.sort(np.unique(circle_ids))
        last_circle_id = circle_id_set[-1]

        prev_circ_data = data[circle_ids < last_circle_id]
        last_circ_data = data[circle_ids == last_circle_id]
        n_features = prev_circ_data.shape[-1]

        prev_route_ids = prev_circ_data[:, self.route_id_idx]
        prev_route_id_set = np.sort(np.unique(prev_route_ids))[::-1]

        route_data = []
        route_lengths = []
        for i_route, route_id in enumerate(prev_route_id_set):
            if route_id <= 0:
                continue

            _route_points = prev_circ_data[prev_route_ids == route_id]
            # shape=(n_points_in_route, n_features), increasingly sorted by time
            n_points_in_route = len(_route_points)
            route_lengths.append(n_points_in_route)

            # Fill in the data of latest points of the route
            n_filled_points = min(n_points_in_route, self.route_length)
            _route_data = np.zeros(
                [self.route_length, n_features], dtype=np.float32
            )
            _route_data[:n_filled_points] = \
                _route_points[-n_filled_points:]
            route_data.append(_route_data)

        self._route_data = np.array(route_data)
        self._route_lengths = np.array(route_lengths)
        self._last_circ_data = last_circ_data
        return self._route_data, self._route_lengths, self._last_circ_data

    def concat_route_data(
        self,
        route_data,
        route_lengths,
        last_circ_data,
        dist_thresh,
        pad_with=0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine route data of previous circles and point data in the last circle,
        and concatenate them.

        Args:
            route_data: shape=(self.num_routes, self.route_length, n_features)
            route_lengths: shape=(self.num_routes,)
            last_circ_data: shape=(n_points_in_last_circ, n_features)
            dist_thresh: threshold for filtering close route-point pairs
            pad_with: value used to pad

        Return:
             Tuple of combined data, corresponding labels and (route index, point
             index) pairs of each row in combined data, shapes are:

             .. code-block::

                combined_data.shape=(n_combs, self.route_length + 1, n_features)
                labels.shape=(n_combs,)
                index_pair.shape=(n_combs, 2)

            where :code:`n_combs <= self.num_routes * n_points_in_last_circ`
            ("<" because only those (route, point) pairs satisfying dist(route, point)
            <= self.dist_thresh are considered).

        """
        combined_data = []
        labels = []
        index_pair = []

        self._n_features = route_data.shape[-1]
        n_routes = len(route_data)
        n_points_last_circ = len(last_circ_data)

        for i_comb, (i_route, i_point) in enumerate(
            itertools.product(range(n_routes), range(n_points_last_circ))
        ):
            _route_data = route_data[i_route]
            _route_len = route_lengths[i_route]
            _point_data = last_circ_data[i_point]

            if _route_len < self.min_route_length:
                continue

            _route_data = _route_data[:_route_len]  # clip tailing padded zero points

            route_point_dist = np.linalg.norm(
                (_route_data[-1, [self.x_idx, self.y_idx]]
                 - _point_data[[self.x_idx, self.y_idx]])
            )
            if route_point_dist > dist_thresh:
                continue

            combined_data.append(
                np.concatenate([
                    _route_data,
                    [_point_data],
                    np.ones([self.route_length - len(_route_data),
                             self._n_features]) * pad_with
                ], axis=0)
            )
            labels.append(
                _route_data[0, self.route_id_idx] ==
                _point_data[self.route_id_idx]
            )
            index_pair.append(
                (i_route, i_point)
            )

        if len(combined_data) == 0:  # boundary case
            combined_data = np.zeros(
                (0, self.route_length + 1, self._n_features),
                dtype=np.float32
            )

        return (np.array(combined_data, dtype=np.float32),
                np.array(labels, dtype=bool),
                np.array(index_pair))

    def process_route_data(self, data_ori):
        self._n_features = data_ori.shape[-1]
        combined_data_shape = (self.sample_length, self.route_length + 1, self._n_features)

        combined_data, labels, _ = self.concat_route_data(
            *self.collect_route_data(data_ori),
            dist_thresh=self.dist_thresh
        )

        n_combs = len(combined_data)
        if n_combs < self.sample_length:
            combined_data = np.concatenate(
                [combined_data,
                 np.zeros(
                     [self.sample_length - n_combs, *combined_data_shape[1:]],
                     dtype=combined_data.dtype
                 )],
                axis=0
            )
            labels = np.concatenate(
                [labels,
                 np.zeros([self.sample_length - n_combs], dtype=labels.dtype)],
                axis=0
            )
        elif self.sample_length > 0:
            combined_data = combined_data[:self.sample_length]
            labels = labels[:self.sample_length]
            n_combs = self.sample_length

        if self.sample_length > 0:
            assert combined_data.shape == combined_data_shape
            assert labels.shape == combined_data_shape[:1]
        else:
            assert combined_data.shape[1:] == combined_data_shape[1:]
        if self.sample_length > 0:
            assert n_combs <= self.sample_length
        return combined_data, labels, n_combs


@dataclass
class FeaturesColumnIndices:
    time: int
    range: int
    azim: int
    elev: int
    height: int
    amp: int
    snr: int
    x: int
    y: int
    route_id: int
    circle_id: int

    @staticmethod
    def from_dict(**kwargs):
        return FeaturesColumnIndices(
            time=kwargs["time"],
            range=kwargs["range"],
            azim=kwargs["azim"],
            elev=kwargs["elev"],
            height=kwargs["height"],
            amp=kwargs["amp"],
            snr=kwargs["snr"],
            x=kwargs["x"],
            y=kwargs["y"],
            route_id=kwargs["route_id"],
            circle_id=kwargs["circle_id"],
        )


class HorizontalCoordSplitter(abc.ABC):
    """
    Split horizontal coordinate into multiple areas, and calculate area mask for
    input points.

    """

    @property
    @abc.abstractmethod
    def n_split_areas(self) -> int:
        ...

    @abc.abstractmethod
    def split(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        points_r: np.ndarray,
        points_azim: np.ndarray
    ) -> np.ndarray:
        """
        Telling which split area each point belongs to.

        Args:
            points_x: shape=(n_points,).
            points_y: shape=(n_points,).
            points_r: shape=(n_points,).
            points_azim: shape=(n_points,).

        Returns:
            Boolean mask of shape (n_points, n_split_areas). In each column,
            there is at most one `True` element, telling the point belongs to
            the corresponding split area.

        """
        ...

    @classmethod
    @abc.abstractmethod
    def generate_fully_covering_splitter(
        cls, r_max: float, area_width: float, *args, **kwargs
    ) -> "HorizontalCoordSplitter":
        """
        Generate a splitter that contains areas fully covering 0<=r<=r_max and
        -pi<=azim<=pi.

        Args:
            r_max: Maximum range of radial distance, in meters.
            area_width: Width of each area, whose definition is
                implementation-dependent.
            *args: Positional arguments passed to sub-class implementation.
            **kwargs: Keyword arguments passed to sub-class implementation.

        Returns:
            Fully covering splitter.

        """
        ...


@numba.experimental.jitclass(
    spec=[
        ("rect_widths", numba.float64[::1]),
        ("rect_lengths", numba.float64[::1]),
        ("rect_depths", numba.float64[::1]),
        ("rect_azims", numba.float64[::1]),
        ("_inverse_rot", numba.float64[:, :, ::1]),
        ("_rect_corners", numba.float64[:, :, ::1])
    ]
)
class _RectSplitterImpl:
    def __init__(
        self,
        rect_widths: npt.NDArray[float],
        rect_lengths: npt.NDArray[float],
        rect_depths: npt.NDArray[float],
        rect_azims: npt.NDArray[float]
    ):
        self.rect_widths = rect_widths
        self.rect_lengths = rect_lengths
        self.rect_depths = rect_depths
        self.rect_azims = rect_azims

        cos_azims = np.cos(self.rect_azims)
        sin_azims = np.sin(self.rect_azims)
        self._inverse_rot = np.zeros(
            (len(self.rect_azims), 2, 2),
            dtype=float
        )
        for i in numba.prange(self._inverse_rot.shape[0]):
            self._inverse_rot[i, 0, 0] = cos_azims[i]
            self._inverse_rot[i, 0, 1] = -sin_azims[i]
            self._inverse_rot[i, 1, 0] = sin_azims[i]
            self._inverse_rot[i, 1, 1] = cos_azims[i]
        # shape=(n_split_areas, 2, 2)

        self._rect_corners = self._init_rect_corners()

    def _init_rect_corners(self):
        corners = np.zeros(
            (len(self.rect_azims), 4, 2),
            dtype=float
        )
        for i in numba.prange(corners.shape[0]):
            corners[i, 0, 0] = -self.rect_widths[i] / 2
            corners[i, 0, 1] = -self.rect_depths[i]
            corners[i, 1, 0] = self.rect_widths[i] / 2
            corners[i, 1, 1] = -self.rect_depths[i]
            corners[i, 2, 0] = self.rect_widths[i] / 2
            corners[i, 2, 1] = self.rect_lengths[i] - self.rect_depths[i]
            corners[i, 3, 0] = -self.rect_widths[i] / 2
            corners[i, 3, 1] = self.rect_lengths[i] - self.rect_depths[i]
        # shape=(n_split_areas, 4 corners, 2)

        # return np.einsum(
        #     "mji,mkj->nki",
        #     self._inverse_rot,  # U^T = U^{-1}
        #     corners
        # )
        rot_corners = np.zeros((len(self.rect_azims), 4, 2), dtype=float)
        for m in numba.prange(len(self.rect_azims)):
            rot_corners[m] = corners[m] @ self._inverse_rot[m]
        return rot_corners

    def inverse_transform_on_points(
        self,
        xys: npt.NDArray[float],
        indices_of_split_areas: npt.NDArray[int]
    ):
        # xys_rot_back = np.einsum(
        #     "mij,nj->nmi", self._inverse_rot, xys
        # )  # shape=(n_points, n_split_areas, 2)

        xys_rot_back = np.zeros(
            (len(xys), len(indices_of_split_areas), 2),
            dtype=float
        )
        for n in numba.prange(xys_rot_back.shape[0]):
            for i_m in numba.prange(len(indices_of_split_areas)):
                m = indices_of_split_areas[i_m]
                # xys_rot_back[n, i_m] = self._inverse_rot[m] @ np.ascontiguousarray(xys[n])
                for i in numba.prange(self._inverse_rot.shape[1]):
                    for j in numba.prange(self._inverse_rot.shape[2]):
                        xys_rot_back[n, i_m, i] += self._inverse_rot[m, i, j] * xys[n, j]
        return xys_rot_back

    def split(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray
    ) -> np.ndarray:
        xys = np.zeros(
            (len(points_x), 2),
            dtype=float
        )
        for i in numba.prange(len(xys)):
            xys[i, 0] = points_x[i]
            xys[i, 1] = points_y[i]
        xys_rot_back = self.inverse_transform_on_points(
            xys,
            np.arange(len(self.rect_azims))
        )  # shape=(n_points, n_split_areas, 2)
        mask = (
            (-self.rect_depths <= xys_rot_back[:, :, 1]) &
            (xys_rot_back[:, :, 1] <= self.rect_lengths - self.rect_depths) &
            (np.abs(xys_rot_back[:, :, 0]) <= self.rect_widths / 2)
        )  # shape=(n_points, n_split_areas)
        return mask


class HorizontalCoordRectSplitter(HorizontalCoordSplitter):
    def __init__(
        self,
        rect_widths: Union[float, np.ndarray],
        rect_lengths: Union[float, np.ndarray],
        rect_depths: Union[float, np.ndarray],
        rect_azims: Union[float, np.ndarray]
    ):
        """
        Initialize parameters of each split rectangular area.

        Args:
            rect_widths: shape=(n_split_areas,)
            rect_lengths: shape=(n_split_areas,)
            rect_depths: shape=(n_split_areas,)
            rect_azims: shape=(n_split_areas,), each element is represented in radians
                and ascends in clockwise direction, defined as the angle between
                the radius vector and +x direction (north).

        Note:
            Each area can be obtained by following procedure:
                1. Draw a rectangle in range -W/2<=x<=W/2, 0<=y<=L;
                2. Shift it along y-axis (north) by a distance -D;
                3. Rotate it along z-axis (up) by an angle (positive for
                    counter-clockwise) -A (noted as :math:`R_z(-A)`),
            where W, L, D and A are elements in :attr:`rect_widths`,
            :attr:`rect_lengths`, :attr:`rect_depths`, :attr:`rect_azims`.

        """
        (self.rect_widths,
         self.rect_lengths,
         self.rect_depths,
         self.rect_azims) = np.broadcast_arrays(
            np.squeeze(rect_widths),
            np.squeeze(rect_lengths),
            np.squeeze(rect_depths),
            np.squeeze(rect_azims)
        )

        if np.ndim(self.rect_azims) > 1:
            raise ValueError(
                "Input parameters of rectangular areas must be 0-D or 1-D arrays."
            )
        elif np.ndim(self.rect_azims) == 0:
            self.rect_widths = self.rect_widths[None]
            self.rect_lengths = self.rect_lengths[None]
            self.rect_depths = self.rect_depths[None]
            self.rect_azims = self.rect_azims[None]

        if np.any(self.rect_depths) < 0:
            raise ValueError(
                "Some values in `rect_depths` are negative."
            )

        self._impl = _RectSplitterImpl(
            self.rect_widths.astype(float),
            self.rect_lengths.astype(float),
            self.rect_depths.astype(float),
            self.rect_azims.astype(float)
        )

    @property
    def rect_corners(self) -> np.ndarray:
        """
        Returns:
            Four corners of each rectangular area, shape=(n_split_areas, 4, 2 [(x, y)])

        """
        return self._impl._rect_corners

    @property
    def areas_inverse_transforms(self):
        """
        Return the inverse transforms of :math:`R_z(-A)` for all split areas
        (see :py:meth:`HorizontalCoordRectSplitter.__init__`).

        """
        return self._impl._inverse_rot

    @property
    def areas_transforms(self):
        """
        Return the transforms of :math:`R_z(-A)` for all split areas
        (see :py:meth:`HorizontalCoordRectSplitter.__init__`).

        """
        return np.swapaxes(self._impl._inverse_rot, -1, -2)

    def inverse_transform_on_points(
        self,
        xys,
        indices_of_split_areas: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Perform the inverse transform of :math:`R_z(-A)`
        (see :py:meth:`HorizontalCoordRectSplitter.__init__`) on input points.

        Args:
            xys: (x,y) coordinates of points to be transformed, shape=(n_points, 2)
            indices_of_split_areas: Indices of split areas inverses of whose
                transforms are to be performed on points. If being None, all
                split areas will be considered.

        Returns:
            (x, y) coordinates of transformed points, shape=(n_points,
            len(indices_of_split_areas), 2).

        """

        if indices_of_split_areas is None:
            # iterate over all split areas:
            indices_of_split_areas = np.arange(self.n_split_areas)
        elif isinstance(indices_of_split_areas, numbers.Integral):
            indices_of_split_areas = np.array([indices_of_split_areas])
        else:
            indices_of_split_areas = np.array(indices_of_split_areas)

        return self._impl.inverse_transform_on_points(
            np.ascontiguousarray(xys),
            indices_of_split_areas
        )

    @property
    def n_split_areas(self) -> int:
        return len(self.rect_azims)

    def split(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray
    ) -> npt.NDArray[bool]:
        return self._impl.split(
            points_x,
            points_y
        )

    @classmethod
    def generate_fully_covering_splitter(
        cls, r_max: float, area_width: float, area_depth: float
    ) -> "HorizontalCoordRectSplitter":
        """
        Generate a splitter that contains areas fully covering 0<=r<=r_max and
        -pi<=azim<=pi.

        Args:
            r_max: Maximum range of radial distance, in meters.
            area_width: Width of each rectangular area, in meters.
            area_depth: Depth of each rectangular area, in meters. If it
                is smaller than zero, a warning will be triggered.

        Returns:
            Fully covering splitter.

        """

        if area_depth < 0:
            warnings.warn(
                "Input splitter depth is smaller than zero, which may cause "
                "generated split areas not able to cover the region near to "
                "the origin."
            )

        rect_azim_width = 2 * np.arctan2(area_width / 2, r_max)
        n_areas = int(np.ceil(2 * np.pi / rect_azim_width))
        rect_azims = np.arange(0, n_areas) * 2 * np.pi / n_areas
        rect_azims[rect_azims > np.pi] -= 2 * np.pi

        return cls(
            rect_widths=area_width,
            rect_lengths=r_max + area_depth,
            rect_depths=area_depth,
            rect_azims=rect_azims
        )


@numba.njit(
    parallel=False,  # switch off parallel due to numba issue 6065
    fastmath=True
)
def _fan_split_impl_slow(
    points_r,
    points_azim,
    fan_widths,
    fan_lengths,
    fan_depths,
    fan_azims
):
    mask = (
        (points_r < fan_depths) |
        (
            (points_r < fan_lengths) &
            (np.abs(((points_azim - fan_azims + np.pi) % (2 * np.pi) - np.pi)) < fan_widths / 2.)
        )
    )

    return mask  # shape=(n_points, n_split_areas)


class HorizontalCoordFanSplitter(HorizontalCoordSplitter):
    def __init__(
        self,
        fan_widths: Union[float, np.ndarray],
        fan_lengths: Union[float, np.ndarray],
        fan_depths: Union[float, np.ndarray],
        fan_azims: Union[float, np.ndarray]
    ):
        """
        Initialize parameters of each split fan area.

        Args:
            fan_widths: shape=(n_split_areas,)
            fan_lengths: shape=(n_split_areas,)
            fan_depths: shape=(n_split_areas,)
            fan_azims: shape=(n_split_areas,), each element is represented in radians
                and ascends in clockwise direction, defined as the angle between
                the radius vector and +y direction (north).

        Note:
            Each area can be obtained by following procedure:
                1. Draw a fan :math:`F` in range -W/2<=azim<=W/2, 0<=r<=L;
                2. Draw a circle :math:`C` defined by r<=D;
                3. Set :math:`\\mathcal{A} = \\mathcal{F} \\cup \\mathcal{C}`
                    and rotate it along z-axis (up) by an angle (positive for
                    counter-clockwise) :math:`-A`,
            where W, L, D and A are elements in :attr:`fan_widths`,
            :attr:`fan_lengths`, :attr:`fan_depths`, :attr:`fan_azims`.

        """
        (self.fan_widths,
         self.fan_lengths,
         self.fan_depths,
         self.fan_azims) = np.broadcast_arrays(
            np.squeeze(fan_widths),
            np.squeeze(fan_lengths),
            np.squeeze(fan_depths),
            np.squeeze(fan_azims)
        )

        if np.ndim(self.fan_azims) > 1:
            raise ValueError(
                "Input parameters of fan areas must be 0-D or 1-D arrays."
            )
        elif np.ndim(self.fan_azims) == 0:
            self.fan_widths = self.fan_widths[None]
            self.fan_lengths = self.fan_lengths[None]
            self.fan_depths = self.fan_depths[None]
            self.fan_azims = self.fan_azims[None]

        if np.any((self.fan_azims < -np.pi) | (np.pi < self.fan_azims)):
            warnings.warn(
                "Some values in `fan_azims` exceed [-pi, pi]. "
                "They will be automatically wrapped into this range."
            )
            self.fan_azims = (self.fan_azims + np.pi) % (2 * np.pi) - np.pi

    @property
    def n_split_areas(self) -> int:
        return len(self.fan_azims)

    def split(
        self,
        points_r: np.ndarray,
        points_azim: np.ndarray
    ) -> np.ndarray:
        points_r = points_r[:, None]
        points_azim = points_azim[:, None]

        mask = (
            (points_r < self.fan_depths) |
            (
                (points_r < self.fan_lengths) &
                (np.abs(((points_azim - self.fan_azims + np.pi) % (2 * np.pi) - np.pi)) < self.fan_widths / 2.)
            )
        )

        return mask  # shape=(n_points, n_split_areas)

    @classmethod
    def generate_fully_covering_splitter(
        cls, r_max: float, area_width: float, area_depth: float
    ) -> "HorizontalCoordFanSplitter":
        """
        Generate a splitter that contains areas fully covering 0<=r<=r_max and
        -pi<=azim<=pi.

        Args:
            r_max: Maximum range of radial distance, in meters.
            area_width: Width of each fan area, in radians.
            area_depth: Depth of each fan area, in meters. If it
                is smaller than zero, a ValueError will be raised.

        Returns:
            Fully covering splitter.

        """
        if area_depth < 0:
            raise ValueError(
                "Input splitter depth cannot be smaller than zero."
            )

        n_areas = int(np.ceil(2 * np.pi / area_width))
        fan_azims = np.arange(0, n_areas) * 2 * np.pi / n_areas
        fan_azims[fan_azims > np.pi] -= 2 * np.pi

        return cls(
            fan_widths=area_width,
            fan_lengths=r_max,
            fan_depths=area_depth,
            fan_azims=fan_azims
        )


class HorizontalCoordSmoothFanSplitter(HorizontalCoordFanSplitter):
    @staticmethod
    def smooth_area_border(azims, r_min, r_max, azim_center, azim_width):
        def _f(x):
            return 1 / np.sqrt(1 + x ** 2)

        azims = (azims + np.pi) % (2 * np.pi) - np.pi
        azims_diff = (azims[:, None] - azim_center + np.pi) % (2 * np.pi) - np.pi
        # shape=(n_points, n_split_areas)

        f_min = _f(np.pi / azim_width)  # shape=(n_split_areas,)
        weights = (_f(azims_diff / azim_width) - f_min) / (1 - f_min)
        # shape=(n_points, n_split_areas)

        rs = r_min + (r_max - r_min) * weights

        return rs

    def split(
        self,
        points_r: np.ndarray,
        points_azim: np.ndarray
    ) -> np.ndarray:
        mask = points_r[:, None] < HorizontalCoordSmoothFanSplitter.smooth_area_border(
            points_azim,
            self.fan_depths, self.fan_lengths,
            self.fan_azims, self.fan_widths
        )
        return mask


@numba.njit(parallel=True)
def _padded_fan_split_impl_slow(
    points_r,
    points_azim,
    fan_widths,
    fan_lengths,
    pad_widths,
    fan_azims
):
    n_points = len(points_r)
    n_split_areas = len(fan_azims)

    mask = np.zeros((n_points, n_split_areas), dtype=np.bool_)

    for i_point in numba.prange(n_points):
        r = points_r[i_point]
        azim = points_azim[i_point]
        for i_area in numba.prange(n_split_areas):
            _area_width = fan_widths[i_area]
            _area_length = fan_lengths[i_area]
            _pad_width = pad_widths[i_area]
            _area_azim = fan_azims[i_area]

            _delta_azim = np.abs(
                (azim - _area_azim + np.pi) % (2 * np.pi) - np.pi
            )

            _flag = False
            if (_delta_azim - _area_width / 2.) > np.pi / 2:
                if r <= _pad_width:
                    _flag = True
            elif _delta_azim < _area_width / 2.:
                _flag = True
            elif r <= _pad_width / np.sin(_delta_azim - _area_width / 2.):
                _flag = True

            mask[i_point, i_area] = _flag

    return mask


class HorizontalCoordPaddedFanSplitter(HorizontalCoordFanSplitter):
    def __init__(
        self,
        fan_widths: Union[float, np.ndarray],
        fan_lengths: Union[float, np.ndarray],
        pad_widths: Union[float, np.ndarray],
        fan_azims: Union[float, np.ndarray],
        r1: Union[float, np.ndarray] = -1,
        r2: Union[float, np.ndarray] = -1
    ):
        """
        Initialize parameters of each split fan area.

        Args:
            fan_widths: shape=(n_split_areas,), each element is represented in radians
                in range [0, 2pi].
            fan_lengths: shape=(n_split_areas,)
            pad_widths: shape=(n_split_areas,)
            fan_azims: shape=(n_split_areas,), each element is represented in radians
                and ascends in clockwise direction, defined as the angle between
                the radius vector and +y direction (north).
            r1: shape=(n_split_areas,), radii of near regions.
            r2: shape=(n_split_areas,), r of remote borders of padding decay transition
                regions.

        Note:
            Each area can be obtained by following procedure:
                1. Draw a fan :math:`F` in range -W/2<=azim<=W/2, 0<=r<=L;
                2. Add area including all points whose distances to the border of
                    :math:`F` do not exceed D into :math:`F`;
                3. Exclude area of padding decay transition region;
                4. Rotate :math:`F` along z-axis (up) by an angle (positive for
                    counter-clockwise) :math:`-A` (noted as :math:`R_z(-A)`),
            where W, L, D and A are elements in :attr:`fan_widths`,
            :attr:`fan_lengths`, :attr:`pad_widths`, :attr:`fan_azims`.

        """
        (self.fan_widths,
         self.fan_lengths,
         self.pad_widths,
         self.fan_azims,
         self.r1,
         self.r2) = np.broadcast_arrays(
            np.squeeze(fan_widths),
            np.squeeze(fan_lengths),
            np.squeeze(pad_widths),
            np.squeeze(fan_azims),
            np.squeeze(r1),
            np.squeeze(r2)
        )

        if np.ndim(self.fan_azims) > 1:
            raise ValueError(
                "Input parameters of fan areas must be 0-D or 1-D arrays."
            )
        elif np.ndim(self.fan_azims) == 0:
            self.fan_widths = self.fan_widths[None]
            self.fan_lengths = self.fan_lengths[None]
            self.pad_widths = self.pad_widths[None]
            self.fan_azims = self.fan_azims[None]
            self.r1 = self.r1[None]
            self.r2 = self.r2[None]

        if np.any((self.fan_azims < -np.pi) | (np.pi < self.fan_azims)):
            warnings.warn(
                "Some values in `fan_azims` exceed [-pi, pi]. "
                "They will be automatically wrapped into this range."
            )
            self.fan_azims = (self.fan_azims + np.pi) % (2 * np.pi) - np.pi

        if np.any((self.fan_widths < 0) | (2 * np.pi < self.fan_widths)):
            raise ValueError(
                f"Some values in `fan_azims` exceed [0, 2pi]: {self.fan_widths}"
            )

        if np.any(self.fan_lengths < self.pad_widths):
            raise ValueError(
                f"For some areas the pad widths exceed their fan lengths."
            )

        # x coordinate of intersection point between padded border line
        # and +x axis:
        self.r1 = np.where(self.r1 < 0, self.pad_widths, self.r1)
        self.r2 = np.where(self.r2 < 0, self.fan_lengths, self.r2)
        if np.any((self.r2 < self.r1) | (self.fan_lengths < self.r2)):
            raise ValueError(
                f"For some areas the r2 values do not lie in "
                f"[r1, fan_length]: "
                f"lower bounds = {self.r1}, "
                f"upper bounds = {self.fan_lengths}"
            )
        if np.any(self.pad_widths < self.r1):
            raise ValueError(
                f"For some areas the r1 values exceed their padding widths."
            )

        circumrect_corners = self.circumrect_corners
        self._circumrect_splitter: HorizontalCoordRectSplitter = HorizontalCoordRectSplitter(
            rect_widths=circumrect_corners[:, 1] - circumrect_corners[:, 0],
            rect_lengths=circumrect_corners[:, 3] - circumrect_corners[:, 2],
            rect_depths=-circumrect_corners[:, 2],
            rect_azims=self.fan_azims
        )

        self._delta_theta1 = np.arcsin(self.pad_widths / self.r2)
        self._delta_theta2 = self.fan_widths / 2. + self._delta_theta1 \
                             - np.arcsin(self.r1 / self.r2)
        self._half_fan_widths = self.fan_widths / 2.

    def split(
        self,
        points_r: np.ndarray,
        points_azim: np.ndarray
    ) -> np.ndarray:
        points_r = points_r[:, None]
        points_azim = points_azim[:, None]
        delta_azims = np.abs(
            (points_azim - self.fan_azims + np.pi) % (2 * np.pi) - np.pi
        )

        rs_on_padded_border = self.pad_widths / np.sin(delta_azims - self._half_fan_widths)
        mask = np.where(
            self.fan_widths < np.pi,
            np.where(
                self._delta_theta2 + np.pi / 2 < delta_azims,
                # Area 4
                points_r <= self.r1,
                np.where(
                    self._half_fan_widths + self._delta_theta1 < delta_azims,
                    # Area 3
                    points_r <= self.r1 / np.sin(delta_azims - self._delta_theta2),
                    np.where(
                        self._half_fan_widths < delta_azims,
                        # Area 2
                        points_r <= np.min(
                            [np.broadcast_to(self.fan_lengths, rs_on_padded_border.shape),
                             rs_on_padded_border],
                            axis=0
                        ),
                        # Area 1
                        points_r <= self.fan_lengths
                    )
                )
            ),
            np.where(
                (delta_azims > self._half_fan_widths),
                points_r <= rs_on_padded_border,
                True
            )
        )

        return mask

    @property
    def areas_inverse_transforms(self):
        """
        Return the inverse transforms of :math:`R_z(-A)` for all split areas
        (see :py:meth:`HorizontalCoordPaddedFanSplitter.__init__`).

        """
        return self._circumrect_splitter.areas_inverse_transforms

    @property
    def areas_transforms(self):
        """
        Return the transforms of :math:`R_z(-A)` for all split areas
        (see :py:meth:`HorizontalCoordPaddedFanSplitter.__init__`).

        """
        return self._circumrect_splitter.areas_transforms

    def inverse_transform_on_points(
        self,
        xys,
        indices_of_split_areas: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Perform the inverse transform of :math:`R_z(-A)`
        (see :py:meth:`HorizontalCoordPaddedFanSplitter.__init__`) on input points.

        Args:
            xys: (x,y) coordinates of points to be transformed, shape=(n_points, 2)
            indices_of_split_areas: Indices of split areas inverses of whose
                transforms are to be performed on points. If being None, all
                split areas will be considered.

        Returns:
            (x, y) coordinates of transformed points, shape=(n_points,
            len(indices_of_split_areas), 2).

        """
        return self._circumrect_splitter.inverse_transform_on_points(
            xys, indices_of_split_areas
        )

    @classmethod
    def generate_fully_covering_splitter(
        cls, r_max: float, area_width: float, pad_width: float, r1: float = -1, r2: float = -1
    ) -> "HorizontalCoordPaddedFanSplitter":
        """
        Generate a splitter that contains areas fully covering 0<=r<=r_max and
        -pi<=azim<=pi.

        Args:
            r_max: Maximum range of radial distance, in meters.
            area_width: Width of each fan area, in radians.
            pad_width: Padding width of each fan area, in meters. If it
                is smaller than zero, a ValueError will be raised.
            r1: Radius of near region.
            r2: r of remote border of padding decay transition region.

        Returns:
            Fully covering splitter.

        """
        if pad_width < 0:
            raise ValueError(
                "Input splitter depth cannot be smaller than zero."
            )

        if area_width == 0:
            raise ValueError(
                "Area width cannot be zero in automatic splitter generating."
            )

        n_areas = int(np.ceil(2 * np.pi / area_width))
        fan_azims = np.arange(0, n_areas) * 2 * np.pi / n_areas
        fan_azims[fan_azims > np.pi] -= 2 * np.pi

        return HorizontalCoordPaddedFanSplitter(
            fan_widths=area_width,
            fan_lengths=r_max,
            pad_widths=pad_width,
            fan_azims=fan_azims,
            r1=r1,
            r2=r2
        )

    @functools.cached_property
    def circumrect_corners(self) -> np.ndarray:
        """

        Returns: The (x_min, x_max, y_min, y_max) corners of rectangles
            circumscribing padded fan areas (rotated back to orienting to the
            north), shape=(n_split_areas, 4).

        """
        corners = np.zeros((self.n_split_areas, 4))

        reflex_angle_area_mask = self.fan_widths > np.pi

        cos_half_width = np.cos(self.fan_widths / 2)
        sin_half_width = np.sin(self.fan_widths / 2)

        _d_term = self.pad_widths * cos_half_width
        _sqrt_term = np.sqrt(self.fan_lengths ** 2 - self.pad_widths ** 2) * sin_half_width

        corners[reflex_angle_area_mask, 1] = +self.fan_lengths[reflex_angle_area_mask]
        corners[reflex_angle_area_mask, 2] = (
            _d_term[reflex_angle_area_mask] - _sqrt_term[reflex_angle_area_mask]
        )

        corners[~reflex_angle_area_mask, 1] = (
            _d_term[~reflex_angle_area_mask] + _sqrt_term[~reflex_angle_area_mask]
        )
        corners[~reflex_angle_area_mask, 2] = -self.pad_widths[~reflex_angle_area_mask]

        corners[:, 0] = -corners[:, 1]
        corners[:, 3] = +self.fan_lengths

        return corners
