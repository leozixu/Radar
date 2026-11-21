from __future__ import annotations

import numba
import numpy as np
import torch
from typing import Dict, Optional, Sequence, Union

from .. import radar_definition as rd
from ..data import data_preprocess


@numba.njit(cache=True)
def split_area_samples(
    data,
    n_split_areas,
    sample_len,
    radial_split_borders: Optional[tuple[
        tuple[float, float],
        ...
    ]] = ((-1, -1),),
):
    """
    Args:
        data: Points data with area masks appended, shape=(n_points, n_features +
            n_split_areas).
        n_split_areas: Number of split areas.
        sample_len: Length of each generated sample.
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

            - Real numbers of points in areas, shape=(n_nonempty_split_areas,),
              whose values MAY EXCEED :attr:`sample_len`.

    """
    indices_sorted_by_time = np.argsort(data[:, rd.PointFeatureIndex.RADAR_TIME.value])
    data = data[indices_sorted_by_time]

    n_point_features = data.shape[-1]  # NOTE: area masks are included

    area_points_counts = np.sum(data[:, -n_split_areas:], axis=0).astype(np.int64)
    # shape=(n_split_areas,)
    nonempty_area_mask = area_points_counts > 0
    next_indices_to_write = np.where(
        area_points_counts <= sample_len,
        0,
        sample_len - area_points_counts
    )

    samples = np.full((n_split_areas, sample_len, n_point_features),
                      fill_value=np.nan, dtype=data.dtype)
    point_indices = np.full((n_split_areas, sample_len), fill_value=-1, dtype=int)
    for i_point in range(data.shape[0]):
        for i_area in range(n_split_areas):
            if not data[i_point, -n_split_areas + i_area]:  # this point is not in the area
                continue

            next_index_to_write = next_indices_to_write[i_area]
            if next_index_to_write >= 0:  # following points of this area can be kept
                samples[i_area, next_index_to_write] = data[i_point]
                point_indices[i_area, next_index_to_write] = indices_sorted_by_time[i_point]

            next_indices_to_write[i_area] += 1

    area_ids = np.arange(n_split_areas)[nonempty_area_mask]
    samples = samples[nonempty_area_mask]
    point_indices = point_indices[nonempty_area_mask]
    n_points_in_areas = area_points_counts[nonempty_area_mask]

    if radial_split_borders == ((-1, -1),) or radial_split_borders is None:
        return samples, area_ids, point_indices, n_points_in_areas

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

            if _point_count_in_region < sample_len:
                _sample_in_region = np.concatenate(
                    (_sample_in_region,
                     np.full((sample_len - _point_count_in_region, n_point_features),
                             fill_value=np.nan,
                             dtype=_sample_in_region.dtype)),
                    axis=0
                )
                _point_indices_in_region = np.concatenate(
                    (point_indices[i_sample][in_region_mask],
                     np.full((sample_len - _point_count_in_region,),
                             fill_value=-1,
                             dtype=point_indices.dtype))
                )
            else:
                if r_max < 100e3:  # for near regions, drop points with smallest SNRs
                    _indices_sorted_by_snr = np.argsort(_sample_in_region[:, rd.PointFeatureIndex.SNR.value])
                    _sample_in_region = _sample_in_region[_indices_sorted_by_snr[-sample_len:]]
                    _point_indices_in_region = _point_indices_in_region[-sample_len:]
                else:  # for remote regions, drop earliest points
                    _sample_in_region = _sample_in_region[-sample_len:]
                    _point_indices_in_region = _point_indices_in_region[-sample_len:]

            samples_r_split[n_sample_r_split] = _sample_in_region
            area_ids_r_split[n_sample_r_split] = area_ids[i_sample]
            point_indices_r_split[n_sample_r_split] = _point_indices_in_region
            n_points_in_areas_r_split[n_sample_r_split] = n_points_in_areas[i_sample]
            n_sample_r_split += 1

    return (samples_r_split[:n_sample_r_split],
            area_ids_r_split[:n_sample_r_split],
            point_indices_r_split[:n_sample_r_split],
            n_points_in_areas_r_split[:n_sample_r_split])


@numba.njit(parallel=True, cache=True)
def _normalize_samples_txys_impl(
    points_in_areas: np.ndarray,
    area_borders: np.ndarray,
    time_normalization_factors: np.ndarray,
    areas_inverse_transforms: np.ndarray,
    t_idx: int,
    x_idx: int,
    y_idx: int
):
    data_normalized = points_in_areas.copy()

    min_x, max_x, min_y, max_y = area_borders.T

    x_norm_factor = max_x - min_x  # shape=(n_areas,)
    y_norm_factor = max_y - min_y  # shape=(n_areas,)

    n_areas = data_normalized.shape[0]

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
    az_min: Union[float, np.ndarray] = -np.pi,
    az_range: Union[float, np.ndarray] = 2 * np.pi,
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
    az_min = np.broadcast_to(az_min, (len(points_in_areas),))
    az_range = np.broadcast_to(az_range, (len(points_in_areas),))
    data_normalized = _normalize_samples_txys_impl(
        points_in_areas,
        area_borders,
        time_normalization_factor,
        area_inverse_transforms,
        rd.PointFeatureIndex.RADAR_TIME.value,
        rd.PointFeatureIndex.X.value,
        rd.PointFeatureIndex.Y.value
    )

    data_normalized[:, :, rd.PointFeatureIndex.R.value] /= r_normalization_factor[:, None]
    data_normalized[:, :, rd.PointFeatureIndex.AZ.value] = \
        ((data_normalized[:, :, rd.PointFeatureIndex.AZ.value] - az_min[:, None])
         / az_range[:, None]) % 1.
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


def az_enc_to_rad(
    azimuth: np.ndarray,
    *,
    az_range: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Convert azimuth from the raw encoding unit to radians, where the full
    circle (`az_range` raw units) corresponds to `2 * pi` radians.

    """
    return azimuth * (2.0 * np.pi / az_range)


def denormalize_samples(
    points: np.ndarray,
    *,
    az_range: Union[float, np.ndarray],
    elev_range: Union[float, np.ndarray],
    r_nf: Union[float, np.ndarray],
    amp_nf: Union[float, np.ndarray],
    snr_nf: Union[float, np.ndarray],
    h_nf: Union[float, np.ndarray],
    half_speed_nf: Union[float, np.ndarray, None]
) -> np.ndarray:
    """
    Denormalize generated samples back to the raw (pre-normalization) units.

    Range, amplitude, SNR, height and radial speed are scaled by the
    corresponding normalization factors, while azimuth and elevation are
    restored to values starting from 0 (i.e. `[0, range)`), without any
    degree conversion or wrapping. Note that this is not a strict inverse of
    `normalize_samples_v2`, which subtracts `az_min`/`elev_min`.

    Args:
        points: shape=(n_points, n_features), modified in place.
        az_range: azimuth normalization range (e.g. the full circle
            corresponds to `az_range` raw units).
        elev_range: elevation normalization range.
        r_nf, amp_nf, snr_nf, h_nf: normalization factors for range, amplitude,
            SNR and height respectively.
        half_speed_nf: half of the radial-speed normalization factor.

    Returns:
        The denormalized `points` array (same object as the input).

    """
    points[:, rd.PointFeatureIndex.R] *= r_nf
    points[:, rd.PointFeatureIndex.AMP] *= amp_nf
    points[:, rd.PointFeatureIndex.SNR] *= snr_nf
    points[:, rd.PointFeatureIndex.R_FILT] *= r_nf
    points[:, rd.PointFeatureIndex.X] *= r_nf
    points[:, rd.PointFeatureIndex.Y] *= r_nf
    points[:, rd.PointFeatureIndex.X_FILT] *= r_nf
    points[:, rd.PointFeatureIndex.Y_FILT] *= r_nf
    points[:, rd.PointFeatureIndex.HEIGHT] *= h_nf

    if half_speed_nf is not None:
        points[:, rd.PointFeatureIndex.RADIAL_SPEED] = (
            points[:, rd.PointFeatureIndex.RADIAL_SPEED] * 2 * half_speed_nf
            - half_speed_nf
        )

    points[:, rd.PointFeatureIndex.AZ] = (
        points[:, rd.PointFeatureIndex.AZ] * az_range
    )
    points[:, rd.PointFeatureIndex.AZ_FILT] = (
        points[:, rd.PointFeatureIndex.AZ_FILT] * az_range
    )
    points[:, rd.PointFeatureIndex.EL] = (
        points[:, rd.PointFeatureIndex.EL] * elev_range
    )
    points[:, rd.PointFeatureIndex.EL_FILT] = (
        points[:, rd.PointFeatureIndex.EL_FILT] * elev_range
    )

    return points


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
