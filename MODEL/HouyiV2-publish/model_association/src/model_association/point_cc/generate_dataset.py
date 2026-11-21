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

import functools
import json
import logging
import os
import pprint
import re
import traceback
import zlib
from pathlib import Path
from typing import Annotated, Callable, Optional, Union, Tuple

import click
import h5py
import numba
import numpy as np
import pandas as pd
import tqdm
import typer
from matplotlib import pyplot as plt
from model_association import radar_definition as rd
from model_association.point_cc.config import DataConfig, CollectedDataConfig, DATA_CONFIG_FILENAME, \
    COLLECTED_DATA_CONFIG_FILENAME, DataConfigMaxValue, DataConfigNormalizationFactor
from model_association.point_cc.data import (data_preprocess,
                                             split_area_samples,
                                             split_intervals,
                                             sample_points_filter,
                                             clip_samples)
from model_association.route_point_data_processor import \
    HorizontalCoordPaddedFanSplitter
from model_association.utils import (create_dir, dict_to_numba,
                                     parse_nested_float_tuple, setup_logger)

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

GLOBAL_MIN_TRACK_LEN = 4

# The columns recording the coords of a route's NEXT point, in the same order
# as the rows returned by `compute_route_next_point_coords`, i.e. (t, x, y, r).
ROUTE_NEXT_POINT_COLS = [
    rd.PointFeatureIndex.ROUTE_NEXT_POINT_T.value,
    rd.PointFeatureIndex.ROUTE_NEXT_POINT_X.value,
    rd.PointFeatureIndex.ROUTE_NEXT_POINT_Y.value,
    rd.PointFeatureIndex.ROUTE_NEXT_POINT_R.value,
]


def check_sample_validity(
    _sample: np.ndarray,
    _area_id: int,
    _real_n_points: int,
    sample_len: int,
    n_valid_points: int,
    n_target_points: int,
    min_sample_time_range: float,
    max_sample_time_range: float,
    TIME_COL_IDX: int,
    TRACK_ID_COL_IDX: int
) -> tuple[bool, str]:
    # 检查目标点数量
    if n_valid_points < 15 or n_target_points / n_valid_points < 0.01:
        return False, "TOO_FEW_TARGETS"

    # 提取公共逻辑：检查某时间点前后 10 秒的数据
    def check_circle_data(ref_time: float, is_start: bool) -> str:
        # 第一圈：ref_time 到 ref_time + 10
        # 最后一圈：ref_time - 10 到 ref_time
        if is_start:
            start_t, end_t = ref_time, ref_time + 10
        else:
            start_t, end_t = ref_time - 10, ref_time

        valid_mask = (
            (_sample[:, TIME_COL_IDX] >= start_t) &
            (_sample[:, TIME_COL_IDX] <= end_t) &
            (_sample[:, TRACK_ID_COL_IDX] >= 0)
        )
        data_choose = _sample[valid_mask]

        if len(data_choose) == 0:
            return "START_DATA_INSUFFICIENT" if is_start else "END_DATA_INSUFFICIENT"

        data_label = data_choose[data_choose[:, TRACK_ID_COL_IDX] > 0]

        len_label = len(data_label)
        if len_label <= 1:
            return "START_NO_LABEL" if is_start else "END_NO_LABEL"
        if len_label < 0.01 * len(data_choose):
            return "START_LABEL_RATIO_LOW" if is_start else "END_LABEL_RATIO_LOW"
        return ""

    # 根据 real_n_points 计算时间阈值
    if _real_n_points < sample_len:
        t_end_sample: float = _sample[_real_n_points - 1, TIME_COL_IDX]
        t_start_sample: float = t_end_sample - max_sample_time_range * 0.9  # 修正：从 t_end 往前推
    else:
        t_start_sample: float = _sample[0, TIME_COL_IDX]
        t_end_sample: float = _sample[sample_len - 1, TIME_COL_IDX]

    # 检查第一圈
    start_check = check_circle_data(t_start_sample, is_start=True)
    if start_check:
        return False, start_check

    # 检查时间范围
    # NOTE: `_sample` has already been converted to SI units (seconds) by
    # `data_preprocess`, so `t_end_sample - t_start_sample` is in seconds and
    # `min_sample_time_range` is compared directly (no `features_to_si_coeffs`
    # conversion of the RADAR_TIME column here).
    if t_end_sample - t_start_sample < min_sample_time_range:
        return False, "TIME_RANGE_TOO_SHORT"

    # 检查最后一圈
    end_check = check_circle_data(t_end_sample, is_start=False)
    if end_check:
        return False, end_check

    return True, ""


def compute_route_next_point_coords(
    data_mat: np.ndarray,
    dt_between_circles: float
) -> np.ndarray:
    """
    For each point, compute the (t, x, y, r) coordinates of its next point in
    the same track.

    Args:
        data_mat: Points data sorted by ascending time.
        dt_between_circles: Maximum time gap (in seconds) between two consecutive
            points of the same track. See ``DataConfig.dt_between_circles``.

    Returns:
        An array of shape ``(len(data_mat), 4)`` whose entry ``i`` is the
        ``(t, x, y, r)`` of the next same-track point of point ``i`` (the point
        immediately following ``i`` in time with the same ``TRACK_ID_MANUAL``),
        or an all-NaN row if there is no such point or the time gap exceeds
        ``dt_between_circles``. Points with ``TRACK_ID_MANUAL <= 0`` always get
        an all-NaN row.
    """
    n_points = len(data_mat)
    next_coords = np.full((n_points, 4), np.nan, dtype=data_mat.dtype)

    track_id = data_mat[:, rd.PointFeatureIndex.TRACK_ID_MANUAL.value]
    time = data_mat[:, rd.PointFeatureIndex.RADAR_TIME.value]
    x = data_mat[:, rd.PointFeatureIndex.X.value]
    y = data_mat[:, rd.PointFeatureIndex.Y.value]
    r = data_mat[:, rd.PointFeatureIndex.R.value]

    # Group point indices by track id (only real tracks, i.e. TRACK_ID_MANUAL > 0).
    track_point_indices: dict = {}
    for i_point in range(n_points):
        tid = track_id[i_point]
        if tid > 0:
            track_point_indices.setdefault(tid, []).append(i_point)

    for indices in track_point_indices.values():
        for i_this, i_next in zip(indices, indices[1:]):
            if time[i_next] - time[i_this] <= dt_between_circles:
                next_coords[i_this] = (time[i_next], x[i_next], y[i_next], r[i_next])

    return next_coords


def recalculate_route_point_labels(
    sample: np.ndarray,  # shape=(n_points, n_features [+ n_area_masks])
    reg_the_last_but_one_point: bool = False,
    update_next: bool = True
) -> np.ndarray:
    """
    (Re)calculate the route-related label columns of a time-sorted sample.

    The following columns are recalculated in place on all non-padded points
    (padded points are kept all-NaN):

      - ``ROUTE_START/END_POINT_IDX``: index of the first/last point of the
        track that the point belongs to. Points which do not belong to a true
        target (``TRACK_ID_MANUAL <= 0``) label themselves;
      - ``ROUTE_LOC1_POINT_IDX`` (only if ``reg_the_last_but_one_point``):
        index of the second-to-last point of the track;
      - ``ROUTE_NEXT_POINT_{T,X,Y,R}`` (only if ``update_next``): the *own* next
        point of the track's END point (i.e. the point immediately after the
        END point in the original data, which has been recorded on the END
        point by :func:`compute_route_next_point_coords`) broadcast to all
        points of the track. NaN if the END point has no own next point within
        ``dt_between_circles``. The per-point own next coords are left
        untouched when ``update_next`` is false;
      - ``FOCUSING_POINT_FLAG``: set on the start & end points of the track
        (and on the second-to-last point if ``reg_the_last_but_one_point``),
        and reset on all other non-padded points.

    This function should be called on the final sample (after filtering &
    clipping) since all these point-removing operations make previously
    calculated positional indices invalid. When it is only used to seed the
    route labels (e.g. so that ``sample_points_filter`` can identify the last
    point of each track), ``update_next`` should be set to false to preserve
    the per-point own next coords for a later recalculation.
    """
    track_id_column = sample[:, rd.PointFeatureIndex.TRACK_ID_MANUAL.value]
    valid_points_mask = ~np.isnan(track_id_column)
    point_indices = np.arange(len(sample))

    # Initialize the labels of each valid point to be itself, and clear the
    # focusing flags.
    sample[valid_points_mask, rd.PointFeatureIndex.ROUTE_START_POINT_IDX.value] = \
        point_indices[valid_points_mask]
    sample[valid_points_mask, rd.PointFeatureIndex.ROUTE_END_POINT_IDX.value] = \
        point_indices[valid_points_mask]
    sample[valid_points_mask, rd.PointFeatureIndex.FOCUSING_POINT_FLAG.value] = 0
    if reg_the_last_but_one_point:
        sample[valid_points_mask, rd.PointFeatureIndex.ROUTE_LOC1_POINT_IDX.value] = \
            point_indices[valid_points_mask]

    if update_next:
        # Points not belonging to any true target have no next point.
        _non_target_rows = np.flatnonzero(
            valid_points_mask & (track_id_column <= 0))
        if _non_target_rows.size:
            sample[np.ix_(_non_target_rows, ROUTE_NEXT_POINT_COLS)] = np.nan

    for tid in np.unique(track_id_column[valid_points_mask]):
        if tid <= 0:
            continue

        track_points_indices = np.where(track_id_column == tid)[0]

        sample[track_points_indices, rd.PointFeatureIndex.ROUTE_START_POINT_IDX.value] = \
            track_points_indices[0]
        sample[track_points_indices, rd.PointFeatureIndex.ROUTE_END_POINT_IDX.value] = \
            track_points_indices[-1]
        if update_next:
            # The END point's own next point (recorded in the ROUTE_NEXT_POINT_*
            # columns of it) is the NEXT point of the whole route.
            sample[np.ix_(track_points_indices, ROUTE_NEXT_POINT_COLS)] = \
                sample[np.ix_([track_points_indices[-1]], ROUTE_NEXT_POINT_COLS)]
        if reg_the_last_but_one_point:
            sample[track_points_indices, rd.PointFeatureIndex.ROUTE_LOC1_POINT_IDX.value] = \
                track_points_indices[-2] if len(track_points_indices) >= 2 else track_points_indices[-1]

        # Flags for circle loss
        sample[track_points_indices[0], rd.PointFeatureIndex.FOCUSING_POINT_FLAG.value] = 1
        sample[track_points_indices[-1], rd.PointFeatureIndex.FOCUSING_POINT_FLAG.value] = 1
        if reg_the_last_but_one_point and len(track_points_indices) >= 2:
            sample[track_points_indices[-2], rd.PointFeatureIndex.FOCUSING_POINT_FLAG.value] = 1

    return sample


def generate_samples(
    original_data_files,
    max_sample_time_range: float,
    min_sample_time_range: float,
    max_values: DataConfigMaxValue,
    time_step: float,
    points_splitter: HorizontalCoordPaddedFanSplitter,
    n_features: int,
    radar_type_id: int,
    fn_on_points_batch_with_area_mask: Optional[Callable[[np.ndarray], int]],
    callback_on_file_processed: Optional[Callable[[], int]],
    logger: logging.Logger,
    only_primary_and_fused_points: bool = True,
    remove_uncertain_points: bool = True,
    dt_between_circles: float = np.inf
):
    TIME_COL_IDX = rd.PointFeatureIndex.RADAR_TIME.value

    features_to_si_coeffs = dict_to_numba(
        rd.POINT_FEATURES_TO_SI_UNIT_COEFF_MAPPINGS,
        key_type=numba.int64,
        value_type=numba.float64
    )

    file_list = [f for f in original_data_files if f.is_file()]
    _width_warned = {"pad": False, "trunc": False}
    with tqdm.tqdm(file_list) as progressbar:
        for file in progressbar:
            logger.debug(f"Processing {file}...")
            progressbar.set_postfix_str(file.name)

            try:
                # Detect whether the first line is a header row. This is necessary because a
                # data file may have a header whose column count differs from that of the data rows
                # (e.g. a 64-column header over data rows with extra derived columns). Reading such a
                # file with an auto-detected header (`pd.read_csv` with default header inference) makes
                # pandas silently promote the surplus leading fields into a MultiIndex (corrupting
                # `data_mat` without raising an error), so we strip the header explicitly and let the
                # `n_features`-based trimming/padding below unify the number of columns.
                #
                # `np.loadtxt` is preferred: it is measurably faster than `pd.read_csv` for these
                # purely-numeric files (~1.3-1.8x in local benchmarks), and it also returns a plain
                # ndarray so no MultiIndex is ever produced.
                _has_header = False
                try:
                    with open(file) as _fh:
                        _first = _fh.readline().rstrip("\r\n")
                    try:
                        float(_first.split(",", 1)[0])
                    except ValueError:
                        _has_header = True

                    data_mat = np.loadtxt(
                        file, delimiter=',', skiprows=1 if _has_header else 0
                    )
                except Exception as e:
                    logger.warning(
                        "Cannot load file %s using `np.loadtxt` (exception: %s). Trying `pd.read_csv`.",
                        file, e
                    )
                    data_mat = pd.read_csv(
                        file, delimiter=',', header=None,
                        skiprows=1 if _has_header else 0
                    ).to_numpy(dtype=float)
            except Exception as e:
                # Give a clearer diagnostic than a bare traceback: inconsistent column
                # counts across the data rows (a non-rectangular file, i.e. different
                # rows have different field counts) are the usual cause. Both `np.loadtxt`
                # and `pd.read_csv` require every data row to share the same field count, so
                # we report the actual per-row field counts found.
                try:
                    _counts: dict = {}
                    with open(file) as _dh:
                        for _di, _line in enumerate(_dh):
                            if _di > 20:
                                break
                            _n = _line.count(",") + 1
                            _counts[_n] = _counts.get(_n, 0) + 1
                    _count_desc = ", ".join(
                        f"{n} fields x{cnt} lines" for n, cnt in _counts.items()
                    )
                    logger.error(
                        "Failed to load file %s (n_features=%s). Skip it. The "
                        "data rows have inconsistent column counts (%s), while both "
                        "`np.loadtxt` and `pd.read_csv` need all data rows to share "
                        "one field count (>= n_features). Detail: %s",
                        file, n_features, _count_desc, traceback.format_exc()
                    )
                except Exception:
                    logger.error(
                        "Failed to load file %s. Skip it. %s",
                        file, traceback.format_exc()
                    )
                continue

            if n_features > data_mat.shape[-1]:
                if not _width_warned["pad"]:
                    _width_warned["pad"] = True
                    logger.warning(
                        "Input file %s has %d columns (< --n-features %d). The "
                        "missing columns will be zero-padded to the configured "
                        "width; this typically indicates an outdated (narrower) "
                        "input format mixed in with the current one.",
                        file, data_mat.shape[-1], n_features
                    )
                data_mat = np.concatenate(
                    [data_mat,
                     np.zeros((data_mat.shape[0], n_features - data_mat.shape[1]), dtype=data_mat.dtype)],
                    axis=-1
                )
            elif 0 < n_features < data_mat.shape[-1]:
                if not _width_warned["trunc"]:
                    _width_warned["trunc"] = True
                    logger.warning(
                        "Input file %s has %d columns (> --n-features %d). Extra "
                        "columns will be discarded. Increase --n-features if they "
                        "are needed.",
                        file, data_mat.shape[-1], n_features
                    )
                data_mat = data_mat[:, :n_features]
            else:
                pass

            # TODO: [EXPERIMENTAL] Drop points in the near region
            # data_mat = data_mat[
            #     data_mat[:, rd.PointFeatureIndex.R.value]
            #     * features_to_si_coeffs.get(rd.PointFeatureIndex.R.value)
            #     > 100e3
            # ]

            # ########## data preprocess (including SI-unit normalization) ##########
            data_mat[:, rd.PointFeatureIndex.FOCUSING_POINT_FLAG] = 0
            n_points_original = len(data_mat)
            value_bounds = dict_to_numba(
                {
                    rd.PointFeatureIndex.R.value: (0, max_values.r),
                    rd.PointFeatureIndex.AMP.value: (0, max_values.amp),
                    rd.PointFeatureIndex.SNR.value: (0, max_values.snr),
                    rd.PointFeatureIndex.HEIGHT.value: (0, max_values.height),  # TODO: min value?
                    rd.PointFeatureIndex.EL.value: (max_values.elev_bounds_in_rads.min,
                                                    max_values.elev_bounds_in_rads.max),
                    rd.PointFeatureIndex.RADIAL_SPEED: (-max_values.speed, max_values.speed)
                },
                key_type=numba.int64,
                value_type=numba.types.UniTuple(numba.float64, 2)
            )
            data_mat, filter_stat, out_of_bounds_stat = data_preprocess(
                data_mat,
                value_bounds=value_bounds,
                out_of_bounds_filters=dict_to_numba(
                    {rd.PointFeatureIndex.R.value: True,
                     rd.PointFeatureIndex.EL.value: False},
                    key_type=numba.int64,
                    value_type=numba.bool_
                ),
                to_si_unit_coeff_mapping=features_to_si_coeffs,
                only_primary_and_fused_points=only_primary_and_fused_points,
                nan_filter=True,
                inf_filter=True,
                only_certain_points_filter=remove_uncertain_points,
                calculate_xy=True
            )
            n_points_kept = len(data_mat)
            detailed_stat = {
                "filtered": filter_stat,
                "out_of_bounds": {rd.PointFeatureIndex(idx).name: v
                                  for idx, v in out_of_bounds_stat.items()}
            }
            logger.info(f"{n_points_original - n_points_kept} points are "
                        f"filtered out of original {n_points_original} points. "
                        f"Detailed stat: {pprint.pformat(detailed_stat)}")

            # Sort points by time
            data_mat = data_mat[np.argsort(data_mat[:, TIME_COL_IDX])]

            # Pre-compute (t, x, y) of the next same-track point of each point.
            # This has to be done on the whole file (not per batch), because the
            # next point of a track may lie beyond the current time window.
            next_point_coords = compute_route_next_point_coords(
                data_mat, dt_between_circles
            )

            # Split points into different time intervals
            _points_batch_ranges = split_intervals(
                data_mat[:, TIME_COL_IDX],
                max_sample_time_range,
                time_step
            )

            logger.debug(
                "Points are divided into %d batches for sample generating.",
                len(_points_batch_ranges)
            )

            for _i_point_batch, (_start_point_idx, _end_point_idx) in enumerate(_points_batch_ranges):
                if _start_point_idx == _end_point_idx:
                    logger.debug(
                        "Skip points batch %d since it is empty",
                        _i_point_batch
                    )
                    continue  # This time interval is empty

                _points_batch = data_mat[_start_point_idx:_end_point_idx].copy()

                _points_batch_times = _points_batch[:, TIME_COL_IDX]
                _batch_time_range = _points_batch_times[-1] - _points_batch_times[0]

                assert _batch_time_range <= max_sample_time_range + 1e-3
                if _batch_time_range < min_sample_time_range:
                    logger.debug(
                        "Skip points batch %d since the time range it covers does not "
                        "exceed `min_sample_time_range` (%f)",
                        _i_point_batch,
                        min_sample_time_range
                    )
                    continue

                # Split points into several areas
                points_area_mask = points_splitter.split(
                    points_r=_points_batch[:, rd.PointFeatureIndex.R.value],
                    points_azim=_points_batch[:, rd.PointFeatureIndex.AZ.value]
                )  # shape=(n_points, n_split_areas)
                _points_batch_with_area_mask = np.concatenate(
                    (_points_batch, points_area_mask),
                    axis=-1
                )
                # Write radar type ID field
                _points_batch_with_area_mask[:, rd.PointFeatureIndex.RADAR_TYPE] = \
                    radar_type_id
                # Write the next same-track point coords into ROUTE_NEXT_POINT columns.
                _points_batch_with_area_mask[:, ROUTE_NEXT_POINT_COLS] = \
                    next_point_coords[_start_point_idx:_end_point_idx]

                if fn_on_points_batch_with_area_mask is not None:
                    ret_val = fn_on_points_batch_with_area_mask(_points_batch_with_area_mask)
                    if ret_val == 1:
                        break

            if callback_on_file_processed is not None:
                ret_val = callback_on_file_processed()
                if ret_val == 1:
                    break


def gen_dataset_from_one_repo_impl(
    original_data_dir: Union[str, Path],
    datasets_save_dir: Union[str, Path],
    max_sample_time_range: float,
    min_sample_time_range: float,
    min_track_len: int,
    time_step: float,
    max_values: DataConfigMaxValue,
    normalization_factors: DataConfigNormalizationFactor,
    split_azim_width: float,
    split_pad_width: float,
    r1: float,
    sample_len: int,
    n_features: int,
    n_areas_to_sample_in_each_step: int,
    only_primary_and_fused_points: bool,
    remove_uncertain_points: bool,
    reg_the_last_but_one_point: bool,
    dt_between_circles: Optional[float],
    radial_split_borders: Optional[Tuple[Tuple[float, float], ...]],
    suppress_false_points: bool,
    remove_the_last_track_points: bool,
    remove_the_last_circle_points: bool,
    remove_old_fa_points: bool,
    r_clutter_region: float,
    cache_size: int,
    dataset_type: str,
    radar_type_name: str,
    radar_type_id: int,
    max_sample_count: Optional[int],
    plot_samples: bool,
    rewrite: bool,
    logger: logging.Logger,
    validate_seed: int = 42
):
    import shutil

    subtype_ori_data_dir = Path(original_data_dir) / dataset_type

    dt_between_circles = dt_between_circles or np.inf

    if min_track_len < GLOBAL_MIN_TRACK_LEN:
        logger.warning(f"`min_track_len` of {original_data_dir} is {min_track_len}, "
                       f"which must be >= 3. Set it to 3.")
        min_track_len = GLOBAL_MIN_TRACK_LEN

    if dataset_type == "validate":
        if not subtype_ori_data_dir.exists() or not subtype_ori_data_dir.is_dir():
            test_dir = Path(original_data_dir) / "test"
            if not test_dir.exists() or not test_dir.is_dir():
                raise ValueError(
                    f"Neither validate directory ({subtype_ori_data_dir}) nor test directory ({test_dir}) exists. "
                    f"Cannot create validate dataset."
                )

            test_files = sorted(test_dir.iterdir())
            if not test_files:
                raise ValueError(
                    f"Test directory {test_dir} is empty. Cannot sample for validate dataset."
                )

            rng = np.random.RandomState(seed=validate_seed)
            selected_file = rng.choice(test_files)

            subtype_ori_data_dir.mkdir(parents=True, exist_ok=True)
            dest_file = subtype_ori_data_dir / selected_file.name
            shutil.copy2(selected_file, dest_file)
            logger.info(
                f"Created validate directory and sampled {selected_file.name} from test directory "
                f"(seed={validate_seed}) to {dest_file}"
            )
        else:
            validate_files = sorted(f for f in subtype_ori_data_dir.iterdir() if f.is_file())
            if not validate_files:
                test_dir = Path(original_data_dir) / "test"
                if not test_dir.exists() or not test_dir.is_dir():
                    raise ValueError(
                        f"Validate directory {subtype_ori_data_dir} is empty and test directory "
                        f"({test_dir}) does not exist. Cannot create validate dataset."
                    )

                test_files = sorted(test_dir.iterdir())
                if not test_files:
                    raise ValueError(
                        f"Test directory {test_dir} is empty. Cannot sample for validate dataset."
                    )

                rng = np.random.RandomState(seed=validate_seed)
                selected_file = rng.choice(test_files)

                dest_file = subtype_ori_data_dir / selected_file.name
                shutil.copy2(selected_file, dest_file)
                logger.info(
                    f"Validate directory was empty. Sampled {selected_file.name} from test directory "
                    f"(seed={validate_seed}) to {dest_file}"
                )

    if not subtype_ori_data_dir.exists() or not subtype_ori_data_dir.is_dir():
        raise ValueError(
            "Directory `original_data_dir`/`dataset_type` == %s does not exist.",
            subtype_ori_data_dir
        )

    original_data_files = sorted(subtype_ori_data_dir.iterdir())

    if not original_data_files:
        logger.warning(
            "There is not data file in %s. Exit.",
            subtype_ori_data_dir
        )
        return

    logger.info(f"`radial_split_borders` is: {radial_split_borders}")

    logger.info(f"Start generating samples for dataset: {dataset_type}")
    logger.info("Original data directory: %s", subtype_ori_data_dir)

    create_dir(datasets_save_dir)
    dataset_file_path = Path(datasets_save_dir, f"{dataset_type}.hdf5")
    if dataset_file_path.is_file() and rewrite:
        logger.info(
            f"Delete existing dataset file {dataset_file_path} since 'rewrite' "
            f"option is on."
        )
        os.remove(dataset_file_path)

    if normalization_factors.r is not None:
        r_max = normalization_factors.r
    else:
        r_max = max_values.r
    points_splitter = HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
        r_max=r_max,
        area_width=np.deg2rad(split_azim_width),
        pad_width=split_pad_width,
        r1=r1
    )

    if n_features < 0:
        n_features = rd.PointFeatureIndex.required_columns
    max_sample_count = max_sample_count or np.inf
    with h5py.File(dataset_file_path, mode="a") as dataset_file:
        samples_dataset = \
            dataset_file.get("points_in_areas") \
            or dataset_file.create_dataset(
                f"points_in_areas",
                shape=(0, sample_len, n_features),
                maxshape=(None, sample_len, None),
                dtype=np.float32
            )
        area_ids_dataset = \
            dataset_file.get("area_ids") \
            or dataset_file.create_dataset(
                f"area_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32
            )
        area_masks_dataset = \
            dataset_file.get("area_masks") \
            or dataset_file.create_dataset(
                f"area_masks",
                shape=(0, sample_len, points_splitter.n_split_areas),
                maxshape=(None, sample_len, None),
                dtype=np.bool_
            )
        logger.info("Writing to file %s (path: %s)",
                    dataset_file,
                    dataset_file_path)

        TRACK_ID_COL_IDX = rd.PointFeatureIndex.TRACK_ID_MANUAL.value
        TIME_COL_IDX = rd.PointFeatureIndex.RADAR_TIME.value
        X_COL_IDX = rd.PointFeatureIndex.X.value
        Y_COL_IDX = rd.PointFeatureIndex.Y.value

        features_to_si_coeffs = dict_to_numba(
            rd.POINT_FEATURES_TO_SI_UNIT_COEFF_MAPPINGS,
            key_type=numba.int64,
            value_type=numba.float64
        )

        real_saved_sample_lengths = []
        real_sample_lengths = []

        # The scan period of a circle.
        # TODO: don't hardcode the number of circles (10) here, which is
        #       consistent with `DataConfig.fill_in_defaults`.
        circle_period = max_sample_time_range / 10

        # Per-sample probability of keeping all false alarms that are 5+ circles
        # old in the sample (instead of dropping them before clipping). Can be
        # overridden via the MA_OLD_FA_KEEP_PROB environment variable.
        OLD_FA_KEEP_PROB = float(os.environ.get("MA_OLD_FA_KEEP_PROB", "0.1"))
        if not 0.0 <= OLD_FA_KEEP_PROB <= 1.0:
            raise ValueError(
                f"MA_OLD_FA_KEEP_PROB must be in [0, 1], got {OLD_FA_KEEP_PROB}"
            )

        samples_cache = []
        area_ids_cache = []
        area_masks_cache = []

        def _process_points_with_area_masks(_points_batch_with_area_mask):
            nonlocal samples_cache, area_ids_cache, area_masks_cache

            # ########## split samples by areas ##########
            # (samples are NOT clipped here; clipping is done after filtering,
            #  see `sample_points_filter` & `clip_samples` below)
            _samples, _area_ids, point_indices, _real_n_points_in_areas = split_area_samples(
                _points_batch_with_area_mask,
                points_splitter.n_split_areas,
                radial_split_borders=radial_split_borders
                # TODO: [EXPERIMENTAL] split each area into near and remote regions
            )

            # If only the samples containing points of the last 1s in the time step
            # are to be considered, drop other samples:
            if n_areas_to_sample_in_each_step == -1:
                ts = _points_batch_with_area_mask[:, TIME_COL_IDX]
                last_1s_points_mask = (ts.max() - ts) <= 1
                # -0.5 < t - t_min - 89.5 < 0.5
                # last_1s_points_mask = np.abs(ts - ts.min() - 89.5) < 0.5
                remote_region_points_mask = \
                    _points_batch_with_area_mask[:, rd.PointFeatureIndex.R] > r1
                areas_to_keep = np.where(
                    _points_batch_with_area_mask[
                        last_1s_points_mask & remote_region_points_mask
                        ][:, -points_splitter.n_split_areas:].any(0)
                )[0]
                areas_to_keep = set(areas_to_keep)
            elif n_areas_to_sample_in_each_step == 0:
                areas_to_keep = set(range(points_splitter.n_split_areas))
            elif 0 < n_areas_to_sample_in_each_step <= points_splitter.n_split_areas:
                # Seed the area sampling from a stable identity (radar name, the
                # first point time and the batch size) rather than the global
                # `np.random`, so that datasets generated on different machines
                # are identical (required for DDP, see the `remove_old_fa_points`
                # branch below which does the same).
                _batch_id_key = (
                    f"{radar_type_name}|"
                    f"{float(_points_batch_with_area_mask[0, TIME_COL_IDX]):.3f}|"
                    f"{len(_points_batch_with_area_mask)}"
                )
                _batch_rng = np.random.RandomState(
                    zlib.crc32(_batch_id_key.encode("utf-8"))
                )
                areas_to_keep = _batch_rng.choice(
                    range(points_splitter.n_split_areas),
                    n_areas_to_sample_in_each_step,
                    replace=False
                )
            else:
                raise ValueError(
                    f"Invalid `n_areas_to_sample_in_each_step`: "
                    f"{n_areas_to_sample_in_each_step}. "
                    f"It must be an integer in range [-1, {points_splitter.n_split_areas}]."
                )
            logger.debug(f"Areas {areas_to_keep} are kept ({len(areas_to_keep)} in total) since "
                         f"n_areas_to_sample_in_each_step={n_areas_to_sample_in_each_step}")

            # ########## process each sample: fill in labels of each point ##########
            # labels are (t,x,y) coordinates of route's start and end points
            for i_area, (_sample, _area_id, _real_n_points) in enumerate(
                zip(_samples, _area_ids, _real_n_points_in_areas)
            ):
                if i_area not in areas_to_keep:
                    continue

                # Sort points by time
                _sample = _sample[np.argsort(_sample[:, TIME_COL_IDX])]

                # Filter out routes which are too short or do not satisfy dt-between-point constraint
                track_id_set = np.unique(_sample[:, TRACK_ID_COL_IDX])
                for track_id in track_id_set:
                    if track_id <= 0:
                        continue

                    track_points_indices = np.where(
                        _sample[:, TRACK_ID_COL_IDX]
                        == track_id
                    )[0]

                    set_as_uncertain = (
                        _sample[track_points_indices].shape[0] < min_track_len
                        # or
                        # _sample[track_points_indices[-1], TIME_COL_IDX]
                        # - _sample[track_points_indices[-2], TIME_COL_IDX]
                        # >= dt_between_circles
                        or
                        np.any(
                            np.diff(_sample[track_points_indices, TIME_COL_IDX])
                            >= 3.1 * dt_between_circles
                        )
                    )

                    if set_as_uncertain:
                        _sample[track_points_indices, TRACK_ID_COL_IDX] = -1

                # Remove points of uncertain routes if required (since some may
                # be relabeled in previous processing)
                if remove_uncertain_points:
                    to_remove_mask = np.where(_sample[:, TRACK_ID_COL_IDX] == -1)
                    _sample[to_remove_mask] = np.nan
                    sorted_indices = np.argsort(_sample[:, rd.PointFeatureIndex.RADAR_TIME])
                    _sample = _sample[sorted_indices]

                # Record real length of this sample no matter whether it will be saved
                real_sample_lengths.append(_real_n_points)

                # Calculate the number of valid (i.e. non-NaN) points and target points
                _real_n_points = (_sample[:, TRACK_ID_COL_IDX] >= -1).sum()
                n_valid_points = (~np.isnan(_sample[:, TRACK_ID_COL_IDX])).sum()
                n_target_points = (_sample[:, TRACK_ID_COL_IDX] > 0).sum()

                # If the sample cannot pass the checking, skip it
                is_sample_ok, msg = check_sample_validity(
                    _sample,
                    _area_id,
                    _real_n_points,
                    sample_len,
                    n_valid_points,
                    n_target_points,
                    min_sample_time_range,
                    max_sample_time_range,
                    TIME_COL_IDX,
                    TRACK_ID_COL_IDX
                )
                if not is_sample_ok:
                    logger.debug(f"Sample is skipped due to {msg}")
                    continue

                # Record real length of this sample if it is to be saved
                real_saved_sample_lengths.append(_real_n_points)

                # If there are too many false points, randomly drop some of them
                if suppress_false_points and (n_valid_points - n_target_points) > 3 * n_target_points:
                    false_points_indices = np.where(_sample[:, TRACK_ID_COL_IDX] <= 0)[0]
                    logger.debug(
                        "There are %d false points (> 3 * n_target_points = %d). "
                        "Randomly keep %d of them.",
                        len(false_points_indices),
                        3 * n_target_points,
                        2 * n_target_points
                    )
                    kept_fp_indices = np.random.choice(
                        false_points_indices,
                        size=2 * n_target_points,
                        replace=False
                    )
                    _sample = np.concatenate(
                        [_sample[_sample[:, TRACK_ID_COL_IDX] > 0],
                         _sample[kept_fp_indices]],
                        axis=0
                    )
                    _sample = _sample[np.argsort(_sample[:, TIME_COL_IDX])]

                # If there is no points, skip this sample (for example, this area
                # only contains false points and they are all removed because
                # `suppress_false_points` is true).
                if len(_sample) == 0:
                    continue

                # ########## filter points, clip sample & generate labels ##########
                # Seed the route labels (especially ROUTE_END_POINT_IDX) so that
                # `sample_points_filter` can recognize the last point of each
                # track. All labels are recalculated at last because filtering
                # & clipping move points around; the per-point next coords are
                # preserved here (update_next=False) for the final update.
                _sample = recalculate_route_point_labels(
                    _sample, reg_the_last_but_one_point, update_next=False
                )

                # Each rule below drops points BEFORE clipping, so that the
                # clipped sample is still filled up to `sample_len` points; the
                # training/eval pipeline does no point removal anymore.
                if remove_old_fa_points:
                    # Deterministically (reproducible on every machine) keep all
                    # old false alarms in ~`OLD_FA_KEEP_PROB` of the samples: the
                    # decision is a hash of a stable sample identity (radar name,
                    # area id, first point time, raw point count) instead of an
                    # unseeded `np.random.rand()`, so that datasets generated on
                    # different machines are identical (required for DDP).
                    _valid_rows = np.flatnonzero(~np.isnan(_sample[:, TIME_COL_IDX]))
                    _id_key = (
                        f"{radar_type_name}|{float(_area_id):.3f}|"
                        f"{_sample[_valid_rows[0], TIME_COL_IDX]:.3f}|{_valid_rows.size}"
                    )
                    _keep_old_fa = (
                        zlib.crc32(_id_key.encode("utf-8")) % 1_000_000
                    ) < OLD_FA_KEEP_PROB * 1_000_000
                    _fa_lookback_window_length = (
                        np.inf if _keep_old_fa else 5 * circle_period
                    )
                else:
                    _fa_lookback_window_length = np.inf
                _sample = sample_points_filter(
                    _sample[None, ...],
                    remove_the_last_track_points=remove_the_last_track_points,
                    # Do not input the points in the latest circle into the model,
                    # and (usually) not the false alarms older than 5 circles.
                    # TODO: don't hardcode the numbers of circles (1 & 5) here
                    points_drop_window_length=circle_period if remove_the_last_circle_points else 0.0,
                    fa_lookback_window_length=_fa_lookback_window_length,
                    pad_token_value=np.nan,
                    return_mask=False
                )[0]

                # If no target point remains after filtering, skip this sample.
                if (_sample[:, TRACK_ID_COL_IDX] > 0).sum() == 0:
                    logger.debug(f"Sample (area id: {_area_id}) is skipped since no "
                                 f"target point remains after filtering")
                    continue

                # Choose the culling key: for samples in which all points are in
                # the near region, the points with the highest SNRs (instead of
                # the newest ones) are kept when the sample has to be clipped.
                # This is expressed by using the SNR column (resp. the time
                # column) as the culling key `select_by_index`.
                # _select_by_index = (
                #     rd.PointFeatureIndex.SNR.value
                #     if np.nanmax(_sample[:, rd.PointFeatureIndex.R.value]) < 100e3
                #     else TIME_COL_IDX
                # )
                _select_by_index = TIME_COL_IDX
                # Over-length samples are culled inside `clip_samples`. When
                # old-FA removal is enabled (`remove_old_fa_points`),
                # `sample_points_filter` already dropped the old false alarms
                # uniformly, so `clip_samples` only sorts ALL points (targets and
                # false alarms alike) by the culling key and drops the smallest
                # ones -- no clutter-region / FA-priority is applied (pass
                # `fa_track_id_index=None`). When it is disabled (default),
                # `clip_samples` first drops the false alarms inside the clutter
                # region (`r <= r_clutter_region`) with the smallest culling-key
                # values, then falls back to dropping the smallest-key points.
                _sample = clip_samples(
                    _sample[None, ...],
                    max_sample_len=sample_len,
                    select_by_index=_select_by_index,
                    fa_track_id_index=(None if remove_old_fa_points else TRACK_ID_COL_IDX),
                    fa_track_id_value=0,
                    r_index=rd.PointFeatureIndex.R.value,
                    r_clutter_region=r_clutter_region
                )[0]

                # (Re)generate the labels of the final sample: START/END point
                # indices of each route, NEXT point coords (& LOC1 indices if
                # required) & the flags for circle loss.
                _sample = recalculate_route_point_labels(_sample, reg_the_last_but_one_point)

                logger.debug(f'Track IDs in sample (area id: {_area_id}): '
                             f'{np.unique(_sample[:, TRACK_ID_COL_IDX])}')

                # add sample data into cache
                samples_cache.append(_sample[:, :-points_splitter.n_split_areas])

                _area_ids[np.isnan(_area_ids)] = -1
                area_ids_cache.append(_area_id.astype(np.int32))

                _area_masks = _sample[:, -points_splitter.n_split_areas:]
                _area_masks[np.isnan(_area_masks)] = 0
                area_masks_cache.append(_area_masks.astype(np.bool_))

                # plot if needed
                if plot_samples:
                    index_tar = np.where((_sample[:, TRACK_ID_COL_IDX] > 0))[0]
                    index_fa = np.where((_sample[:, TRACK_ID_COL_IDX] == 0))[0]
                    index_uncertain_points = np.where((_sample[:, TRACK_ID_COL_IDX] == -1))[0]

                    fig1, ax1 = plt.subplots()

                    ax1.scatter(_sample[index_fa, rd.PointFeatureIndex.X.value] / 1e3,
                                _sample[index_fa, rd.PointFeatureIndex.Y.value] / 1e3,
                                s=6, color='r', label="False points")
                    ax1.scatter(_sample[index_tar, rd.PointFeatureIndex.X.value] / 1e3,
                                _sample[index_tar, rd.PointFeatureIndex.Y.value] / 1e3,
                                s=6, color='g', label="Target points")
                    ax1.scatter(_sample[index_uncertain_points, rd.PointFeatureIndex.X.value] / 1e3,
                                _sample[index_uncertain_points, rd.PointFeatureIndex.Y.value] / 1e3,
                                s=6, color='orange', label="Uncertain points")

                    _angles = np.linspace(-np.pi, np.pi, 200)
                    _sin_angles = np.sin(_angles)
                    _cos_angles = np.cos(_angles)
                    if radial_split_borders is not None:
                        for r in functools.reduce(lambda l1, l2: list(l1) + list(l2), radial_split_borders):
                            if r <= 0:
                                continue
                            r = r * 1e-3
                            ax1.plot(r * _cos_angles, r * _sin_angles, linestyle="--", linewidth=1)

                    circumrect_corners = points_splitter._circumrect_splitter.rect_corners[_area_id] / 1e3
                    # shape=(4, 2)
                    corners = circumrect_corners.T  # shape=(2, 4)
                    corners = np.concatenate([corners, corners[:, [0]]], axis=-1)

                    ax1.plot(*corners, color="orange")
                    ax1.set_aspect(1)
                    ax1.legend()
                    _t_max = np.nanmax(_sample[:, TIME_COL_IDX])
                    ax1.set_title(f"{radar_type_name}-Area{_area_id}-t={_t_max:.2f}s")

                    plt.show()

            if len(samples_cache) < cache_size:
                logger.debug(
                    "Sample counts = %d, not enough for saving. Caching.",
                    len(samples_cache)
                )
                return 0  # continue to next points batch

            # Pop leading `cache_size` samples from cache and append them into HDF5 dataset:
            samples_to_save = np.array(
                samples_cache[:cache_size]
            )  # shape=(cache_size, sample_len, n_features)
            samples_cache = samples_cache[cache_size:]
            area_ids_to_save = np.array(area_ids_cache[:cache_size])  # shape=(cache_size,)
            area_ids_cache = area_ids_cache[cache_size:]
            area_masks_to_save = np.array(area_masks_cache[:cache_size])
            # shape=(cache_size,)
            area_masks_cache = area_masks_cache[cache_size:]

            if samples_dataset.shape[0] == 0:
                if samples_dataset.shape[2] != samples_to_save.shape[2]:
                    samples_dataset.resize(samples_to_save.shape[2], axis=2)
                if area_masks_dataset.shape[2] != area_masks_to_save.shape[2]:
                    area_masks_dataset.resize(samples_to_save.shape[2], axis=2)
            samples_dataset.resize(samples_dataset.shape[0] + samples_to_save.shape[0], axis=0)
            samples_dataset[-samples_to_save.shape[0]:] = samples_to_save
            area_ids_dataset.resize(area_ids_dataset.shape[0] + samples_to_save.shape[0], axis=0)
            area_ids_dataset[-samples_to_save.shape[0]:] = area_ids_to_save
            area_masks_dataset.resize(area_masks_dataset.shape[0] + samples_to_save.shape[0], axis=0)
            area_masks_dataset[-samples_to_save.shape[0]:] = area_masks_to_save
            dataset_file.flush()

            logger.info(f"Samples saved (samples shape={samples_to_save.shape}, "
                        f"area_ids shape={area_ids_to_save.shape}, "
                        f"area_masks shape={area_masks_to_save.shape}).")
            logger.info(f"Dataset shapes: samples shape={samples_dataset.shape}, "
                        f"area_ids shape={area_ids_dataset.shape}, "
                        f"area_masks shape={area_masks_dataset.shape}.")
            if samples_dataset.shape[0] > max_sample_count:
                logger.info(f"There are already {samples_dataset.shape[0]} samples saved. "
                            f"Finish data generating.")
                return 1  # break iteration over next points batch

        def _callback_on_file_processed():
            if samples_dataset.shape[0] > max_sample_count:
                logger.info(f"There are already {samples_dataset.shape[0]} samples saved. "
                            f"Finish data generating.")
                return 1
            return 0

        generate_samples(
            original_data_files,
            max_sample_time_range,
            min_sample_time_range,
            max_values,
            time_step,
            points_splitter,
            n_features,
            radar_type_id,
            _process_points_with_area_masks,
            _callback_on_file_processed,
            logger,
            only_primary_and_fused_points,
            remove_uncertain_points,
            dt_between_circles=dt_between_circles
        )

        # If there are still samples in cache when all original .dat files were iterated,
        # save them:
        if len(samples_cache):
            samples_to_save = np.array(
                samples_cache[:cache_size]
            )  # shape=(n_samples_in_file, sample_len, n_features)
            samples_cache = samples_cache[cache_size:]
            area_ids_to_save = np.array(area_ids_cache[:cache_size])  # shape=(n_samples_in_file,)
            area_ids_cache = area_ids_cache[cache_size:]
            area_masks_to_save = np.array(area_masks_cache[:cache_size])
            area_masks_cache = area_masks_cache[cache_size:]

            samples_dataset.resize(samples_dataset.shape[0] + samples_to_save.shape[0], axis=0)
            samples_dataset[-samples_to_save.shape[0]:] = samples_to_save
            area_ids_dataset.resize(area_ids_dataset.shape[0] + samples_to_save.shape[0], axis=0)
            area_ids_dataset[-samples_to_save.shape[0]:] = area_ids_to_save
            area_masks_dataset.resize(area_masks_dataset.shape[0] + samples_to_save.shape[0], axis=0)
            area_masks_dataset[-samples_to_save.shape[0]:] = area_masks_to_save
            dataset_file.flush()

            logger.info(f"Samples saved (samples shape={samples_to_save.shape}, "
                        f"area_ids shape={area_ids_to_save.shape}, "
                        f"area_masks shape={area_masks_to_save.shape}).")
            logger.info(f"Dataset shapes: samples shape={samples_dataset.shape}, "
                        f"area_ids shape={area_ids_dataset.shape}, "
                        f"area_masks shape={area_masks_dataset.shape}.")

        if len(real_saved_sample_lengths) == 0:
            logger.warning(f"No samples generated from data files {original_data_files}.")
        else:
            logger.info(
                f"Statistics of \033[1mREAL\033[0m number of points in \033[1mSAVED\033[0m samples: "
                f"avg.=%.2f,  std.=%.2f,  [0%%,25%%,50%%,75%%,100%%]-percentiles=%s",
                np.mean(real_saved_sample_lengths),
                np.std(real_saved_sample_lengths, ddof=1),
                np.percentile(real_saved_sample_lengths, q=[0, 25, 50, 75, 100])
            )
            logger.info(
                f"Statistics of \033[1mREAL\033[0m number of points in \033[1mALL\033[0m samples: "
                f"avg.=%.2f,  std.=%.2f,  [0%%,25%%,50%%,75%%,100%%]-percentiles=%s",
                np.mean(real_sample_lengths),
                np.std(real_sample_lengths, ddof=1),
                np.percentile(real_sample_lengths, q=[0, 25, 50, 75, 100])
            )
            logger.info(
                f"Finish dataset generating. All data are saved to {dataset_file} "
                f"(path: {dataset_file_path}). "
                f"Saved data: {[str(_dset) for _dset in dataset_file.values()]}"
            )


@app.command()
def from_one_repo(
    original_data_dir: Annotated[str, typer.Argument(
        help="Directory where original data resides, which must contains at least "
             "subdirectories named 'train' and 'test' and a 'data_config.json' file."
    )],
    datasets_save_dir: Annotated[Optional[str], typer.Argument(
        help="Directory where generated dataset is to be saved. "
             "If not existing, it will be created."
    )],
    dataset_type: Annotated[str, typer.Option(
        help="Type of dataset to generate.",
        click_type=click.Choice([
            "train",
            "test",
            "validate",
            "all"
        ]),
        case_sensitive=False
    )] = "all",
    max_sample_time_range: Annotated[Optional[float], typer.Option(
        help="Maximum time range to collect points for generating samples each time."
    )] = None,
    min_sample_time_range: Annotated[Optional[float], typer.Option(
        help="Minimum time range to collect points for generating samples each time."
    )] = None,
    min_track_len: Annotated[Optional[int], typer.Option(
        help="Minimum length of tracks in samples. For those tracks whose lengths "
             "are smaller than this value, their points will be treated as false "
             "points."
    )] = None,
    time_step: Annotated[Optional[int], typer.Option(
        help="Per how long time to generate new samples."
    )] = None,
    max_dist: Annotated[Optional[float], typer.Option(
        help="Maximum detecting distance (in meters)."
    )] = None,
    max_amp: Annotated[Optional[float], typer.Option(
        help="Maximum signal amplitude."
    )] = None,
    max_snr: Annotated[Optional[float], typer.Option(
        help="Maximum signal-to-noise ratio."
    )] = None,
    max_height: Annotated[Optional[float], typer.Option(
        help="Maximum height (in meters)."
    )] = None,
    split_azim_width: Annotated[Optional[float], typer.Option(
        help="Azimuthal width of each split area (in degrees)."
    )] = None,
    split_pad_width: Annotated[Optional[float], typer.Option(
        help="Padding width of each split area (in meters)."
    )] = None,
    r1: Annotated[Optional[float], typer.Option(
        help="Radius of near region of each split area (in meters)."
    )] = None,
    sample_len: Annotated[Optional[int], typer.Option(
        help="Length of each sample, i.e. the number of points preserved in "
             "each split area. If there are not that much points in one sample, "
             "points filled with NaN will be appended; if there are more than "
             "`sample_len` points, the latest `sample_len` points will be saved."
    )] = None,
    n_features: Annotated[Optional[int], typer.Option(
        help="Number of feature columns of each point in the saved dataset. "
             "If the input has fewer columns, the missing ones are zero-padded; "
             "if it has more, the surplus columns are discarded. Extended "
             "PointFeatureIndex attributes (value >= 64) are only kept when "
             "this option covers them. Pass a negative value (e.g. -1) to use "
             "PointFeatureIndex.required_columns instead."
    )] = 64,
    n_areas_to_sample_in_each_step: Annotated[Optional[int], typer.Option(
        help="The number of areas to be sampled in each step. "
             "-1 means only the sample containing remote region points in "
             "the last second of each time step will be kept; "
             "0 means all samples will be kept, i.e. ceil(360/split_azim_width) "
             "samples will be generated for each time step; "
             "other integral value n in [1,ceil(360/split_azim_width)] means "
             "n samples will be randomly chosen from all areas."
    )] = 0,
    only_primary_and_fused_points: Annotated[Optional[bool], typer.Option(
        help="If True, only primary and fused points will be kept in dataset."
    )] = None,
    radial_split_borders: Annotated[Optional[str], typer.Option(
        help="Designate borders of radial splits of each fan area split (experimental).",
        callback=parse_nested_float_tuple
    )] = "((-1,-1),)",
    suppress_false_points: Annotated[Optional[bool], typer.Option(
        help="If True, suppress false points such that the number of false points "
             "<= 2 * the number of target points in each generated sample."
    )] = None,
    remove_uncertain_points: Annotated[Optional[bool], typer.Option(
        help="If True, remove uncertain points (track ID = -1) from generated "
             "dataset. This option will globally overwrite the one set in "
             "data config files."
    )] = None,
    reg_the_last_but_one_point: Annotated[bool, typer.Option(
        help="If true, the last but one point in each track will also be recorded "
             "in regression labels."
    )] = True,
    cache_size: Annotated[Optional[int], typer.Option(
        help="Number of samples to cache during dataset generation."
    )] = None,
    radar_type_name: Annotated[Optional[str], typer.Option(
        help="Radar type string. If not provided, the name of `original_data_dir` "
             "will be used."
    )] = None,
    max_sample_count: Annotated[Optional[int], typer.Option(
        help="Maximum number of samples to generate."
    )] = None,
    plot_samples: Annotated[Optional[bool], typer.Option(
        help="Whether to plot samples. Switch on this option only for debugging."
    )] = False,
    rewrite: Annotated[Optional[bool], typer.Option(
        help="Whether to rewrite HDF5 files. If this is true and there is a "
             "<dataset_type>.hdf5 file under <original_data_dir>, that file "
             "will be removed before any writing; otherwise samples will be "
             "appended into the file."
    )] = False,
    log_level: Annotated[
        str, typer.Option(
            help="Logging level.",
            click_type=click.Choice([
                "CRITICAL",
                "FATAL",
                "ERROR",
                "WARNING",
                "WARN",
                "INFO",
                "DEBUG",
                "NOTSET"
            ]),
            case_sensitive=False
        )
    ] = "INFO",
    remove_the_last_track_points: Annotated[bool, typer.Option(
        help="Whether to remove the last point of each track from samples "
             "(applied before clipping)."
    )] = False,
    remove_the_last_circle_points: Annotated[bool, typer.Option(
        help="Whether to remove all points in the latest scan circle from "
             "samples (applied before clipping)."
    )] = False,
    remove_old_fa_points: Annotated[bool, typer.Option(
        help="Whether to remove false alarms older than 5 scan circles from "
             "samples (applied before clipping). The fraction of samples that "
             "keep their old false alarms anyway is controlled by the "
             "MA_OLD_FA_KEEP_PROB environment variable (default 0.1). When "
             "true, `sample_points_filter` drops the old false alarms uniformly, "
             "and `clip_samples` drops the points with the smallest culling-key "
             "value (no clutter-region FA priority is applied). When false "
             "(default), old false alarms are kept and `clip_samples` culls "
             "over-length samples by first dropping the clutter-region false "
             "alarms, then the remaining smallest-key points."
    )] = False,
    r_clutter_region: Annotated[float, typer.Option(
        help="Radius (in meters) of the clutter region used when clipping "
             "over-length samples without removing old false alarms. Defaults to "
             "the value in the data config (200 km if unset)."
    )] = None,
    validate_seed: Annotated[
        int, typer.Option(
            help="Random seed for validate dataset sampling (to ensure reproducibility "
                 "across different machines and times). Only used when dataset_type is "
                 "'validate' and validate directory is empty."
        )
    ] = 42
):
    dataset_type = dataset_type.lower()
    if dataset_type == "train":
        dataset_types = ("train",)
    elif dataset_type == "test":
        dataset_types = ("test",)
    elif dataset_type == "validate":
        dataset_types = ("validate",)
    elif dataset_type == "all":
        dataset_types = ("train", "test", "validate")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    main_logger = setup_logger(f"data_gen", log_level=log_level)

    original_data_dir = Path(original_data_dir)
    datasets_save_dir = Path(datasets_save_dir)

    data_config: DataConfig = DataConfig.from_file(original_data_dir / DATA_CONFIG_FILENAME)
    data_config_dict = data_config.model_dump()

    if isinstance(radial_split_borders, str):
        radial_split_borders = parse_nested_float_tuple(radial_split_borders)

    # Overwrite config if some values are explicitly given through cmdline options:
    locals_copy = dict(locals())
    for p_name, p_val in locals_copy.items():
        if p_name in data_config_dict and p_val is not None:
            original_p_val = getattr(data_config, p_name)
            setattr(data_config, p_name, p_val)
            main_logger.info(f"Data config field {p_name} is overwritten by manual cmdline setting: "
                             f"{original_p_val} -> {getattr(data_config, p_name)}")

    if radar_type_name is None:
        main_logger.info(
            f"--radar-type-name is not set. Use the directory name {original_data_dir.name}."
        )
        radar_type_name = original_data_dir.name
    collected_data_config = CollectedDataConfig().add_config(radar_type_name, data_config, "ignore")

    if remove_uncertain_points is None:
        remove_uncertain_points = data_config.remove_uncertain_points
    else:
        main_logger.info(f"--{'' if remove_uncertain_points else 'no-'}remove-uncertain-points "
                         f"is globally set. Uncertain points (track ID = -1) will be "
                         f"{'REMOVED FROM' if remove_uncertain_points else 'KEPT IN'} "
                         f"the dataset.")

    for _dataset_type in dataset_types:
        logger = setup_logger(f"data_gen_{_dataset_type}", log_level=log_level)
        gen_dataset_from_one_repo_impl(
            original_data_dir,
            datasets_save_dir,
            data_config.max_sample_time_range,
            data_config.min_sample_time_range,
            data_config.min_track_len,
            data_config.time_step,
            data_config.max_value,
            data_config.normalization_factor,
            data_config.split_azim_width,
            data_config.split_pad_width,
            data_config.r1,
            data_config.sample_len,
            n_features,
            data_config.n_areas_to_sample_in_each_step,
            data_config.only_primary_and_fused_points,
            remove_uncertain_points,
            reg_the_last_but_one_point,
            data_config.dt_between_circles,
            data_config.radial_split_borders,
            data_config.suppress_false_points,
            remove_the_last_track_points,
            remove_the_last_circle_points,
            remove_old_fa_points,
            data_config.r_clutter_region,
            data_config.cache_size,
            _dataset_type,
            radar_type_name,
            0,
            data_config.max_sample_count,
            plot_samples,
            rewrite,
            logger,
            validate_seed
        )

    with open(datasets_save_dir / COLLECTED_DATA_CONFIG_FILENAME, "w") as f:
        json.dump(collected_data_config.model_dump(), f, indent=2)


@app.command()
def from_multiple_repos(
    original_data_dir: Annotated[str, typer.Argument(
        help="Directory where original data resides."
    )],
    datasets_save_dir: Annotated[str, typer.Argument(
        help="Directory where generated dataset is to be saved. "
             "If not existing, it will be created."
    )],
    n_features: Annotated[int, typer.Option(
        help="Number of feature columns of each point in the saved dataset. "
             "If the input has fewer columns, the missing ones are zero-padded; "
             "if it has more, the surplus columns are discarded. Extended "
             "PointFeatureIndex attributes (value >= 64) are only kept when "
             "this option covers them. Pass a negative value (e.g. -1) to use "
             "PointFeatureIndex.required_columns instead."
    )] = 64,
    cache_size: Annotated[int, typer.Option(
        help="Number of samples to cache during dataset generation."
    )] = 256,
    radar_type_name_pattern: Annotated[str, typer.Option(
        help="Regex pattern for parsing radar type from names of sub-repos. "
             "A pattern group named `radar_type_name` must be contained."
    )] = r"^(?!\.)(?P<radar_type_name>.+)",
    plot_samples: Annotated[bool, typer.Option(
        help="Whether to plot samples. Switch on this option only for debugging."
    )] = False,
    dataset_type: Annotated[str, typer.Option(
        help="Type of dataset to be generated.",
        click_type=click.Choice([
            "train",
            "test",
            "validate",
            "all"
        ]),
        case_sensitive=False
    )] = "all",
    remove_uncertain_points: Annotated[Optional[bool], typer.Option(
        help="If True, remove uncertain points (track ID = -1) from generated "
             "dataset. This option will globally overwrite the one set in "
             "data config files."
    )] = None,
    reg_the_last_but_one_point: Annotated[bool, typer.Option(
        help="If true, the last but one point in each track will also be recorded "
             "in regression labels."
    )] = True,
    rewrite: Annotated[bool, typer.Option(
        help="Whether to rewrite HDF5 files. If this is true and there is a "
             "<dataset_type>.hdf5 file under <original_data_dir>, that file "
             "will be removed before any writing; otherwise samples will be "
             "appended into the file."
    )] = False,
    log_level: Annotated[
        str, typer.Option(
            help="Logging level.",
            click_type=click.Choice([
                "CRITICAL",
                "FATAL",
                "ERROR",
                "WARNING",
                "WARN",
                "INFO",
                "DEBUG",
                "NOTSET"
            ]),
            case_sensitive=False
        )
    ] = "INFO",
    remove_the_last_track_points: Annotated[bool, typer.Option(
        help="Whether to remove the last point of each track from samples "
             "(applied before clipping)."
    )] = False,
    remove_the_last_circle_points: Annotated[bool, typer.Option(
        help="Whether to remove all points in the latest scan circle from "
             "samples (applied before clipping)."
    )] = False,
    remove_old_fa_points: Annotated[bool, typer.Option(
        help="Whether to remove false alarms older than 5 scan circles from "
             "samples (applied before clipping). The fraction of samples that "
             "keep their old false alarms anyway is controlled by the "
             "MA_OLD_FA_KEEP_PROB environment variable (default 0.1). When "
             "true, `sample_points_filter` drops the old false alarms uniformly, "
             "and `clip_samples` drops the points with the smallest culling-key "
             "value (no clutter-region FA priority is applied). When false "
             "(default), old false alarms are kept and `clip_samples` culls "
             "over-length samples by first dropping the clutter-region false "
             "alarms, then the remaining smallest-key points."
    )] = False,
    r_clutter_region: Annotated[float, typer.Option(
        help="Radius (in meters) of the clutter region used when clipping "
             "over-length samples without removing old false alarms. Defaults to "
             "the value in the data config (200 km if unset)."
    )] = None,
    validate_seed: Annotated[
        int, typer.Option(
            help="Random seed for validate dataset sampling (to ensure reproducibility "
                 "across different machines and times). Only used when dataset_type is "
                 "'validate' and validate directory is empty."
        )
    ] = 42,
    max_sample_count_per_radar: Annotated[
        int, typer.Option(
            help="The maximum number of samples to be generated for each radar. "
                 "If not None, this will overwrite those set in data_config files."
        )
    ] = None
):
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = setup_logger(f"data_gen", log_level=log_level)

    original_data_dir = Path(original_data_dir)

    sub_repos = [sub_dir for sub_dir in sorted(original_data_dir.iterdir())
                 if sub_dir.is_dir()]
    rewrite_flags = {"train": rewrite, "test": rewrite, "validate": rewrite}

    dataset_type = dataset_type.lower()
    if dataset_type.lower() == "train":
        dataset_types = ("train",)
    elif dataset_type.lower() == "test":
        dataset_types = ("test",)
    elif dataset_type.lower() == "validate":
        dataset_types = ("validate",)
    elif dataset_type.lower() == "all":
        dataset_types = ("train", "test", "validate")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    collected_data_config_file = Path(datasets_save_dir) / COLLECTED_DATA_CONFIG_FILENAME
    if collected_data_config_file.is_file() and not rewrite:
        # Load existed collected data config (if needed and file exists)
        collected_data_config = CollectedDataConfig.from_file(collected_data_config_file)
    else:
        collected_data_config = CollectedDataConfig()
        if collected_data_config_file.is_file():
            os.remove(collected_data_config_file)
            logger.info(f"Deleted existing data config file {collected_data_config_file} "
                        f"since 'rewrite' option is on.")

    try:
        for sub_repo in sub_repos:
            m = re.match(radar_type_name_pattern, sub_repo.stem)
            if m is None:
                continue

            parsed_radar_type_name = m.groupdict()["radar_type_name"]
            data_config_file = sub_repo / DATA_CONFIG_FILENAME
            if not data_config_file.is_file():
                logger.warning(
                    f"There is not a data_config.json file under {sub_repo}. "
                    f"Skip it."
                )
                continue
            data_config: DataConfig = DataConfig.from_file(data_config_file)
            if r_clutter_region is not None:
                data_config.r_clutter_region = r_clutter_region
            collected_data_config.add_config(parsed_radar_type_name, data_config, conflict_strategy="check")
            radar_type_id = collected_data_config.radar_type_mapping[parsed_radar_type_name]

            if remove_uncertain_points is None:
                _remove_uncertain_points = data_config.remove_uncertain_points
            else:
                # globally overwrite
                _remove_uncertain_points = remove_uncertain_points
                logger.info(f"--{'' if remove_uncertain_points else 'no-'}remove-uncertain-points "
                            f"is globally set. Uncertain points (track ID = -1) will be "
                            f"{'REMOVED FROM' if remove_uncertain_points else 'KEPT IN'} "
                            f"the dataset.")

            if max_sample_count_per_radar is None:
                _max_sample_count = data_config.max_sample_count
            else:
                _max_sample_count = max_sample_count_per_radar

            for _dataset_type in dataset_types:
                sub_type_dir = sub_repo / _dataset_type
                if not sub_type_dir.is_dir():
                    if _dataset_type == "validate":
                        test_dir = sub_repo / "test"
                        if not test_dir.is_dir():
                            logger.warning(
                                f"Neither validate nor test directory exists in {sub_repo}. Skip."
                            )
                            continue
                        logger.info(
                            f"Validate directory does not exist. Will sample from test directory ({test_dir})."
                        )
                    else:
                        logger.warning(
                            f"{_dataset_type} directory does not exist in {sub_repo}. Skip."
                        )
                        continue

                logger.info("Start processing %s", sub_type_dir)
                gen_dataset_from_one_repo_impl(
                    sub_repo,
                    datasets_save_dir,
                    max_sample_time_range=data_config.max_sample_time_range,
                    min_sample_time_range=data_config.min_sample_time_range,
                    min_track_len=data_config.min_track_len,
                    time_step=data_config.time_step,
                    max_values=data_config.max_value,
                    normalization_factors=data_config.normalization_factor,
                    split_azim_width=data_config.split_azim_width,
                    split_pad_width=data_config.split_pad_width,
                    r1=data_config.r1,
                    sample_len=data_config.sample_len,
                    n_features=n_features,
                    n_areas_to_sample_in_each_step=data_config.n_areas_to_sample_in_each_step,
                    only_primary_and_fused_points=data_config.only_primary_and_fused_points,
                    remove_uncertain_points=_remove_uncertain_points,
                    reg_the_last_but_one_point=reg_the_last_but_one_point,
                    dt_between_circles=data_config.dt_between_circles,
                    radial_split_borders=data_config.radial_split_borders,
                    suppress_false_points=data_config.suppress_false_points,
                    remove_the_last_track_points=remove_the_last_track_points,
                    remove_the_last_circle_points=remove_the_last_circle_points,
                    remove_old_fa_points=remove_old_fa_points,
                    r_clutter_region=data_config.r_clutter_region,
                    cache_size=cache_size,
                    dataset_type=_dataset_type,
                    radar_type_name=parsed_radar_type_name,
                    radar_type_id=radar_type_id,
                    max_sample_count=_max_sample_count,
                    plot_samples=plot_samples,
                    rewrite=rewrite_flags[_dataset_type],
                    logger=logger,
                    validate_seed=validate_seed
                )

                rewrite_flags[_dataset_type] = False
    except Exception as e:
        logger.error(f"Generating process is interrupted due to {e}.\n"
                     f"{logger.error(traceback.format_exc())}")
    finally:
        with open(collected_data_config_file, "w") as f:
            json.dump(collected_data_config.model_dump(), f, indent=2)
        logger.info(f"Collected data config is saved to {collected_data_config_file}")


def main():
    app()


if __name__ == "__main__":
    main()
