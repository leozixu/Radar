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

import dataclasses
from typing import Union, Optional, Tuple, List

import numpy as np
import torch
from model_association import radar_definition as rd
from model_association.point_cc.classify import target_prob_from_regression
from model_association.point_cc.data import RealRadarRouteClusterFeatureColumnIndices
from model_association.point_cc.feature_augment import FeatureAugmenter
from model_association.point_cc.statistic import RegressResidualStatistic, AccuracyStatistic
from model_association.utils import get_tensor_array_pair
from route_tools.track_by_corr import multicirc_lookback_tracking, associating_accuracy

TrackedRoutePointsIndices = List[List[int]]


def eval_classify_metrics(
    probs: Union[torch.Tensor, np.ndarray],
    true_target_labels: Union[torch.Tensor, np.ndarray],
    rs: Union[torch.Tensor, np.ndarray],
    non_nan_points_mask: Union[torch.Tensor, np.ndarray],
    region_borders: Optional[tuple[tuple[float, float], ...]] = None
) -> tuple[
    tuple[AccuracyStatistic, AccuracyStatistic],
    list[
        tuple[AccuracyStatistic, AccuracyStatistic]
    ]
]:
    correct_mask = (probs > 0.5) == true_target_labels
    true_positive_mask = (probs > 0.5) & true_target_labels

    classify_accuracy_all = AccuracyStatistic(
        correct=correct_mask[non_nan_points_mask].sum().item(),
        total=non_nan_points_mask.sum().item()
    )
    classify_recall_all = AccuracyStatistic(
        correct=true_positive_mask[non_nan_points_mask].sum().item(),  # TP
        total=true_target_labels[non_nan_points_mask].sum().item()  # TP + FN
    )

    classify_stat_by_r_regions = []
    if region_borders is not None:
        for r_min, r_max in region_borders:
            in_region_mask = (r_min <= rs) & (rs <= r_max) & non_nan_points_mask

            _region_classify_accuracy = AccuracyStatistic(
                correct=correct_mask[in_region_mask].sum().item(),
                total=in_region_mask.sum().item()
            )
            _region_classify_recall = AccuracyStatistic(
                correct=true_positive_mask[in_region_mask].sum().item(),
                total=true_target_labels[in_region_mask].sum().item()
            )
            classify_stat_by_r_regions.append(
                (_region_classify_accuracy, _region_classify_recall)
            )

    return (classify_accuracy_all, classify_recall_all), classify_stat_by_r_regions


def eval_regress_metrics(
    normalized_coords_pred: torch.Tensor,  # shape=(n_points, n_dim_coords)
    normalized_coords_true: torch.Tensor,  # shape=(n_points, n_dim_coords)
    norm_factor: torch.Tensor,  # shape=(n_points, n_dim_coords)
    rs: torch.Tensor,  # shape=(n_points,)
    region_borders: Optional[tuple[tuple[float, float], ...]] = None
) -> tuple[
    RegressResidualStatistic,
    list[
        RegressResidualStatistic
    ]
]:
    residual_all = (
        torch.abs(normalized_coords_true - normalized_coords_pred)
        * norm_factor
    )  # shape=(n_points, n_dim_coords)
    # Some label positions may be NaN (e.g. the NEXT point of a route which has
    # no such point). Those positions must not be counted, otherwise the summed
    # residual and the (per-column) count become NaN.
    valid_mask = torch.isfinite(residual_all)
    residual_all = torch.where(valid_mask, residual_all, torch.zeros_like(residual_all))
    residual_count = valid_mask.sum(dim=0).detach().cpu().numpy()  # per column

    residual_stat_all = RegressResidualStatistic(
        total_res=torch.sum(residual_all, dim=0).detach().cpu().numpy(),
        count=residual_count
    )

    residual_stat_by_r_regions = []
    if region_borders is not None:
        for r_min, r_max in region_borders:
            in_region_mask = (r_min <= rs) & (rs <= r_max)

            residual_in_region = residual_all[in_region_mask]
            valid_in_region = valid_mask[in_region_mask]
            residual_stat_in_region = RegressResidualStatistic(
                total_res=torch.sum(residual_in_region, dim=0).detach().cpu().numpy(),
                count=valid_in_region.sum(dim=0).detach().cpu().numpy()
            )

            residual_stat_by_r_regions.append(
                residual_stat_in_region
            )

    return residual_stat_all, residual_stat_by_r_regions


def eval_association_metrics(
    batch_samples: np.ndarray,  # not normalized, shape=(sample_len, n_features) or (n_samples, sample_len, n_features)
    batch_non_nan_mask: np.ndarray,  # shape=batch_samples.shape[:-1]
    batch_corr_mat: np.ndarray,  # shape=(sample_len, sample_len) or (n_samples, sample_len, sample_len)
    t_index: int,
    x_index: int,
    y_index: int,
    route_id_index: int,
    batch_target_masks: Optional[np.ndarray] = None
) -> Tuple[AccuracyStatistic, List[TrackedRoutePointsIndices]]:
    asso_stat = AccuracyStatistic(correct=0, total=0)

    assert batch_non_nan_mask.shape == batch_samples.shape[:-1]
    assert batch_samples.ndim == batch_corr_mat.ndim
    assert batch_samples.shape[0] == batch_corr_mat.shape[0]
    assert batch_corr_mat.shape[-1] == batch_corr_mat.shape[-2]

    if batch_samples.ndim == 2:
        batch_samples = batch_samples[None]
        batch_non_nan_mask = batch_non_nan_mask[None]
        batch_corr_mat = batch_corr_mat[None]

    tracked_route_points_indices = []
    for i_sample, (sample, non_nan_mask, corr_mat) in enumerate(
        zip(batch_samples, batch_non_nan_mask, batch_corr_mat)
    ):
        points = sample[non_nan_mask]
        corr_mat = corr_mat[non_nan_mask][:, non_nan_mask]
        ts = points[:, t_index]
        route_ids = points[:, route_id_index]
        _true_target_mask = route_ids > 0
        if batch_target_masks is not None:
            _target_mask = batch_target_masks[i_sample][non_nan_mask]
        else:
            _target_mask = _true_target_mask
        fake_circle_ids = np.ceil(
            (ts - ts[0]) / 10
        )
        _tracked_route_points_indices = multicirc_lookback_tracking(
            corr_mat,
            points[:, [t_index, x_index, y_index]],
            circle_ids=(fake_circle_ids).astype(int),
            labels=_target_mask,
            dist_thresh=15e3,
            lookback_n_circles=3,
            reduce_fn_name="max"
        )

        _asso_stat = associating_accuracy(
            _tracked_route_points_indices,
            route_ids,
            labels=_true_target_mask
        )
        asso_stat += _asso_stat
        tracked_route_points_indices.append(_tracked_route_points_indices)

    return asso_stat, tracked_route_points_indices


def batch_evaluation(
    batch_samples: Union[torch.Tensor, np.ndarray],
    batch_labels_expanded: Union[torch.Tensor, np.ndarray],
    non_nan_points_mask: Union[torch.Tensor, np.ndarray],
    true_target_mask: Union[torch.Tensor, np.ndarray],
    batch_model_probs: Union[None, torch.Tensor, np.ndarray],
    batch_model_regress_output: Union[torch.Tensor, np.ndarray],
    batch_corr_mat: Union[torch.Tensor, np.ndarray],
    area_inverse_transforms: np.ndarray,
    area_borders: np.ndarray,
    regress_feature_augmenter: FeatureAugmenter,
    feature_indices: RealRadarRouteClusterFeatureColumnIndices,
    regress_feature_indices: Tuple[int, ...],
    eval_asso_accu: bool = True,
    eval_reg_res: bool = True,
    device: Optional[torch.device] = None,
    probs_from_regression: bool = False,
    r_thresh: float = 1000,
    n_extra_loc_points: int = 0
) -> Tuple[
    "EvalMetrics",
    Optional[np.ndarray],
    np.ndarray,
    List[TrackedRoutePointsIndices]
]:
    """
    Perform evaluation on a batch of data.

    Args:
        batch_samples: shape=(batch_size, sample_len, n_features)
        batch_labels_expanded: shape=(batch_size, sample_len, n_expanded_labels)
        non_nan_points_mask: shape=(batch_size, sample_len)
        true_target_mask: shape=(batch_size, sample_len)
        batch_model_probs: shape=(batch_size, sample_len, 2) or None.
        batch_model_regress_output: shape=(batch_size, sample_len, n_regress_output)
        batch_corr_mat: shape=(batch_size, sample_len, sample_len)
        area_inverse_transforms: shape=(batch_size, 2, 2)
        area_borders: shape=(batch_size, 4 [x_min, x_max, y_min, y_max])
        regress_feature_augmenter: Regress feature augmenter instance.
        feature_indices: Indices of basic features.
        regress_feature_indices: Indices of features to be regressed.
        eval_asso_accu: Whether association accuracy will be evaluated.
        eval_reg_res: Whether regression residual will be evaluated.
        device: Torch device to use.
        probs_from_regression: Whether to evaluate target probabilities
            from regression results.
        r_thresh: Distance threshold for classifying points from regression
            resuls.
        n_extra_loc_points: The number of extra location points in the route
            to be extracted.

    Returns:
        A tuple of:
            - Evaluation metrics.
            - Probabilities of points classified as false points and targets,
              shape=(batch_size, sample_len, 2 [prob(fp), prob(tgt)]).
              If :attr:`probs_from_regression` is true, it will be calculated
              from regression results; otherwise :attr:`batch_model_probs`
              will be returned directly.
            - Predicted start-end (x,y) coordinate of each true target, shape=
              (batch_size, sample_len, 2 * (2 + n_extra_loc_points)
              [x_start, y_start, x_end, y_end, x_loc1, y_loc1, ...]).
              When :attr:`eval_reg_res` is true, this will be an all-NaN array.
            - Point indices of tracked routes of each sample.
              When :attr:`eval_asso_accu` is true, this will be an all-NaN array.

    """
    batch_size = batch_samples.shape[0]
    sample_len = batch_samples.shape[1]
    area_borders = np.tile(np.expand_dims(area_borders, 1), [1, sample_len, 1])  # shape=(batch_size, sample_len, 4)
    false_point_mask = non_nan_points_mask & (~true_target_mask)
    min_x, max_x, min_y, max_y = area_borders[true_target_mask].T  # 4 arrays of shape=(n_pnt,)
    min_x_fspnt, max_x_fspnt, min_y_fspnt, max_y_fspnt = area_borders[false_point_mask].T  # shape=(n_fspnt, 4)

    x_norm_factor = max_x - min_x  # shape=(n_pnt,)
    y_norm_factor = max_y - min_y  # shape=(n_pnt,)
    x_norm_factor_fspnt = max_x_fspnt - min_x_fspnt  # shape=(n_fspnt,)
    y_norm_factor_fspnt = max_y_fspnt - min_y_fspnt  # shape=(n_fspnt,)

    batch_rs = batch_samples[:, :, feature_indices.r]

    xy_indices_in_regress_output = np.array([regress_feature_indices.index(feature_indices.x),
                                             regress_feature_indices.index(feature_indices.y)])
    xy_start_end_indices_in_regress_output = np.concatenate(
        [xy_indices_in_regress_output + i * len(regress_feature_indices)
         for i in range(2 + n_extra_loc_points)],
        axis=-1
    )

    batch_samples_tn, batch_samples_arr = \
        get_tensor_array_pair(batch_samples, device=device)
    batch_labels_expanded_tn, batch_labels_expanded_arr = \
        get_tensor_array_pair(batch_labels_expanded, device=device)
    if batch_model_probs is not None:
        batch_model_probs_tn, batch_model_probs_arr = \
            get_tensor_array_pair(batch_model_probs, device=device)
    batch_model_regress_output_tn, batch_model_regress_output_arr \
        = get_tensor_array_pair(batch_model_regress_output, device=device)
    batch_corr_mat_tn, batch_corr_mat_arr = \
        get_tensor_array_pair(batch_corr_mat, device=device)
    true_target_mask_tn, true_target_mask_arr = \
        get_tensor_array_pair(true_target_mask, device=device)
    batch_rs_tn, batch_rs_arr = \
        get_tensor_array_pair(batch_rs, device=device)
    non_nan_points_mask_tn, non_nan_points_mask_arr = \
        get_tensor_array_pair(non_nan_points_mask, device=device)

    # If regressed features are augmented, use their augmentation components to recover
    # these features:
    endpoints_preds_list: List[torch.Tensor] = []
    if regress_feature_augmenter is not None:
        target_preds_aug_in_batch_tn = \
            batch_model_regress_output_tn[:, :, len(regress_feature_indices) * (2 + n_extra_loc_points):]
        target_preds_raw_in_batch_tn = \
            batch_model_regress_output_tn[:, :, :len(regress_feature_indices) * (2 + n_extra_loc_points)]
        for i in range(2 + n_extra_loc_points):
            col_start = i * regress_feature_augmenter.n_aug_features
            col_end = col_start + regress_feature_augmenter.n_aug_features
            _preds = regress_feature_augmenter.recover(
                target_preds_aug_in_batch_tn[:, :, col_start:col_end],
                indices_of_feat_to_recover=[feature_indices.x, feature_indices.y],
                # The augmented encoding determines the value only modulo 1;
                # the integer part is provided by the model's non-augmented
                # (raw) predictions of the same coordinates. This matters for
                # e.g. the NEXT point labels, which can be out of [0, 1].
                integer_hint=target_preds_raw_in_batch_tn[
                    ..., xy_indices_in_regress_output + i * len(regress_feature_indices)
                ]
            )
            endpoints_preds_list.append(_preds)
    else:
        for i in range(2 + n_extra_loc_points):
            _preds = \
                batch_model_regress_output_tn[:, :, xy_indices_in_regress_output + i * len(regress_feature_indices)]
            endpoints_preds_list.append(_preds)
    endpoints_coord_preds = torch.concatenate(endpoints_preds_list, dim=-1)
    # normalized (x_start, y_start, x_end, y_end, x_loc1, y_loc1, ...)
    # shape=(batch_size, sample_len, 2 * (2 + n_extra_loc_points))

    x_min, x_max, y_min, y_max = area_borders.T  # shape=(sample_len, batch_size)
    x_norm_factor_all = x_max - x_min
    y_norm_factor_all = y_max - y_min
    norm_factor_all = torch.tensor(
        np.repeat(
            np.expand_dims(
                np.array([x_norm_factor_all, y_norm_factor_all]).T,
                # shape=(batch_size, sample_len, 2)
                -2
            ),
            2 + n_extra_loc_points, axis=-2
            # shape=(batch_size, sample_len, 2 + n_extra_loc_points, 2)
        ).reshape(batch_size, sample_len, -1),
        device=device
    )  # shape=(batch_size, sample_len, 2 * (2 + n_extra_loc_points))
    endpoints_coord_preds_denorm = (
        endpoints_coord_preds.clone()
        * norm_factor_all
        + torch.tensor(
            np.repeat(
                np.expand_dims(
                    np.array([x_min, y_min]).T,
                    # shape=(batch_size, sample_len, 2)
                    -2
                ),
                2 + n_extra_loc_points, axis=-2
                # shape=(batch_size, sample_len, 2 + n_extra_loc_points, 2)
            ).reshape(batch_size, sample_len, -1),
            device=device
        )
    )  # shape=(batch_size, sample_len, 2 * (2 + n_extra_loc_points))

    if probs_from_regression:
        # TODO: this function may be replaced with better implementation
        target_probs = target_prob_from_regression(
            points_xy=batch_samples_arr[..., [rd.PointFeatureIndex.X, rd.PointFeatureIndex.Y]],
            points_xy_start=endpoints_coord_preds_denorm[..., :2].cpu().numpy()
                            @ area_inverse_transforms,
            points_xy_end=endpoints_coord_preds_denorm[..., 2:4].cpu().numpy()
                          @ area_inverse_transforms,
            threshold=r_thresh
        )
        probs = np.stack(
            [1 - target_probs, target_probs], axis=-1
        )  # shape=(batch_size, sample_len, 2)
        batch_probs_tn, batch_probs_arr = get_tensor_array_pair(probs)
    else:
        if batch_model_probs is not None:
            batch_probs_tn = batch_model_probs_tn
            batch_probs_arr = batch_model_probs_arr
        else:
            batch_probs_tn, batch_probs_arr = None, None

    # Evaluate the accuracy and recall of classification
    if batch_model_probs is not None:
        (
            (classify_accuracy_stat, classify_recall_stat),
            classify_accuracy_stat_by_r_regions
        ) = eval_classify_metrics(
            batch_probs_tn[:, :, 1],
            true_target_mask_tn,
            batch_rs_tn,
            non_nan_points_mask_tn,
            region_borders=((0, 50e3), (50e3, 500e3))
        )
        (classify_accuracy_stat_near, classify_recall_stat_near) = \
            classify_accuracy_stat_by_r_regions[0]
        (classify_accuracy_stat_remote, classify_recall_stat_remote) = \
            classify_accuracy_stat_by_r_regions[1]
    else:  # fill in empty values
        classify_accuracy_stat = AccuracyStatistic(0, 0)
        classify_recall_stat = AccuracyStatistic(0, 0)
        classify_accuracy_stat_near = AccuracyStatistic(0, 0)
        classify_recall_stat_near = AccuracyStatistic(0, 0)
        classify_accuracy_stat_remote = AccuracyStatistic(0, 0)
        classify_recall_stat_remote = AccuracyStatistic(0, 0)

    # Evaluate the residuals of regression
    if eval_reg_res:
        target_endpoints_coord_labels_in_batch = batch_labels_expanded_tn[true_target_mask]
        target_endpoints_coord_preds_in_batch = endpoints_coord_preds[true_target_mask]
        fspnt_endpoints_coord_labels_in_batch = batch_labels_expanded_tn[false_point_mask]
        fspnt_endpoints_coord_preds_in_batch = endpoints_coord_preds[false_point_mask]

        regress_residual_stat, regress_residual_stat_by_r_regions = \
            eval_regress_metrics(
                target_endpoints_coord_preds_in_batch,
                target_endpoints_coord_labels_in_batch[:, xy_start_end_indices_in_regress_output],
                torch.tensor(
                    np.repeat(
                        np.expand_dims(
                            np.array([x_norm_factor, y_norm_factor]).T,
                            # shape=(batch_size, sample_len, 2)
                            -2
                        ),
                        2 + n_extra_loc_points, axis=-2
                        # shape=(batch_size, sample_len, 2 + n_extra_loc_points, 2)
                    ).reshape(x_norm_factor.shape[0], 2 * (2 + n_extra_loc_points)),
                    device=device
                ),  # shape=(n_pnt, 2*(2+n_extra_loc_points))
                rs=batch_rs_tn[true_target_mask],
                region_borders=((0, 50e3), (50e3, 500e3))
            )
        regress_residual_stat_false_points, regress_residual_stat_fspnt_by_r_regions = \
            eval_regress_metrics(
                fspnt_endpoints_coord_preds_in_batch,
                fspnt_endpoints_coord_labels_in_batch[:, xy_start_end_indices_in_regress_output],
                torch.tensor(
                    np.repeat(
                        np.expand_dims(
                            np.array([x_norm_factor_fspnt, y_norm_factor_fspnt]).T,
                            # shape=(batch_size, sample_len, 2)
                            -2
                        ),
                        2 + n_extra_loc_points, axis=-2
                        # shape=(batch_size, sample_len, 2 + n_extra_loc_points, 2)
                    ).reshape(x_norm_factor_fspnt.shape[0], 2 * (2 + n_extra_loc_points)),
                    device=device
                ),  # shape=(n_fspnt, 2*(2+n_extra_loc_points))
                rs=batch_rs_tn[false_point_mask],
                region_borders=((0, 50e3), (50e3, 500e3))
            )
        regress_residual_stat_near = regress_residual_stat_by_r_regions[0]
        regress_residual_stat_remote = regress_residual_stat_by_r_regions[1]
        regress_residual_stat_false_points_near = regress_residual_stat_fspnt_by_r_regions[0]
        regress_residual_stat_false_points_remote = regress_residual_stat_fspnt_by_r_regions[1]
    else:
        regress_residual_stat = RegressResidualStatistic(
            np.full(4, fill_value=np.nan), 0
        )
        regress_residual_stat_near = RegressResidualStatistic(
            np.full(4, fill_value=np.nan), 0
        )
        regress_residual_stat_remote = RegressResidualStatistic(
            np.full(4, fill_value=np.nan), 0
        )
        regress_residual_stat_false_points = RegressResidualStatistic(
            np.full(4, fill_value=np.nan), 0
        )
        regress_residual_stat_false_points_near = RegressResidualStatistic(
            np.full(4, fill_value=np.nan), 0
        )
        regress_residual_stat_false_points_remote = RegressResidualStatistic(
            np.full(4, fill_value=np.nan), 0
        )
        start_end_coord_preds = np.full(
            (*batch_samples.shape[:-1], 4), fill_value=np.nan
        )

    # Evaluate the accuracy of association
    if eval_asso_accu:
        asso_accuracy_stat, tracked_routes = eval_association_metrics(
            batch_samples_arr,
            non_nan_points_mask_arr,
            batch_corr_mat_arr,
            feature_indices.time,
            feature_indices.x,
            feature_indices.y,
            feature_indices.route_id,
            batch_target_masks=(batch_probs_arr[..., 1] > 0.5
                                if batch_probs_arr is not None
                                else None)
        )
    else:
        asso_accuracy_stat = AccuracyStatistic(0, 0)
        tracked_routes = []

    return EvalMetrics(
        class_accu_stat=classify_accuracy_stat,
        asso_accu_stat=asso_accuracy_stat,
        class_recall_stat=classify_recall_stat,
        regress_res_stat=regress_residual_stat,
        class_accu_stat_near=classify_accuracy_stat_near,
        class_recall_stat_near=classify_recall_stat_near,
        regress_res_stat_near=regress_residual_stat_near,
        class_accu_stat_remote=classify_accuracy_stat_remote,
        class_recall_stat_remote=classify_recall_stat_remote,
        regress_res_stat_remote=regress_residual_stat_remote,
        regress_res_stat_false_points=regress_residual_stat_false_points,
        regress_res_stat_false_points_near=regress_residual_stat_false_points_near,
        regress_res_stat_false_points_remote=regress_residual_stat_false_points_remote
    ), batch_probs_arr, endpoints_coord_preds.cpu().numpy(), tracked_routes


@dataclasses.dataclass
class EvalMetrics:
    class_accu_stat: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    asso_accu_stat: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    class_recall_stat: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    regress_res_stat: RegressResidualStatistic = dataclasses.field(default_factory=RegressResidualStatistic)
    regress_res_stat_false_points: RegressResidualStatistic = \
        dataclasses.field(default_factory=RegressResidualStatistic)
    class_accu_stat_near: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    class_recall_stat_near: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    regress_res_stat_near: RegressResidualStatistic = dataclasses.field(default_factory=RegressResidualStatistic)
    regress_res_stat_false_points_near: RegressResidualStatistic = \
        dataclasses.field(default_factory=RegressResidualStatistic)
    class_accu_stat_remote: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    class_recall_stat_remote: AccuracyStatistic = dataclasses.field(default_factory=AccuracyStatistic)
    regress_res_stat_remote: RegressResidualStatistic = dataclasses.field(default_factory=RegressResidualStatistic)
    regress_res_stat_false_points_remote: RegressResidualStatistic = \
        dataclasses.field(default_factory=RegressResidualStatistic)

    def __add__(self, other):
        return EvalMetrics(
            class_accu_stat=self.class_accu_stat + other.class_accu_stat,
            asso_accu_stat=self.asso_accu_stat + other.asso_accu_stat,
            class_recall_stat=self.class_recall_stat + other.class_recall_stat,
            regress_res_stat=self.regress_res_stat + other.regress_res_stat,
            class_accu_stat_near=self.class_accu_stat_near + other.class_accu_stat_near,
            class_recall_stat_near=self.class_recall_stat_near + other.class_recall_stat_near,
            regress_res_stat_near=self.regress_res_stat_near + other.regress_res_stat_near,
            class_accu_stat_remote=self.class_accu_stat_remote + other.class_accu_stat_remote,
            class_recall_stat_remote=self.class_recall_stat_remote + other.class_recall_stat_remote,
            regress_res_stat_remote=self.regress_res_stat_remote + other.regress_res_stat_remote,
            regress_res_stat_false_points=self.regress_res_stat_false_points + other.regress_res_stat_false_points,
            regress_res_stat_false_points_near=self.regress_res_stat_false_points_near + other.regress_res_stat_false_points_near,
            regress_res_stat_false_points_remote=self.regress_res_stat_false_points_remote + other.regress_res_stat_false_points_remote,
        )

    def __iadd__(self, other):
        self.class_accu_stat += other.class_accu_stat
        self.asso_accu_stat += other.asso_accu_stat
        self.class_recall_stat += other.class_recall_stat
        self.regress_res_stat += other.regress_res_stat
        self.class_accu_stat_near += other.class_accu_stat_near
        self.class_recall_stat_near += other.class_recall_stat_near
        self.regress_res_stat_near += other.regress_res_stat_near
        self.class_accu_stat_remote += other.class_accu_stat_remote
        self.class_recall_stat_remote += other.class_recall_stat_remote
        self.regress_res_stat_remote += other.regress_res_stat_remote
        self.regress_res_stat_false_points += other.regress_res_stat_false_points
        self.regress_res_stat_false_points_near += other.regress_res_stat_false_points_near
        self.regress_res_stat_false_points_remote += other.regress_res_stat_false_points_remote
        return self

    def to_dict(self) -> dict:
        return {
            "class_accu_stat": self.class_accu_stat.to_dict(),
            "asso_accu_stat": self.asso_accu_stat.to_dict(),
            "class_recall_stat": self.class_recall_stat.to_dict(),
            "regress_res_stat": self.regress_res_stat.to_dict(),
            "class_accu_stat_near": self.class_accu_stat_near.to_dict(),
            "class_recall_stat_near": self.class_recall_stat_near.to_dict(),
            "regress_res_stat_near": self.regress_res_stat_near.to_dict(),
            "class_accu_stat_remote": self.class_accu_stat_remote.to_dict(),
            "class_recall_stat_remote": self.class_recall_stat_remote.to_dict(),
            "regress_res_stat_remote": self.regress_res_stat_remote.to_dict(),
            "regress_res_stat_false_points": self.regress_res_stat_false_points.to_dict(),
            "regress_res_stat_false_points_near": self.regress_res_stat_false_points_near.to_dict(),
            "regress_res_stat_false_points_remote": self.regress_res_stat_false_points_remote.to_dict(),
        }

    @classmethod
    def from_dict(self, d: dict) -> "EvalMetrics":
        return EvalMetrics(
            class_accu_stat=AccuracyStatistic.from_dict(d["class_accu_stat"]),
            asso_accu_stat=AccuracyStatistic.from_dict(d["asso_accu_stat"]),
            class_recall_stat=AccuracyStatistic.from_dict(d["class_recall_stat"]),
            regress_res_stat=RegressResidualStatistic.from_dict(d["regress_res_stat"]),
            class_accu_stat_near=AccuracyStatistic.from_dict(d["class_accu_stat_near"]),
            class_recall_stat_near=AccuracyStatistic.from_dict(d["class_recall_stat_near"]),
            regress_res_stat_near=RegressResidualStatistic.from_dict(d["regress_res_stat_near"]),
            class_accu_stat_remote=AccuracyStatistic.from_dict(d["class_accu_stat_remote"]),
            class_recall_stat_remote=AccuracyStatistic.from_dict(d["class_recall_stat_remote"]),
            regress_res_stat_remote=RegressResidualStatistic.from_dict(d["regress_res_stat_remote"]),
            regress_res_stat_false_points=RegressResidualStatistic.from_dict(d["regress_res_stat_false_points"]),
            regress_res_stat_false_points_near=RegressResidualStatistic.from_dict(
                d["regress_res_stat_false_points_near"]),
            regress_res_stat_false_points_remote=RegressResidualStatistic.from_dict(
                d["regress_res_stat_false_points_remote"])
        )

    def to_string(self):
        return (
            f"Classification accuracy: "
            f"{self.class_accu_stat.correct}/{self.class_accu_stat.total} = "
            f"{self.class_accu_stat.accuracy * 100:.2f}% "
            f"(near: "
            f"{self.class_accu_stat_near.correct}/{self.class_accu_stat_near.total} = "
            f"{self.class_accu_stat_near.accuracy * 100:.2f}%, "
            f"remote: "
            f"{self.class_accu_stat_remote.correct}/{self.class_accu_stat_remote.total} = "
            f"{self.class_accu_stat_remote.accuracy * 100:.2f}%).  "
            f"Association accuracy: "
            f"{self.asso_accu_stat.correct}/{self.asso_accu_stat.total} = "
            f"{self.asso_accu_stat.accuracy * 100:.2f}%.  "
            f"Classification recall: "
            f"{self.class_recall_stat.correct}/{self.class_recall_stat.total} = "
            f"{self.class_recall_stat.accuracy * 100:.2f}% "
            f"(near: "
            f"{self.class_recall_stat_near.correct}/{self.class_recall_stat_near.total} = "
            f"{self.class_recall_stat_near.accuracy * 100:.2f}%, "
            f"remote: "
            f"{self.class_recall_stat_remote.correct}/{self.class_recall_stat_remote.total} = "
            f"{self.class_recall_stat_remote.accuracy * 100:.2f}%).  "
            f"Overall regression residual: "
            f"{np.round((self.regress_res_stat + self.regress_res_stat_false_points).avg_res, 3)} "
            f"(near: {np.round((self.regress_res_stat_near + self.regress_res_stat_false_points_near).avg_res, 3)}, "
            f"remote: {np.round((self.regress_res_stat_remote + self.regress_res_stat_false_points_remote).avg_res, 3)}).  "
            f"True targets regression residual: "
            f"{np.round(self.regress_res_stat.avg_res, 3)} "
            f"(near: {np.round(self.regress_res_stat_near.avg_res, 3)}, "
            f"remote: {np.round(self.regress_res_stat_remote.avg_res, 3)}).  "
            f"False points regression residual: "
            f"{np.round(self.regress_res_stat_false_points.avg_res, 3)} "
            f"(near: {np.round(self.regress_res_stat_false_points_near.avg_res, 3)}, "
            f"remote: {np.round(self.regress_res_stat_false_points_remote.avg_res, 3)}).  "
        )
