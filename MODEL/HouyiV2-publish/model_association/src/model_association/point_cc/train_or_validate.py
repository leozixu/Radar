# -*- coding: utf-8 -*-

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

import multiprocessing
import queue
import time
from importlib import resources
from typing import DefaultDict

import torch
import torch.distributed as dist
from model_association.loss import CCLoss
from model_association.point_cc.config import CollectedDataConfig
from model_association.point_cc.data import RealRadarRouteClusterFeatureColumnIndices, process_model_data
from model_association.point_cc.feature_augment import FeatureAugmenter
from model_association.point_cc.metrics import batch_evaluation
from model_association.point_cc.model import *
from model_association.point_cc.statistic import RegressResidualStatistic, AccuracyStatistic
from model_association.route_point_data_processor import HorizontalCoordPaddedFanSplitter
from model_association.utils import *
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

default_bert_model_config_filepath = resources.files("model_association.point_cc") \
    .joinpath("bert_model_config.json")
default_qwen2_model_config_filepath = resources.files("model_association.point_cc") \
    .joinpath("qwen2_model_config.json")


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.open()


def ddp_reduce_accuracy_stats(
    accuracy_stats: List[AccuracyStatistic],
    device: torch.device
) -> AccuracyStatistic:
    n_correct_total = torch.tensor(np.sum(
        [_stat.correct for _stat in accuracy_stats]
    ), device=device)
    n_total = torch.tensor(np.sum(
        [_stat.total for _stat in accuracy_stats]
    ), device=device)

    # Reduce all ranks
    n_correct_total = ddp_reduce(n_correct_total, op=torch.distributed.ReduceOp.SUM).item()
    n_total = ddp_reduce(n_total, op=torch.distributed.ReduceOp.SUM).item()

    avg_stat = AccuracyStatistic(
        correct=n_correct_total,
        total=n_total
    )

    return avg_stat


def ddp_reduce_residual_stats(
    residual_stats: list[RegressResidualStatistic],
    device: torch.device
) -> RegressResidualStatistic:
    regress_res_total = torch.tensor(np.sum(
        [_res_stat.total_res for _res_stat in residual_stats],
        axis=0
    ), device=device)  # shape=(n_regress_features,)
    regress_res_all_count = torch.tensor(np.sum(
        [_res_stat.count for _res_stat in residual_stats]
    ), device=device)  # shape=(n_regress_features,)

    # Reduce all ranks
    regress_res_total = ddp_reduce(regress_res_total, op=torch.distributed.ReduceOp.SUM).cpu().numpy()
    regress_res_all_count = ddp_reduce(regress_res_all_count, op=torch.distributed.ReduceOp.SUM).cpu().numpy()
    avg_regress_residual = RegressResidualStatistic(
        total_res=regress_res_total,
        count=regress_res_all_count
    )
    return avg_regress_residual


class TrainOrValidateMetrics(BaseModel):
    epoch: int
    time_consumed: float
    loss: float
    classify_loss: float
    circle_loss: float
    regress_loss: float
    classify_accuracy: float
    classify_recall: float


def weight_metric_score(metrics: TrainOrValidateMetrics) -> float:
    return 0.5 * metrics.classify_accuracy + 0.5 * metrics.classify_recall


INPUT_QUEUE_EPOCH_END_SIGNAL = "EPOCH_LAST_BATCH"
RESULT_QUEUE_EPOCH_END_SIGNAL = "EPOCH_LAST_RESULT"


def metrics_worker(
    input_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    area_inverse_transforms: np.ndarray,
    area_borders: np.ndarray,
    regress_feature_indices: Tuple[int, ...],
    regress_feature_augmenter: FeatureAugmenter,
    feature_indices: RealRadarRouteClusterFeatureColumnIndices,
    logger_name: str,
    logger_file_name: Path,
    log_level,
    only_regress: bool,
    r_thresh: Optional[float]
):
    logger = setup_logger(
        logger_name,
        log_level=log_level,
        log_file_name=logger_file_name
    )
    pp_checker = ParentProcessChecker()
    while True:
        try:
            data = input_queue.get()
            logger.debug(
                f"Received {'non-empty data' if data is not None else 'None (termination signal)'}."
            )
        except (multiprocessing.TimeoutError, queue.Empty):
            if not pp_checker.parent_proc_ok():
                logger.info(f"Parent process of this metrics process has ended. Exiting.")
                return
            continue

        if data is None:
            logger.info(f"Metrics process exits.")
            return

        if data == INPUT_QUEUE_EPOCH_END_SIGNAL:
            logger.info(f"Input queue received the last batch signal, "
                        f"sending the last result signal...")
            result_queue.put_nowait(RESULT_QUEUE_EPOCH_END_SIGNAL)

            continue

        (i_batch,
         epoch_log_prefix,
         model_probs,
         model_output,
         batch_corr_mat,
         batch_samples,
         batch_area_ids,
         batch_labels_expanded,
         true_target_mask,
         non_nan_points_mask,
         loss,
         classify_loss,
         regress_loss) = data

        try:
            batch_eval_start_time = time.perf_counter()

            metrics, _, _, _ = batch_evaluation(
                batch_samples,
                batch_labels_expanded,
                non_nan_points_mask,
                true_target_mask,
                model_probs,
                model_output,
                batch_corr_mat,
                area_inverse_transforms,
                area_borders,
                regress_feature_augmenter,
                feature_indices,
                regress_feature_indices,
                eval_asso_accu=False,
                eval_reg_res=False,
                probs_from_regression=only_regress,
                r_thresh=r_thresh
            )

            batch_eval_end_time = time.perf_counter()

            result_queue.put({
                "i_batch": i_batch,
                "loss": loss,
                "classify_loss": classify_loss,
                "regress_loss": regress_loss,
                "classify_accuracy_stat": metrics.class_accu_stat,
                "asso_accuracy_stat": metrics.asso_accu_stat,
                "classify_recall_stat": metrics.class_recall_stat,
                "regress_residual_stat": metrics.regress_res_stat,
                "classify_accuracy_stat_near": metrics.class_accu_stat_near,
                "classify_recall_stat_near": metrics.class_recall_stat_near,
                "regress_residual_stat_near": metrics.regress_res_stat_near,
                "classify_accuracy_stat_remote": metrics.class_accu_stat_remote,
                "classify_recall_stat_remote": metrics.class_recall_stat_remote,
                "regress_residual_stat_remote": metrics.regress_res_stat_remote,
                "regress_residual_false_points_stat": metrics.regress_res_stat_false_points,
                "regress_residual_false_points_stat_near": metrics.regress_res_stat_false_points_near,
                "regress_residual_false_points_stat_remote": metrics.regress_res_stat_false_points_remote,
            })

            logger.info(
                f"{epoch_log_prefix} "
                f"batch: {i_batch + 1}  "
                f"Loss: {loss:.4f} "
                f"(classify loss: {classify_loss:.4f}, "
                f"regress loss: {regress_loss:.4f})  "
                f"{metrics.to_string()}  "
                f"batch metric eval time: {batch_eval_end_time - batch_eval_start_time:.2f}s"
            )

        except Exception as e:
            logger.error(
                f"{epoch_log_prefix} batch {i_batch + 1}  "
                f"Error on estimating metrics. {traceback.format_exc()}"
            )
            continue


def train_or_validate(
    model: Union[BertCCModel, Qwen2CCModel, DistributedDataParallel],
    dataloader: DataLoader,
    feature_indices: RealRadarRouteClusterFeatureColumnIndices,
    input_feature_indices: Tuple[int, ...],
    regress_feature_indices: Tuple[int, ...],
    extra_n_in_features: int,
    input_feature_augmenter: Optional[FeatureAugmenter],
    regress_feature_augmenter: Optional[FeatureAugmenter],
    collected_data_config: CollectedDataConfig,
    point_splitters: DefaultDict[int, HorizontalCoordPaddedFanSplitter],
    loss_on_uncertain_points: bool,
    loss_on_track_without_next: bool,
    reg_the_last_but_one_point: bool,
    epoch: int,
    is_training: bool,
    device: torch.device,
    logger: logging.Logger,
    use_attn_as_corr: bool,
    use_ddp: bool = False,
    loss_fn: Optional[CCLoss] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    num_batches: Optional[int] = None,
    model_save_dir: Optional[Path] = None,
    log_interval: int = 100
) -> TrainOrValidateMetrics:
    if use_ddp:
        rank = device.index
    else:
        rank = 0

    if is_training:
        model.train()
    else:
        model.eval()

    only_regress = (not loss_fn.uncertainty_weighting) and (loss_fn.classify_coeff == 0)

    if optimizer is None and is_training:
        raise ValueError(
            "Optimizer is not provided under training mode."
        )

    num_batches = min(num_batches or np.inf, len(dataloader))
    if rank == 0:
        logger.info(
            "Total samples in dataset: %d.  Number of batches: %d.",
            len(dataloader.dataset), num_batches
        )

    epoch_log_prefix = f"[Epoch {epoch} | {'Training' if is_training else 'Testing '}]"
    epoch_start_time = time.perf_counter()

    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)

    i_batch = 0
    loss_all = []
    classify_loss_all = []
    circle_loss_all = []
    regress_loss_all = []
    classify_accuracy_all = []
    classify_recall_all = []

    with torch.set_grad_enabled(is_training):
        if use_ddp:
            dist.barrier()

        batch_start_time = time.perf_counter()

        for i_batch, batch_data in enumerate(dataloader):
            batch_samples, batch_area_ids, _ = batch_data
            data_dict = process_model_data(
                batch_samples,
                batch_area_ids,
                input_feature_indices,
                regress_feature_indices,
                extra_n_in_features,
                input_feature_augmenter,
                regress_feature_augmenter,
                collected_data_config,
                point_splitters,
                reg_the_last_but_one_point,
                device=device,
                loss_on_track_without_next=loss_on_track_without_next
            )
            input_data = data_dict["model_input"]
            batch_data_normalized = data_dict["data_normalized"]
            true_target_labels = data_dict["true_target_labels"]
            r_normalization_factors = data_dict["normalization_factors"]["r"]
            batch_labels_expanded = data_dict["regress_labels"]
            non_nan_points_mask = data_dict["non_nan_points_mask"]

            probs, output, _ = model(input_data)

            if not loss_on_uncertain_points:
                # In following processing for loss evaluation, uncertain points
                # will be all masked out (just like NaN points):
                non_nan_points_mask = non_nan_points_mask & (
                    batch_data_normalized[:, :, feature_indices.route_id] >= 0
                    # remove uncertain points (track ID = -1)
                )
            pred_target_probs_an = probs[:, :, 1][non_nan_points_mask]
            circle_point_mask = batch_data_normalized[:, :, feature_indices.focusing_point_flag] > 0
            pred_circle_probs_an = probs[:, :, 1][circle_point_mask]
            true_circle_labels_an = true_target_labels[circle_point_mask].float()
            true_target_labels_an = true_target_labels[non_nan_points_mask].float()

            points_an = batch_data_normalized[non_nan_points_mask]
            near_region_points_an_mask = \
                points_an[:, feature_indices.r] < 50e3 \
                / torch.broadcast_to(
                    torch.tensor(r_normalization_factors, device=points_an.device)[:, None],
                    batch_data_normalized.shape[:2]
                )[non_nan_points_mask]

            if only_regress:
                loss, terms_dict = loss_fn(
                    true_target_labels_an,
                    None,
                    near_region_points_an_mask,
                    pred_circle_probs_an,
                    true_circle_labels_an,
                    output[non_nan_points_mask],
                    batch_labels_expanded[non_nan_points_mask]
                )
                classify_accuracy = torch.tensor(0.0, device=device)
                classify_recall = torch.tensor(0.0, device=device)
            else:
                pred_target_classes_an = pred_target_probs_an > 0.5
                loss, terms_dict = loss_fn(
                    true_target_labels_an,
                    pred_target_probs_an,
                    near_region_points_an_mask,
                    pred_circle_probs_an,
                    true_circle_labels_an,
                    output[non_nan_points_mask],
                    batch_labels_expanded[non_nan_points_mask]
                )
                classify_accuracy = (
                    (pred_target_classes_an == true_target_labels_an).sum().float()
                    / true_target_labels_an.numel()
                )
                classify_recall = (
                    (pred_target_classes_an.bool() & true_target_labels_an.bool()).sum().float()  # TP
                    / true_target_labels_an.sum().item()  # TP + FN
                )

            classify_loss = terms_dict["classify"]
            circle_loss = terms_dict["circle"]
            regress_loss = terms_dict["regress"]

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

                if i_batch % log_interval == 0:
                    if rank == 0:
                        total_norm, max_grad, min_grad, mean_grad, grad_param_ratio = inspect_gradients(model)
                        logger.debug(
                            f"{epoch_log_prefix} batch: {i_batch + 1}/{num_batches}  "
                            f"Gradient statistic: norm={total_norm:.6f}, max={max_grad:.6f}, "
                            f"min={min_grad:.6f}, mean={mean_grad:.6f}, grad/param={grad_param_ratio:.6f}"
                        )
                optimizer.step()

                if scheduler:
                    scheduler.step()

            est_acc_start_time = time.perf_counter()

            loss_all.append(loss.item())
            classify_loss_all.append(classify_loss.item())
            circle_loss_all.append(circle_loss.item())
            regress_loss_all.append(regress_loss.item())
            classify_accuracy_all.append(classify_accuracy.item())
            classify_recall_all.append(classify_recall.item())

            # For logging batches, put data into metrics input queue for evaluation:
            if i_batch % log_interval == 0:
                if rank == 0:
                    if scheduler:
                        logger.info(f"{epoch_log_prefix} batch: {i_batch + 1}/{num_batches}  "
                                    f"Scheduler steps: lr = {scheduler.get_lr()} "
                                    f"(last step = {scheduler.last_epoch})")

                    logger.info(f"{epoch_log_prefix} batch: {i_batch + 1}/{num_batches}  "
                                f"loss = {loss.item():.4f}  "
                                f"(classify loss = {classify_loss.item():.4f}, "
                                f"circle loss = {circle_loss.item():.4f}, "
                                f"regress loss = {regress_loss.item():.4f})  "
                                f"classify accu. = {classify_accuracy.item():.4f}  "
                                f"classify recall = {classify_recall.item():.4f}  "
                                f"Batch time: {est_acc_start_time - batch_start_time:.2f}s.")

            batch_start_time = time.perf_counter()

            if i_batch >= num_batches - 1:
                break

    logger.info(
        f"{epoch_log_prefix} "
        f"Epoch finished: batch count = {i_batch + 1}"
    )

    epoch_end_time = time.perf_counter()

    # Evaluate average metrics over this epoch on current rank
    avg_loss = torch.tensor(loss_all, device=device).mean()
    avg_classify_loss = torch.tensor(classify_loss_all, device=device).mean()
    avg_circle_loss = torch.tensor(circle_loss_all, device=device).mean()
    avg_regress_loss = torch.tensor(regress_loss_all, device=device).mean()
    avg_classify_accuracy = torch.tensor(classify_accuracy_all, device=device).mean()
    avg_classify_recall = torch.tensor(classify_recall_all, device=device).mean()

    # Reduce all ranks
    avg_loss = ddp_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG).item()
    avg_classify_loss = ddp_reduce(avg_classify_loss, op=torch.distributed.ReduceOp.AVG).item()
    avg_regress_loss = ddp_reduce(avg_regress_loss, op=torch.distributed.ReduceOp.AVG).item()
    avg_classify_accuracy = ddp_reduce(avg_classify_accuracy, op=torch.distributed.ReduceOp.AVG).item()
    avg_classify_recall = ddp_reduce(avg_classify_recall, op=torch.distributed.ReduceOp.AVG).item()

    epoch_time_consumed = epoch_end_time - epoch_start_time

    if rank == 0:
        logger.info(
            f"{epoch_log_prefix} "
            f"avg.loss = {avg_loss:.4f}  "
            f"(avg.classify loss = {avg_classify_loss:.4f}, "
            f"avg.circle loss = {avg_circle_loss:.4f}, "
            f"avg.regress loss = {avg_regress_loss:.4f})  "
            f"avg.classify accu. = {avg_classify_accuracy:.4f}  "
            f"epoch time consumed: {epoch_time_consumed:.4f}s"
        )
    return TrainOrValidateMetrics(
        epoch=epoch,
        time_consumed=epoch_time_consumed,
        loss=avg_loss,
        classify_loss=avg_classify_loss,
        circle_loss=avg_circle_loss,
        regress_loss=avg_regress_loss,
        classify_accuracy=avg_classify_accuracy,
        classify_recall=avg_classify_recall
    )
