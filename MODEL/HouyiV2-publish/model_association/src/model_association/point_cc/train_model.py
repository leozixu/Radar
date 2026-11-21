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

import datetime
import json
import logging
import math
import os
import re
import shutil
import traceback
from collections import OrderedDict, defaultdict
from importlib import resources
from pathlib import Path
from typing import Annotated, Optional, Tuple, Union, Dict, Type, Any, Iterable, List

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as tmp
import typer
from model_association import radar_definition as rd
from model_association.loss import CCLoss
from model_association.optim import WarmupCosineAnnealingLR
from model_association.point_cc.config import (
    CollectedDataConfig, COLLECTED_DATA_CONFIG_FILENAME,
    load_model_config, MODEL_CONFIG_FILENAME, BertCCModelConfig,
    Qwen2CCModelConfig, Houyi2CCModelConfig
)
from model_association.point_cc.data import (
    HDF5Dataset, RealRadarRouteClusterFeatureColumnIndices,
    IndexedDatasetWrapper
)
from model_association.point_cc.feature_augment import FeatureAugmenter
from model_association.point_cc.lora import set_lora_part_requires_grad, report_lora_mount, validate_lora_mount
from model_association.point_cc.model import create_raw_model_from_config
from model_association.point_cc.statistic import RegressResidualStatistic, AccuracyStatistic
from model_association.point_cc.train_or_validate import train_or_validate, TrainOrValidateMetrics, weight_metric_score
from model_association.point_cc.visualization import plot_metric_curve as plot_metric_curve_fn
from model_association.route_point_data_processor import \
    HorizontalCoordPaddedFanSplitter
from model_association.utils import (
    cleanup_ddp, create_dataloader, create_dir, get_host_ip, parse_ddp_env,
    setup_ddp, setup_logger, unwrap_state_dict, unwrap_ddp_model,
    get_num_trainable_parameters
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

default_bert_model_config_filepath = resources.files("model_association.point_cc") \
    .joinpath("bert_model_config.json")
default_qwen2_model_config_filepath = resources.files("model_association.point_cc") \
    .joinpath("qwen2_model_config.json")
default_houyi2_model_config_filepath = resources.files("model_association.point_cc") \
    .joinpath("houyi2_model_config.json")

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]},
                  pretty_exceptions_show_locals=False,
                  pretty_exceptions_short=True)

torch.backends.cuda.matmul.allow_tf32 = True


def create_point_splitters(collected_data_config: CollectedDataConfig) -> Dict[int, HorizontalCoordPaddedFanSplitter]:
    if collected_data_config.default_config:
        point_splitters = defaultdict(lambda: HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            collected_data_config.default_config.normalization_factor.r,
            collected_data_config.default_config.split_azim_width * np.pi / 180,
            collected_data_config.default_config.split_pad_width,
            collected_data_config.default_config.r1
        ))
    else:
        point_splitters = {}

    point_splitters.update({
        rd_type_id: HorizontalCoordPaddedFanSplitter.generate_fully_covering_splitter(
            config.normalization_factor.r or config.max_value.r,
            config.split_azim_width * np.pi / 180,
            config.split_pad_width,
            config.r1
        )
        for rd_type_id, config in collected_data_config.configs.items()
    })
    return point_splitters


@app.command()
def train_model(
    dataset_dir: Annotated[str, typer.Argument(
        help="Directory of processed dataset, which must contain \"train.hdf5\", "
             f"\"test.hdf5\" and \"{COLLECTED_DATA_CONFIG_FILENAME}\" files."
    )],
    weight_save_dir: Annotated[str, typer.Argument(
        help="Directory of saving weights. If not existing, it will be created. "
             "Weights will be saved to 'weight-save-dir/model-YYmmdd_HHMMSS/' "
             "(if --auto-subdir is switched on) or 'weight-save-dir/' (otherwise), "
             "where YYmmdd_HHMMSS is the timestamp of starting training this model."
    )],
    auto_subdir: Annotated[bool, typer.Option(
        help="Whether to create subdirectory labeled with timestamp for model saving automatically."
    )] = True,
    model_config: Annotated[Optional[str], typer.Option(
        help=f"Path to model configuration file. If it is not provided or the file "
             f"does not exist, this program will try to find one at the directory of "
             f"$(dirname <weight_ckpt_dir>)/{MODEL_CONFIG_FILENAME}`; if it cannot be found there "
             f"or `weight_ckpt_dir` is not provided, file {default_bert_model_config_filepath} "
             f"(for BERT backbone model), {default_qwen2_model_config_filepath} "
             f"(for Qwen2 backbone model) or {default_houyi2_model_config_filepath} "
             f"(for Houyi2 backbone model) will be used."
    )] = None,
    freeze_classify_head: Annotated[bool, typer.Option(
        help="If switched on, the classification head is frozen (requires_grad=False) "
             "and the classification/circle loss coefficients are set to 0."
    )] = False,
    freeze_regress_head: Annotated[bool, typer.Option(
        help="If switched on, the regression head is frozen (requires_grad=False) "
             "and the regression loss coefficient is set to 0."
    )] = False,
    freeze_backbone: Annotated[bool, typer.Option(
        help="If switched on, the backbone is frozen (requires_grad=False). No loss "
             "coefficient is affected."
    )] = False,
    only_classify: Annotated[Optional[bool], typer.Option(
        help="[DEPRECATED] Use '--freeze-backbone --freeze-regress-head' instead. "
             "This option will be removed in a future version."
    )] = None,
    only_regress: Annotated[Optional[bool], typer.Option(
        help="[DEPRECATED] Use '--freeze-classify-head' instead. "
             "This option will be removed in a future version."
    )] = None,
    reg_the_last_but_one_point: Annotated[bool, typer.Option(
        help="If true, the last but one point in each track will also be regressed. "
             "This requires datasets also to be generated with --reg-the-last-but-one-point "
             "option. Otherwise the behavior is unexpected."
    )] = True,
    loss_on_uncertain_points: Annotated[Optional[bool], typer.Option(
        help=f"If switched on, losses will also be calculated on uncertain points "
             f"(track ID = -1; if any)."
    )] = False,
    loss_on_track_without_next: Annotated[Optional[bool], typer.Option(
        help="Effective only when --reg-the-last-but-one-point is on. If true, "
             "points of a route whose END point has no NEXT point still "
             "contribute to the loss (only the missing NEXT label itself is "
             "masked out). If false, such whole routes are excluded from the "
             "regression loss."
    )] = True,
    backbone_type: Annotated[
        str, typer.Option(
            help="Backbone type of the model.",
            click_type=click.Choice([
                "bert", "qwen2", "houyi2"
            ]),
            case_sensitive=False
        )
    ] = "bert",
    weight_ckpt_dir: Annotated[Optional[str], typer.Option(
        help="Path to the weight checkpoint directory."
    )] = None,
    ignore_module_pattern: Annotated[List[str], typer.Option(
        help="The pattern of the name of module to be ignored. Can be passed "
             "many times to specify multiple patterns."
    )] = [],
    load_optim_states: Annotated[Optional[bool], typer.Option(
        help="Whether to load checkpoints of optimizer and scheduler. "
             "Only effective when '--weight-ckpt-dir' is given and valid."
    )] = True,
    cuda: Annotated[Optional[bool], typer.Option(
        help="If switched on, use CUDA devices to train model."
    )] = True,
    ddp: Annotated[Optional[bool], typer.Option(
        help="If this option is switched on and \"--cuda\" is switched on, "
             "distributed data parallelism (DDP) will be used for model "
             "training and all visible CUDA devices will be used."
    )] = False,
    lora: Annotated[Optional[bool], typer.Option(
        help="If this option is switched on, LoRA fine-tuning will be performed"
             " on the model (LoRA configuration is supposed to be found in the "
             "config file."
    )] = False,
    lora_weight_dir: Annotated[Optional[str], typer.Option(
        help="Path to the directory of LoRA adapter weights (containing "
             "adapter_config.json). If provided, --lora must be switched on. "
             "This is used for resuming LoRA fine-tuning."
    )] = None,
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
    plot_metric_curve: Annotated[bool, typer.Option(
        help="Whether to plot metric curves. Note that plotted metrics are "
             "INACCURATE since they are estimated from sampled batches."
    )] = False
):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    dataset_dir = Path(dataset_dir)
    weight_save_dir = Path(weight_save_dir)
    weight_save_dir.mkdir(parents=True, exist_ok=True)

    weight_ckpt_path: Optional[Path] = None
    if weight_ckpt_dir is not None:
        if (Path(weight_ckpt_dir) / "model.pth").is_file():
            weight_ckpt_dir = Path(weight_ckpt_dir)
            weight_ckpt_path = weight_ckpt_dir / "model.pth"
        else:
            raise ValueError(
                f"'--weight-ckpt-dir' is given with value {weight_ckpt_dir}, "
                f"but there is not a 'model.pth' file under it. "
                f"Did you incorrectly provided the full path of the .pth file as '--weight-ckpt-dir'?"
            )

    logger_name = "train_model_main"
    if auto_subdir:
        model_weight_save_dir = create_dir(os.path.join(weight_save_dir, "model-" + timestamp))
    else:
        model_weight_save_dir = create_dir(weight_save_dir)
    logger = setup_logger(
        logger_name,
        log_level=log_level,
        log_file_name=model_weight_save_dir / "logs" /
                      f"{logger_name}_{timestamp}.log"
    )

    if lora_weight_dir is not None and not lora:
        msg = "--lora-weight-path requires --lora to be switched on."
        logger.error(msg)
        raise ValueError(msg)

    if backbone_type == "houyi2":
        default_model_config_filepath = default_houyi2_model_config_filepath
        config_type = Houyi2CCModelConfig
    elif backbone_type == "qwen2":
        default_model_config_filepath = default_qwen2_model_config_filepath
        config_type = Qwen2CCModelConfig
    else:
        default_model_config_filepath = default_bert_model_config_filepath
        config_type = BertCCModelConfig

    model_config_obj, model_config_filepath = load_model_config(
        [model_config,
         weight_ckpt_dir.parent / MODEL_CONFIG_FILENAME if weight_ckpt_dir else None,
         default_model_config_filepath],
        config_type=config_type,
        logger=logger
    )
    model_config: Union[Houyi2CCModelConfig, BertCCModelConfig, Qwen2CCModelConfig] = model_config_obj
    if model_config is None:
        raise ValueError("Failed to load model_config.")
    logger.info(f"Use model configuration from file {model_config_filepath}")
    logger.info(f"Model configuration: {model_config}")
    shutil.copy(model_config_filepath, model_weight_save_dir / MODEL_CONFIG_FILENAME)

    logger.info(f"Start training on server: {get_host_ip()}")
    logger.info(f"PID: {os.getpid()}")

    if cuda:
        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise ValueError(
                "There are no available CUDA devices, but \"--cuda\" option is switched on."
            )
    else:
        available_gpus = 0
        ddp = False

    _, _, _, master_addr, master_port, _ = parse_ddp_env()

    if only_classify:
        logger.error(
            "`--only-classify` is deprecated and will be removed in a future version. "
            "Use `--freeze-backbone --freeze-regress-head` instead."
        )
        raise typer.Exit(code=1)
    if only_regress:
        logger.error(
            "`--only-regress` is deprecated and will be removed in a future version. "
            "Use `--freeze-classify-head` instead."
        )
        raise typer.Exit(code=1)

    if freeze_classify_head:
        model_config.loss_coeffs.classify_coeff = 0.
        model_config.loss_coeffs.circle_coeff = 0.
        logger.info("model_config.loss_coeffs.classify_coeff and "
                    "circle_coeff are set to 0.0 since `--freeze-classify-head` is provided.")
    if freeze_regress_head:
        model_config.loss_coeffs.regress_coeff = 0.
        logger.info("model_config.loss_coeffs.regress_coeff is set to 0.0 "
                    "since `--freeze-regress-head` is provided.")

    ignore_module_pattern = tuple(re.compile(p) for p in ignore_module_pattern)

    if ddp and available_gpus > 1:
        logger.info("Use DDP for model training. World size: %d", available_gpus)
        tmp.spawn(
            worker,
            args=(
                available_gpus,
                None,
                master_addr,
                master_port,
                timestamp,
                dataset_dir,
                model_weight_save_dir,
                model_config,
                weight_ckpt_path,
                ignore_module_pattern,
                load_optim_states,
                loss_on_uncertain_points,
                loss_on_track_without_next,
                reg_the_last_but_one_point,
                True,  # use_cuda
                True,  # use_ddp
                log_level,
                lora,
                lora_weight_dir,
                plot_metric_curve,
                freeze_classify_head,
                freeze_regress_head,
                freeze_backbone
            ),
            nprocs=available_gpus,
            join=True
        )
    else:
        logger.info("Training on single %s device.", "CUDA" if cuda else "CPU")
        worker(
            rank=0,
            world_size=1,
            local_rank=0,
            master_addr=master_addr,
            master_port=master_port,
            timestamp=timestamp,
            dataset_dir_path=dataset_dir,
            model_weight_save_dir=model_weight_save_dir,
            model_config=model_config,
            weight_ckpt_path=weight_ckpt_path,
            ignore_module_pattern=ignore_module_pattern,
            load_optim_states=load_optim_states,
            loss_on_uncertain_points=loss_on_uncertain_points,
            loss_on_track_without_next=loss_on_track_without_next,
            reg_the_last_but_one_point=reg_the_last_but_one_point,
            use_cuda=cuda,
            use_ddp=False,
            log_level=log_level,
            use_lora=lora,
            lora_weight_path=lora_weight_dir,
            plot_metric_curve_flag=plot_metric_curve,
            freeze_classify_head=freeze_classify_head,
            freeze_regress_head=freeze_regress_head,
            freeze_backbone=freeze_backbone
        )

    logger.info("Finished all epochs.")


def worker(
    rank: int,
    world_size: int,
    local_rank: Optional[int],  # Under single node mode, this can be None
    master_addr: Optional[str],
    master_port: Optional[int],
    timestamp: str,
    dataset_dir_path: Path,
    model_weight_save_dir: Path,
    model_config: Union[BertCCModelConfig, Houyi2CCModelConfig, Qwen2CCModelConfig],
    weight_ckpt_path: Optional[Path],
    ignore_module_pattern: Iterable[re.Pattern[str]],
    load_optim_states: bool,
    loss_on_uncertain_points: bool,
    loss_on_track_without_next: bool,
    reg_the_last_but_one_point: bool,
    use_cuda: bool,
    use_ddp: bool,
    log_level,
    use_lora: bool = False,
    lora_weight_path: Optional[str] = None,
    plot_metric_curve_flag: bool = False,
    freeze_classify_head: bool = False,
    freeze_regress_head: bool = False,
    freeze_backbone: bool = False
):
    if local_rank is None:
        local_rank = rank

    if use_ddp:
        setup_ddp(rank, world_size, local_rank=local_rank,
                  master_addr=master_addr or "127.0.0.1",
                  master_port=master_port or 29501)

    if world_size == 1:
        logger_name = "train_model"
    else:
        logger_name = f"train_model_rank_{rank}"
    logger = setup_logger(
        logger_name,
        log_level=log_level,
        log_file_name=model_weight_save_dir / "logs" /
                      f"{logger_name}_{timestamp}.log"
    )

    logger.info("Time stamp: %s", timestamp)

    seed = model_config.seed
    if seed is not None:
        seed = seed + rank
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)

    if use_cuda:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device("cpu")

    logger.info(f"Rank {rank} starts training on server: {get_host_ip()}  using device: {device}")
    logger.info(f"PID: {os.getpid()}")

    # Load datasets
    train_dataset_path = dataset_dir_path / "train.hdf5"
    test_dataset_path = dataset_dir_path / "test.hdf5"

    train_dataset = HDF5Dataset(
        hdf5_path=train_dataset_path,
        dset_names=("points_in_areas", "area_ids", "area_masks")
    )
    train_num_samples = model_config.train_num_samples
    logger.info(f"Using train_num_samples: {train_num_samples}")
    total_train_samples = len(train_dataset)
    if train_num_samples is not None and train_num_samples < total_train_samples:
        rng = np.random.RandomState(seed=seed)
        selected_indices = rng.choice(total_train_samples, size=train_num_samples, replace=False)
        selected_indices = sorted(selected_indices)
        logger.info(f"Selected {train_num_samples} samples out of {total_train_samples} train samples.")
        train_dataset = IndexedDatasetWrapper(train_dataset, selected_indices)
    else:
        logger.info(f"Using all {total_train_samples} samples.")

    test_dataset = HDF5Dataset(
        hdf5_path=test_dataset_path,
        dset_names=("points_in_areas", "area_ids", "area_masks")
    )
    test_num_samples = model_config.test_num_samples
    logger.info(f"Using test_num_samples: {test_num_samples}")
    total_test_samples = len(test_dataset)
    if test_num_samples is not None and test_num_samples < total_test_samples:
        rng = np.random.RandomState(seed=seed + 1)
        selected_indices = rng.choice(total_test_samples, size=test_num_samples, replace=False)
        selected_indices = sorted(selected_indices)
        logger.info(f"Selected {test_num_samples} samples out of {total_test_samples} test samples.")
        test_dataset = IndexedDatasetWrapper(test_dataset, selected_indices)
    else:
        logger.info(f"Using all {total_test_samples} samples.")

    train_dataloader, train_sampler = create_dataloader(
        train_dataset,
        rank,
        world_size,
        use_ddp,
        shuffle=True,
        batch_size=model_config.batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    test_dataloader, test_sampler = create_dataloader(
        test_dataset,
        rank,
        world_size,
        use_ddp,
        shuffle=False,
        batch_size=model_config.batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )

    feature_indices_obj = RealRadarRouteClusterFeatureColumnIndices(
        time=rd.PointFeatureIndex.RADAR_TIME.value,
        x=rd.PointFeatureIndex.X.value,
        y=rd.PointFeatureIndex.Y.value,
        r=rd.PointFeatureIndex.R.value,
        azim=rd.PointFeatureIndex.AZ.value,
        route_id=rd.PointFeatureIndex.TRACK_ID_MANUAL.value,
        focusing_point_flag=rd.PointFeatureIndex.FOCUSING_POINT_FLAG.value
    )
    input_feature_indices: Tuple[rd.PointFeatureIndex, ...] = tuple(model_config.input_features)
    regress_feature_indices: Tuple[rd.PointFeatureIndex, ...] = tuple(filter(
        lambda f_idx: f_idx in input_feature_indices,
        model_config.regress_features
    ))
    feature_aug_freqs = model_config.feature_aug_freqs
    regress_feature_aug_freqs = model_config.regress_feature_aug_freqs
    input_aug_feature_indices: Tuple[rd.PointFeatureIndex] = tuple(filter(
        lambda f_idx: f_idx in input_feature_indices,
        model_config.input_aug_features
    ))
    regress_aug_feature_indices: Tuple[rd.PointFeatureIndex] = tuple(filter(
        lambda f_idx: f_idx in regress_feature_indices,
        model_config.regress_aug_features
    ))
    if not set(input_aug_feature_indices).issubset(set(feature_aug_freqs.keys())):
        raise ValueError(
            "Input augmented features are not subset of feature indices "
            "in features_aug_freqs."
        )
    if not set(regress_aug_feature_indices).issubset(set(regress_feature_aug_freqs.keys())):
        raise ValueError(
            "Regress augmented features are not subset of feature indices "
            "in regress_features_aug_freqs."
        )
    if len(input_aug_feature_indices) > 0:
        input_feature_augmenter = FeatureAugmenter(
            feature_indices=input_feature_indices,
            augment_freqs=OrderedDict([
                (f_idx, feature_aug_freqs[f_idx])
                for f_idx in input_aug_feature_indices
            ]),
            norm_to_01=model_config.regressor_output_sigmoid
        )
    else:
        input_feature_augmenter = None

    if len(regress_aug_feature_indices) > 0:
        regress_feature_augmenter = FeatureAugmenter(
            feature_indices=regress_feature_indices,
            augment_freqs=OrderedDict([
                (f_idx, regress_feature_aug_freqs[f_idx])
                for f_idx in regress_aug_feature_indices
            ]),
            norm_to_01=model_config.regressor_output_sigmoid
        )
    else:
        regress_feature_augmenter = None

    logger.info(f"Using train dataset: {train_dataset_path}, "
                f"test dataset: {test_dataset_path}")

    # Create raw model
    model = create_raw_model_from_config(model_config, reg_the_last_but_one_point)
    model_class = type(model)
    logger.info(f"Created model using class: %s", model_class.__name__)

    # Load base model checkpoint
    if weight_ckpt_path is None:
        logger.info(f"Training model from scratch.")

        loss_ckpt_path = None
        optimizer_ckpt_path = None
        scheduler_ckpt_path = None

        load_optim_states = False

    else:
        if not weight_ckpt_path.is_file():
            error_msg = (
                f"weight_ckpt_path ({weight_ckpt_path}) does not point to "
                f"a valid file. Please check it."
            )
            logger.error(error_msg)
            raise ValueError(
                error_msg
            )

        loaded = torch.load(
            weight_ckpt_path,
            map_location=torch.device("cpu"),
            weights_only=True
        )
        if isinstance(loaded, dict):
            loaded_state_dict = loaded
        else:
            loaded_state_dict = loaded.state_dict()

        model_state_dict = model.state_dict()
        for k, v in loaded_state_dict.items():
            _skip = False
            if k not in model_state_dict:
                logger.warning(f"Ignore module {k} in the checkpoint since it is "
                               f"not in the parameter set of the new model.")
                _skip = True
            for p in ignore_module_pattern:
                if re.match(p, k):
                    logger.debug(f"Ignore module {k} in the checkpoint since "
                                 f"it matches the ignore pattern {p}")
                    _skip = True
                    break
            if not _skip:
                model_state_dict[k] = v

        model.load_state_dict(unwrap_state_dict(model_state_dict), strict=False)
        logger.info(f"Loaded existed base model checkpoint successfully: {weight_ckpt_path}")

        loss_ckpt_path = weight_ckpt_path.parent / "loss.pth"
        optimizer_ckpt_path = weight_ckpt_path.parent / "optimizer.pth"
        scheduler_ckpt_path = weight_ckpt_path.parent / "scheduler.pth"

    # LoRA injection.  LoRA fine-tuning always needs a base model (either the
    # checkpoint loaded from ``--weight-ckpt-dir`` earlier, or implicitly the
    # base model that an existing ``--lora-weight-path`` adapter was mounted on).
    if use_lora:
        if lora_weight_path is None:
            if weight_ckpt_path is None:
                raise ValueError(
                    "LoRA fine-tuning requires a base model: either pass "
                    "--weight-ckpt-dir (base model weights) or --lora-weight-path "
                    "(an existing LoRA adapter of a base model)."
                )
            logger.info("Initializing LoRA fine-tuning...")
            from model_association.point_cc.model import create_lora_model

            lora_config = model_config.lora
            if lora_config is None:
                raise ValueError(
                    "LoRA config is not found in the config dict."
                )
            model = create_lora_model(
                model,
                lora_config
            )
            logger.info(f"Performing LoRA fine-tuning.")
        else:
            # Load LoRA weights
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter weights from {lora_weight_path}")
            model = PeftModel.from_pretrained(model, lora_weight_path, is_trainable=True)
            # pretrained_model = pretrained_model.merge_and_unload()

            logger.info("LoRA weights loaded.")

        # Report (always) and validate (semantic) the adapter that was mounted.
        if lora_weight_path is not None:
            report_lora_mount(model, lora_weight_path, logger)

    else:
        logger.info(f"Performing base model full-training.")

    # Move model to designated device
    model = model.to(device)

    if use_lora and lora_weight_path is not None:
        # Semantic check: a resumed LoRA adapter must be injectable into the
        # current model (shared parts must have matching shapes).
        validate_lora_mount(model, lora_weight_path, logger)
    # Freeze the requested parts.  For the base model the raw modules are
    # frozen; for a LoRA model the *active* trainable parameters (backbone
    # lora_A/lora_B adapters and the modules_to_save head copies) are frozen.
    if freeze_backbone:
        if use_lora:
            set_lora_part_requires_grad(model, "backbone", False)
        else:
            model.model_part_requires_grad("backbone", False)
    if freeze_classify_head:
        if use_lora:
            set_lora_part_requires_grad(model, "classify_branch", False)
        else:
            model.model_part_requires_grad("classify_branch", False)
    if freeze_regress_head:
        if use_lora:
            set_lora_part_requires_grad(model, "regress_branch", False)
        else:
            model.model_part_requires_grad("regress_branch", False)
    logger.debug(
        f"Parameters requiring grad: {[name for name, p in unwrap_ddp_model(model).named_parameters(recurse=True) if p.requires_grad]}"
    )

    trainable_params, all_param = get_num_trainable_parameters(model)
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

    # Create loss
    use_uncertainty_weighting = model_config.loss_coeffs.use_uncertainty_weighting
    loss_fn: CCLoss = CCLoss(
        loss_coeffs=model_config.loss_coeffs,
        device=device
    )
    logger.info(f"Initialized loss of type {type(loss_fn)} with coeffs: "
                f"{loss_fn.coeffs} ({'' if use_uncertainty_weighting else 'not '}"
                f"use uncertainty weighting)")
    # Try to load loss checkpoint
    if use_uncertainty_weighting:
        try:
            loaded = torch.load(loss_ckpt_path, map_location=torch.device("cpu"), weights_only=True)
            if isinstance(loaded, dict):
                loaded_state_dict = loaded
            else:
                loaded_state_dict = loaded.state_dict()
            loss_fn.load_state_dict(unwrap_state_dict(loaded_state_dict))
        except:
            logger.error(
                f"Failed to load loss checkpoint at {loss_ckpt_path} (which is "
                f"mandatory when using uncertainty weighting). "
                f"{traceback.format_exc()}"
            )
            exit(1)

    # Create optimizer
    opt_params = [{"params": model.parameters()}]
    if use_uncertainty_weighting:
        opt_params.append({"params": loss_fn.parameters(), "lr": 1e-2})
    optimizer: Optimizer = model_config.optimizer.load(opt_params)
    logger.info(f"Initialized optimizer of type {type(optimizer)} with params: "
                f"{model_config.optimizer.params}")
    # Try to load optimizer checkpoint
    if load_optim_states and optimizer_ckpt_path and optimizer_ckpt_path.is_file():
        loaded = torch.load(
            optimizer_ckpt_path,
            map_location=torch.device("cpu"),
            weights_only=True
        )
        if isinstance(loaded, dict):
            loaded_state_dict = loaded
        else:
            loaded_state_dict = loaded.state_dict()
        optimizer.load_state_dict(unwrap_state_dict(loaded_state_dict))
        logger.info(f"Loaded existed optimizer checkpoint successfully: {optimizer_ckpt_path}")

    # Create scheduler (if required)
    scheduler: Optional[LRScheduler] = None
    if model_config.scheduler:
        SchedulerType: Type[LRScheduler] = model_config.scheduler.initializer
        params: Dict[str, Any] = model_config.scheduler.params
        if issubclass(SchedulerType, WarmupCosineAnnealingLR):
            steps_per_epoch = len(train_dataloader)
            total_steps = steps_per_epoch * model_config.stop_epoch
            logger.debug(
                f"steps_per_epoch = {steps_per_epoch}"
            )

            warmup_steps_ratio = params.pop("warmup_steps_ratio")
            params["warmup_steps"] = math.ceil(warmup_steps_ratio * total_steps)
            params["total_steps"] = total_steps

        scheduler = SchedulerType(optimizer, **params)
        logger.info(f"Initialized scheduler of type {SchedulerType} with params: {params}")
    # Try to load scheduler checkpoint
    if load_optim_states and scheduler and scheduler_ckpt_path:
        if not scheduler_ckpt_path.is_file():
            logger.warning("Scheduler is initialized and model weight checkpoint is loaded. "
                           f"But the scheduler checkpoint is not found at {scheduler_ckpt_path}")
        else:
            loaded = torch.load(
                scheduler_ckpt_path,
                map_location=torch.device("cpu"),
                weights_only=True
            )
            if isinstance(loaded, dict):
                loaded_state_dict = loaded
            else:
                loaded_state_dict = loaded.state_dict()
            scheduler.load_state_dict(unwrap_state_dict(loaded_state_dict))
            logger.info(f"Loaded existed scheduler checkpoint successfully: {scheduler_ckpt_path}")

    collected_data_config: CollectedDataConfig = \
        CollectedDataConfig.from_file(dataset_dir_path / COLLECTED_DATA_CONFIG_FILENAME)
    for rd_type_id, config in collected_data_config.configs.items():
        collected_data_config.configs[rd_type_id].fill_in_normalization_factors()
    if collected_data_config.default_config:
        collected_data_config.default_config.fill_in_normalization_factors()
    logger.info(f"Loaded collected_data_config from file {dataset_dir_path / COLLECTED_DATA_CONFIG_FILENAME}: "
                f"{collected_data_config}")

    point_splitters = create_point_splitters(collected_data_config)

    if rank == 0:
        test_loss = []
        test_classify_loss = []
        test_regress_loss = []

        test_metrics_record_file = open(model_weight_save_dir / "test_metrics.txt", "a")
        metrics_names = TrainOrValidateMetrics.model_fields.keys()
        metrics_txt_col_widths = tuple(
            max(len(col) + 2, 10) if "regress_res" not in col else 47
            for col in metrics_names
        )
        title = "".join(f"{col: >{w}}" for col, w in zip(metrics_names, metrics_txt_col_widths))
        test_metrics_record_file.write(title + '\n')
        test_metrics_record_file.flush()

        best_weight_score = -np.inf

    if use_ddp:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    epoch = 0
    for epoch in range(model_config.stop_epoch):
        if use_ddp:
            dist.barrier()  # Synchronize all processes
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        train_metrics = \
            train_or_validate(
                model,
                dataloader=train_dataloader,
                feature_indices=feature_indices_obj,
                input_feature_indices=input_feature_indices,
                regress_feature_indices=regress_feature_indices,
                extra_n_in_features=model_config.extra_n_in_features,
                input_feature_augmenter=input_feature_augmenter,
                regress_feature_augmenter=regress_feature_augmenter,
                collected_data_config=collected_data_config,
                point_splitters=point_splitters,
                loss_on_uncertain_points=loss_on_uncertain_points,
                loss_on_track_without_next=loss_on_track_without_next,
                reg_the_last_but_one_point=reg_the_last_but_one_point,
                epoch=epoch,
                is_training=True,
                device=device,
                logger=logger,
                use_attn_as_corr=model_config.use_attn_as_corr,
                use_ddp=use_ddp,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                num_batches=None,
                model_save_dir=model_weight_save_dir if rank == 0 else None,
                log_interval=model_config.batches_per_logging
            )

        if use_ddp:
            dist.barrier()
        is_last_epoch = (epoch == model_config.stop_epoch - 1)
        to_test = ((epoch % model_config.epochs_per_test == 0) or is_last_epoch)
        to_save = ((epoch % model_config.epochs_per_saving == 0) or is_last_epoch)

        if to_test:
            test_metrics = \
                train_or_validate(
                    model,
                    dataloader=test_dataloader,
                    feature_indices=feature_indices_obj,
                    input_feature_indices=input_feature_indices,
                    regress_feature_indices=regress_feature_indices,
                    extra_n_in_features=model_config.extra_n_in_features,
                    input_feature_augmenter=input_feature_augmenter,
                    regress_feature_augmenter=regress_feature_augmenter,
                    collected_data_config=collected_data_config,
                    point_splitters=point_splitters,
                    loss_on_uncertain_points=loss_on_uncertain_points,
                    loss_on_track_without_next=loss_on_track_without_next,
                    reg_the_last_but_one_point=reg_the_last_but_one_point,
                    epoch=epoch,
                    is_training=False,
                    device=device,
                    logger=logger,
                    use_attn_as_corr=model_config.use_attn_as_corr,
                    use_ddp=use_ddp,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_batches=None,
                    model_save_dir=model_weight_save_dir if rank == 0 else None,
                    log_interval=model_config.batches_per_logging
                )
        else:
            test_metrics = None

        if use_ddp:
            dist.barrier()

        if rank == 0:
            if to_save:
                save_dir = model_weight_save_dir / f"epoch_{epoch}"
                save_dir.mkdir(exist_ok=True)
                unwrapped_model = unwrap_ddp_model(model)

                if use_lora:
                    logger.info(f"[Epoch {epoch}] Saving LoRA adapter...")
                    unwrapped_model.save_pretrained(save_dir)
                    logger.info(f"[Epoch {epoch}] LoRA adapter saved to {save_dir}")
                else:
                    logger.info(f"[Epoch {epoch}] Saving full model checkpoint...")
                    torch.save(unwrapped_model.state_dict(), save_dir / "model.pth")
                    logger.info(f"[Epoch {epoch}] Full model checkpoint saved to {save_dir / 'model.pth'}")

                torch.save(optimizer.state_dict(), save_dir / "optimizer.pth")
                if scheduler:
                    torch.save(scheduler.state_dict(), save_dir / "scheduler.pth")

                with open(save_dir / "metrics.json", "w") as f:
                    json.dump(
                        {"train": train_metrics.model_dump(),
                         "test": test_metrics.model_dump()},
                        f
                    )

            if to_test:
                test_loss.append(test_metrics.loss)
                test_classify_loss.append(test_metrics.classify_loss)
                test_regress_loss.append(test_metrics.regress_loss)
                record_str = ""
                for _field_name, w in zip(metrics_names, metrics_txt_col_widths):
                    v = getattr(test_metrics, _field_name)

                    if _field_name in ("epoch",) or isinstance(v, int):
                        type_char = "d"
                    elif isinstance(v, RegressResidualStatistic):
                        v = str(np.round(v.avg_res, 3))
                        type_char = "s"
                    elif isinstance(v, AccuracyStatistic):
                        v = v.accuracy
                        type_char = ".4f"
                    elif isinstance(v, float):
                        type_char = ".4f"
                    else:
                        v = str(v)
                        type_char = "s"
                    record_str += f"{v: >{w}{type_char}}"
                record_str += "\n"

                test_metrics_record_file.write(record_str)
                test_metrics_record_file.flush()

                if plot_metric_curve_flag:
                    plot_metric_curve_fn([test_loss, test_classify_loss, test_regress_loss],
                                         legends=['loss', 'classify loss', "regress loss"],
                                         title='Test metrics',
                                         save_path=model_weight_save_dir / 'test_metrics_curve.png')

                if weight_metric_score(test_metrics) > best_weight_score:
                    logger.info("The weight has the currently best performance. Save the checkpoint.")
                    save_dir = model_weight_save_dir / f"best_weight"
                    save_dir.mkdir(exist_ok=True)
                    unwrapped_model = unwrap_ddp_model(model)

                    if use_lora:
                        logger.info(f"[Epoch {epoch}] Saving currently best LoRA adapter...")
                        unwrapped_model.save_pretrained(save_dir)
                        logger.info(f"[Epoch {epoch}] Currently best LoRA adapter saved to {save_dir}")
                    else:
                        logger.info(f"[Epoch {epoch}] Saving currently best full model checkpoint...")
                        torch.save(unwrapped_model.state_dict(), save_dir / "model.pth")
                        logger.info(
                            f"[Epoch {epoch}] Currently best full model checkpoint saved to {save_dir / 'model.pth'}")

                    torch.save(optimizer.state_dict(), save_dir / "optimizer.pth")
                    if scheduler:
                        torch.save(scheduler.state_dict(), save_dir / "scheduler.pth")

                    with open(save_dir / "metrics.json", "w") as f:
                        json.dump(
                            {"train": train_metrics.model_dump(),
                             "test": test_metrics.model_dump()},
                            f
                        )

    if rank == 0 and epoch == model_config.stop_epoch - 1:
        save_dir = model_weight_save_dir / f"epoch_last"
        save_dir.mkdir(exist_ok=True)
        unwrapped_model = unwrap_ddp_model(model)

        if use_lora:
            logger.info(f"[Epoch {epoch}] Saving the last LoRA adapter...")
            unwrapped_model.save_pretrained(save_dir)
            logger.info(f"[Epoch {epoch}] The last LoRA adapter saved to {save_dir}")
        else:
            logger.info(f"[Epoch {epoch}] Saving the last full model checkpoint...")
            torch.save(unwrapped_model.state_dict(), save_dir / "model.pth")
            logger.info(f"[Epoch {epoch}] The last full model checkpoint saved to {save_dir / 'model.pth'}")

        torch.save(optimizer.state_dict(), save_dir / "optimizer.pth")
        if scheduler:
            torch.save(scheduler.state_dict(), save_dir / "scheduler.pth")

        with open(save_dir / "metrics.json", "w") as f:
            json.dump(
                {"train": train_metrics.model_dump(),
                 "test": test_metrics.model_dump()},
                f
            )

    if use_ddp:
        cleanup_ddp()

    logger.info(
        f"Rank {rank} finished training."
    )


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.open()


def main():
    app()


if __name__ == "__main__":
    main()
