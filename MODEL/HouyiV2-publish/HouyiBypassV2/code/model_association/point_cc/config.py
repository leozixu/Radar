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

import json
import logging
import numbers
import traceback
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any, Literal, Iterable, Type

import numpy as np
from ..radar_definition import PointFeatureIndex
from ..utils import TBaseModel
from pydantic import BaseModel, field_validator, field_serializer, ValidationInfo, Field, AliasChoices, ConfigDict, \
    model_validator

DATA_CONFIG_FILENAME = "data_config.json"
MODEL_CONFIG_FILENAME = "model_config.json"


class BaseModelFromFileMixin:
    @classmethod
    def from_file(cls: Type[TBaseModel], file_path: Union[str, Path]):
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        config = cls.model_validate(config_dict)
        return config


class ValueBounds(BaseModel):
    min: float = -np.inf
    max: float = np.inf


class NormalizationFactor(BaseModel):
    min: float = 0
    range: float


class DataConfigMaxValue(BaseModel, BaseModelFromFileMixin):
    model_config = ConfigDict(extra="allow")

    r: Optional[float] = np.inf
    "Maximum detecting distance (in meters)."
    amp: Optional[float] = np.inf
    "Maximum signal amplitude."
    snr: Optional[float] = np.inf
    "Maximum signal-to-noise ratio."
    height: Optional[float] = np.inf
    "Maximum height (in meters)."
    elev: Optional[Union[float, ValueBounds]] = None
    """
    Maximum elevation (in degrees). If it is a single float, bounds will be 
    set to (-elev, elev).
    """
    speed: Optional[float] = np.inf
    """
    Maximum radial speed (absolute value; in m/s), i.e. bounds are (-speed, 
    speed).
    """

    @field_validator(
        "r", "amp", "snr", "height", "elev", "speed"
    )
    @classmethod
    def convert_none_to_inf(cls, v):
        if v is None:
            return np.inf
        return v

    @field_validator("elev")
    @classmethod
    def parse_elev_bounds(cls, v):
        if v is None:
            return ValueBounds(min=-90, max=90)
        if isinstance(v, float):
            return ValueBounds(min=max(-v, -90), max=min(v, 90))
        return v

    @property
    def elev_bounds_in_rads(self):
        return ValueBounds(
            min=self.elev.min / 180 * np.pi,
            max=self.elev.max / 180 * np.pi
        )


class DataConfigNormalizationFactor(BaseModel, BaseModelFromFileMixin):
    model_config = ConfigDict(extra="allow")

    time: Optional[float] = None
    r: Optional[float] = None
    amp: Optional[float] = None
    snr: Optional[float] = None
    height: Optional[float] = None
    elev: Optional[Union[float, NormalizationFactor]] = None
    az: Optional[NormalizationFactor] = None
    speed_half: Optional[float] = None
    lon: Optional[NormalizationFactor] = None
    lat: Optional[NormalizationFactor] = None

    @field_validator("elev")
    @classmethod
    def parse_nf(cls, v):
        if isinstance(v, numbers.Number):
            return NormalizationFactor(min=-v, range=2 * v)
        return v


class DataConfig(BaseModel, BaseModelFromFileMixin):
    model_config = ConfigDict(extra="allow")

    max_sample_time_range: float
    "Maximum time range to collect points for generating samples each time."
    min_sample_time_range: float
    "Minimum time range to collect points for generating samples each time."
    min_track_len: int
    """
    Minimum length of tracks in samples. For those tracks whose lengths are 
    smaller than this value, their points will be treated as false points.
    """
    time_step: float
    "Per how long time to generate new samples."
    max_value: DataConfigMaxValue
    "Maximum values of physical quantities."
    split_azim_width: float
    "Azimuthal width of each split area (in degrees)."
    split_pad_width: float
    "Padding width of each split area (in meters)."
    r1: float
    "Radius of near region of each split area (in meters)."
    sample_len: int
    """
    Length of each sample, i.e. the number of points preserved in 
    each split area. If there are not that much points in one sample, 
    points filled with NaN will be appended; if there are more than 
    `sample_len` points, the latest `sample_len` points will be saved.
    """
    dt_between_circles: Optional[float] = None
    """
    Maximum time interval between the last point and the last but one point in 
    a route (in seconds). If dt_between_circles is not None, the routes not
    satisfying this constraint will be marked as uncertain; otherwise there is 
    no constraints.
    It is suggested to set this value to 1.8x scan period of the radar.
    """
    only_last_second_of_each_step: Optional[bool] = Field(
        default=None,
        deprecated="Use `n_areas_to_sample_in_each_step` instead."
    )
    """
    [DEPRECATED]
    If true, only the samples containing remote region points in the last second 
    of each time step will be kept; otherwise all samples will be kept, i.e. 
    ceil(360/split_azim_width) samples will be generated for each time step.
    """
    n_areas_to_sample_in_each_step: int = 0
    """
    The number of areas to be sampled in each step. 
    -1 means only the sample containing remote region points in 
    the last second of each time step will be kept; 
    0 means all samples will be kept, i.e. ceil(360/split_azim_width) 
    samples will be generated for each time step; 
    other integral value n in [1,ceil(360/split_azim_width)] means 
    n samples will be randomly chosen from all areas.
    """
    only_primary_and_fused_points: bool
    "If True, only primary and fused points will be kept in dataset."
    remove_uncertain_points: bool = False
    "If True, those points marked as uncertain (track ID = -1) will be filtered out."
    radial_split_borders: Optional[Tuple[Tuple[float, float], ...]]
    """
    [EXPERIMENTAL] Designate borders of radial splits of each fan area split. 
    -1 border stands for unlimited single sided border; None value stands for 
    full-range.
    """
    suppress_false_points: bool
    """
    [EXPERIMENTAL] If True, suppress false points such that the number of 
    false points <= 2 * the number of target points in each generated sample.
    """
    max_sample_count: Optional[int] = None
    """
    Maximum number of samples to generate for each sub-repo.
    None for using all samples.
    """
    normalization_factor: Optional[DataConfigNormalizationFactor] = None

    @model_validator(mode="before")
    @classmethod
    def compat_mapping(cls, data: dict):
        old_flag = data.get("only_last_second_of_each_step", None)
        new_state = data.get("n_areas_to_sample_in_each_step", None)
        if old_flag is not None and new_state is None:
            data["n_areas_to_sample_in_each_step"] = -1 if old_flag else 0
        elif new_state is not None and old_flag is None:
            data["only_last_second_of_each_step"] = (data["n_areas_to_sample_in_each_step"] == -1)
        return data

    @model_validator(mode="after")
    def fill_in_defaults(self):
        if self.dt_between_circles is None:
            t_max = self.max_sample_time_range
            self.dt_between_circles = t_max / 10 * 1.8  # TODO: don't hardcode the number of circles (10) here
        return self

    @field_serializer("only_last_second_of_each_step")
    def serialize_only_last_second_of_each_step(
        self,
        only_last_second_of_each_step: Optional[bool],
        info: ValidationInfo
    ) -> Optional[bool]:
        return self.n_areas_to_sample_in_each_step == -1

    def fill_in_normalization_factors(self):
        if self.normalization_factor.time is None:
            self.normalization_factor.time = self.max_sample_time_range

        for name in ("r", "amp", "snr", "height"):
            if getattr(self.normalization_factor, name) is not None:
                continue

            _max_v = getattr(self.max_value, name)
            if _max_v is not None and _max_v != np.inf:
                setattr(self.normalization_factor, name, _max_v)

        for name in ("speed",):
            if not hasattr(self.normalization_factor, name):
                nf_name = name + "_half"
            else:
                nf_name = name

            if not getattr(self.normalization_factor, nf_name):
                continue

            _max_v = getattr(self.max_value, name)
            if _max_v and _max_v != np.inf:
                setattr(self.normalization_factor, nf_name, _max_v)
            else:
                setattr(self.normalization_factor, nf_name, 1)

        for name in ("elev",):
            if getattr(self.normalization_factor, name) is not None:
                continue

            _v_bounds: Optional[ValueBounds] = getattr(self.max_value, name)
            if _v_bounds is not None:
                setattr(self.normalization_factor,
                        name,
                        NormalizationFactor(min=_v_bounds.min,
                                            range=_v_bounds.max - _v_bounds.min))


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    lora_r: int
    lora_alpha: int
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: Optional[Union[List[str], str]] = None
    modules_to_save: List[str] = []
    init_lora_weights: Union[bool, str] = True
    use_dora: bool = False
    use_rslora: bool = False
    rank_pattern: Dict[str, int] = {}
    alpha_pattern: Dict[str, int] = {}


class DynamicSearchConfig(BaseModel):
    name: str
    params: Dict[str, Any]
    search_module: str

    @property
    def initializer(self):
        import importlib

        module = importlib.import_module(self.search_module)
        initializer = getattr(module, self.name)

        return initializer

    def load(self, **kwargs):
        instance = self.initializer(**self.params, **kwargs)

        return instance


class ActivationConfig(DynamicSearchConfig):
    search_module: str = "torch.nn"


class OptimizerConfig(DynamicSearchConfig):
    search_module: str = "torch.optim"

    def load(self, params, **kwargs):
        return super().load(params=params, **kwargs)


class SchedulerConfig(DynamicSearchConfig):
    search_module: str = "torch.optim.lr_scheduler"

    def load(self, optimizer, **kwargs):
        return super().load(optimizer=optimizer, **kwargs)


class CCModelTrainingLossTermCoeffs(BaseModel):
    model_config = ConfigDict(extra="allow")

    regress_coeff: float
    classify_coeff: float
    circle_coeff: float = 2000
    use_uncertainty_weighting: bool


class CCModelTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dropout: float
    activation: ActivationConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    loss_coeffs: CCModelTrainingLossTermCoeffs
    stop_epoch: int
    train_num_samples: Optional[int] = None
    test_num_samples: Optional[int] = None
    batch_size: int
    batches_per_logging: int = Field(
        validation_alias=AliasChoices("log_interval", "batches_per_logging")
    )
    epochs_per_saving: int = Field(
        validation_alias=AliasChoices("save_interval", "epochs_per_saving")
    )
    epochs_per_test: int = 5
    seed: Optional[int] = None

    lora: Optional[LoraConfig] = None

    @model_validator(mode="before")
    @classmethod
    def compat_assemble_loss_coeffs(cls, values: dict):
        if "loss_coeffs" in values:
            return values

        regress_coeff = values.pop("regress_loss_coeff")
        classify_coeff = values.pop("classify_loss_coeff")
        circle_coeff = 2000
        use_uncertainty_weighting = False
        loss_coeffs = CCModelTrainingLossTermCoeffs(classify_coeff=classify_coeff,
                                                    regress_coeff=regress_coeff,
                                                    circle_coeff=circle_coeff,
                                                    use_uncertainty_weighting=use_uncertainty_weighting)
        values["loss_coeffs"] = loss_coeffs
        return values


class BaseCCModelConfig(CCModelTrainingConfig, BaseModelFromFileMixin):
    model_config = ConfigDict(extra="allow")

    backbone_type: str

    input_features: List[PointFeatureIndex]
    regress_features: List[PointFeatureIndex]
    feature_aug_freqs: Dict[PointFeatureIndex, List[float]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "feature_aug_freqs",
            "features_aug"
        )
    )
    regress_feature_aug_freqs: Optional[Dict[PointFeatureIndex, List[float]]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "regress_feature_aug_freqs",
            "output_features_aug",
            "regress_features_aug"
        )
    )
    input_aug_features: List[PointFeatureIndex]
    regress_aug_features: List[PointFeatureIndex]

    @field_validator(
        "input_features",
        "regress_features",
        "input_aug_features",
        "regress_aug_features",
        mode="before"
    )
    @classmethod
    def validate_feature_list(cls, v):
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    if item in PointFeatureIndex.__members__:
                        result.append(PointFeatureIndex[item])
                    else:
                        raise ValueError(f"Invalid feature name: {item}")
                elif isinstance(item, int):
                    result.append(PointFeatureIndex(item))
                else:
                    result.append(item)
            return result
        return v

    @field_validator(
        "feature_aug_freqs",
        "regress_feature_aug_freqs",
        mode="before"
    )
    @classmethod
    def validate_feature_dict(cls, v, info: ValidationInfo):
        if isinstance(v, dict):
            new_dict = {}
            for k, val in v.items():
                if isinstance(k, str) and k in PointFeatureIndex.__members__:
                    new_key = PointFeatureIndex[k]
                elif isinstance(k, int):
                    new_key = PointFeatureIndex(k)
                else:
                    new_key = k
                new_dict[new_key] = val
            return new_dict

        return v

    @model_validator(mode="after")
    def post_process_defaults(self):
        if self.regress_feature_aug_freqs is None:
            self.regress_feature_aug_freqs = self.feature_aug_freqs
        return self

    @field_serializer(
        "input_features",
        "regress_features",
        "input_aug_features",
        "regress_aug_features"
    )
    def serialize_feature_list(self, v: List[PointFeatureIndex]):
        return [item.name for item in v]

    @field_serializer(
        "feature_aug_freqs",
        "regress_feature_aug_freqs"
    )
    def serialize_feature_dict(self, v: Dict[PointFeatureIndex, Optional[List[float]]]):
        if v is None:
            return None
        return {k.name: val for k, val in v.items()}


class BertCCModelConfig(BaseCCModelConfig):
    model_config = ConfigDict(extra="allow")

    backbone_type: Literal["bert"] = "bert"

    d_model: int
    n_attn_heads: int
    d_ff: int
    n_layers: int
    n_hidden_layers_of_classifier: int
    n_hidden_layers_of_regressor: int

    use_attn_as_corr: bool
    regressor_output_sigmoid: bool
    use_bert_classifier: bool = False
    use_bert_regressor: bool = False


def load_config(
    search_paths: Iterable[Union[str, Path]],
    config_type: Type[BaseModelFromFileMixin],
    logger: Optional[logging.Logger] = None
) -> Union[Tuple[BaseModel, Path], Tuple[None, None]]:
    if logger is not None:
        logger.debug(f"Search paths: {search_paths}")
    config = None
    config_file_path_used = None
    for p in search_paths:
        if p is None:
            continue
        try:
            config = config_type.from_file(p)
            config_file_path_used = p
        except Exception as e:
            if logger is not None:
                logger.warning(
                    f"Cannot load config from file {p} due to {e}. {traceback.format_exc()}"
                )
            config = None
            config_file_path_used = None
        else:
            break

    if logger is not None:
        if config is not None:
            logger.info(f"Loaded config file from {config_file_path_used}: {config}")
        else:
            logger.warning(f"Cannot load any config from {search_paths} "
                           f"using config type {config_type}")

    return config, config_file_path_used


def load_model_config(
    search_paths: Iterable[Union[str, Path]],
    config_type: Type[BaseCCModelConfig] = BertCCModelConfig,
    model_weight_save_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Union[Tuple[BaseModel, Path], Tuple[None, None]]:
    search_paths = [Path(p) for p in search_paths if p is not None]
    if model_weight_save_dir is not None:
        search_paths += [Path(model_weight_save_dir) / MODEL_CONFIG_FILENAME]

    return load_config(search_paths, config_type=config_type, logger=logger)
