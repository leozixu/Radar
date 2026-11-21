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

from typing import Union, Tuple

from .bert_based import BertConfig
from .bert_based import PointClassifyAndClusterModel as BertCCModel
# from .qwen2_based import Qwen2CCModel
from .houyi2 import Houyi2CCModel
from .qwen2_based import Qwen2CCBaselineModel as Qwen2CCModel
from .qwen2_based import Qwen2Config
from ..config import LoraConfig, Qwen2CCModelConfig, Houyi2CCModelConfig, BertCCModelConfig


def create_raw_model_from_config(
    model_config: Union[BertCCModelConfig, Houyi2CCModelConfig, Qwen2CCModelConfig],
    reg_the_last_but_one_point: int
):
    input_feature_indices: Tuple[int, ...] = tuple(model_config.input_features)
    regress_feature_indices: Tuple[int, ...] = tuple(filter(
        lambda f_idx: f_idx in input_feature_indices,
        model_config.regress_features
    ))
    feature_aug_freqs = model_config.feature_aug_freqs
    regress_feature_aug_freqs = model_config.regress_feature_aug_freqs
    input_aug_feature_indices: Tuple[int, ...] = tuple(filter(
        lambda f_idx: f_idx in input_feature_indices,
        model_config.input_aug_features
    ))
    regress_aug_feature_indices: Tuple[int, ...] = tuple(filter(
        lambda f_idx: f_idx in regress_feature_indices,
        model_config.regress_aug_features
    ))

    if not set(input_aug_feature_indices).issubset(set(feature_aug_freqs.keys())):
        raise ValueError(
            "Input augmented features are not subset of feature indices "
            "in features_aug."
        )
    if not set(regress_aug_feature_indices).issubset(set(regress_feature_aug_freqs.keys())):
        raise ValueError(
            "Regress augmented features are not subset of feature indices "
            "in features_aug."
        )

    n_in_features = (len(input_feature_indices)
                     + sum([len(feature_aug_freqs[f_idx])
                            for f_idx in input_aug_feature_indices]) * 2) + model_config.extra_n_in_features
    n_out_features = (len(regress_feature_indices)
                      + sum([len(regress_feature_aug_freqs[f_idx])
                             for f_idx in regress_aug_feature_indices]) * 2) \
                     * (3 if reg_the_last_but_one_point else 2)
    if isinstance(model_config, BertCCModelConfig):
        enc_config = BertConfig(embedding_size=len(input_feature_indices),
                                hidden_size=model_config.d_model,
                                num_hidden_layers=model_config.n_layers,
                                num_attention_heads=model_config.n_attn_heads,
                                intermediate_size=model_config.d_ff)

        if isinstance(model_config, Houyi2CCModelConfig):
            model = Houyi2CCModel(
                n_in_features=n_in_features,
                n_out_features=n_out_features,
                enc_config=enc_config,
                n_hidden_layers_of_classifier=model_config.n_hidden_layers_of_classifier,
                n_hidden_layers_of_regressor=model_config.n_hidden_layers_of_regressor,
                regressor_output_sigmoid=model_config.regressor_output_sigmoid,
                use_bert_regressor=model_config.use_bert_regressor,
                use_bert_classifier=model_config.use_bert_classifier
            )
        else:
            model = BertCCModel(
                n_in_features=n_in_features,
                n_out_features=n_out_features,
                enc_config=enc_config,
                n_hidden_layers_of_classifier=model_config.n_hidden_layers_of_classifier,
                n_hidden_layers_of_regressor=model_config.n_hidden_layers_of_regressor,
                regressor_output_sigmoid=model_config.regressor_output_sigmoid,
                use_bert_regressor=model_config.use_bert_regressor,
                use_bert_classifier=model_config.use_bert_classifier,
                concat_input_to_classifier=model_config.concat_input_to_classifier,
                concat_input_to_regressor=model_config.concat_input_to_regressor
            )
    else:
        config = Qwen2Config(
            hidden_size=model_config.d_model,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            num_key_value_heads=model_config.num_key_value_heads,
            intermediate_size=model_config.d_ff,
            max_position_embeddings=512,
            vocab_size=1000,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            use_cache=False
        )
        model = Qwen2CCModel(
            config=config,
            n_in_features=n_in_features,
            n_out_features=n_out_features,
            n_hidden_layers_of_classifier=model_config.n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor=model_config.n_hidden_layers_of_regressor,
            regressor_output_sigmoid=model_config.regressor_output_sigmoid,
            use_qwen2_classifier=model_config.use_qwen2_classifier,
            use_qwen2_regressor=model_config.use_qwen2_regressor
        )

    return model


def create_lora_model(
    model: Union[BertCCModel, Qwen2CCModel],
    lora_config: LoraConfig
):
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    peft_lora_config = PeftLoraConfig(
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        modules_to_save=lora_config.modules_to_save,
        init_lora_weights=lora_config.init_lora_weights,
        use_dora=lora_config.use_dora,
        use_rslora=lora_config.use_rslora,
        rank_pattern=lora_config.rank_pattern,
        alpha_pattern=lora_config.alpha_pattern
    )
    model = get_peft_model(model, peft_lora_config)

    return model


__all__ = [
    "Qwen2Config",
    "BertConfig",
    "Qwen2CCModel",
    "BertCCModel",
    "Houyi2CCModel",
    "create_lora_model"
]
