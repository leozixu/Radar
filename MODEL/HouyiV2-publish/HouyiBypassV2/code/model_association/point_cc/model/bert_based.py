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

import inspect
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer


class PointClassifyAndClusterModel(nn.Module):
    """Point classification and clustering model with BERT encoder backbone.

    This model provides a BERT encoder backbone with separate classifier and regressor heads.
    Both heads can use either dense networks or BERT encoders for feature transformation.

    Architecture:
        - Input embedding: Linear + ReLU
        - BERT encoder: Multi-layer bidirectional attention
        - Classifier head: Point-wise classification (dense or BERT encoder)
        - Regressor head: Feature regression (dense or BERT encoder)

    Attributes:
        config (BertConfig): BERT configuration object
        input_embedding (nn.Sequential): Input feature embedding
        bert_encoder (BertEncoder): Main BERT encoder
        point_classifier (nn.Module): Classification head (Sequential or BertEncoder)
        regressor (nn.Module): Regression head (Sequential or BertEncoder)
        encoder_last_hidden_state (torch.Tensor): Last encoder hidden state
        encoder_last_attention (torch.Tensor): Last layer attention weights

    Example:
        >>> config = BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4)
        >>> model = PointClassifyAndClusterModel(
        ...     n_in_features=64,
        ...     n_out_features=32,
        ...     enc_config=config,
        ...     use_bert_classifier=True,
        ...     use_bert_regressor=False
        ... )
        >>> input = torch.randn(2, 100, 64)
        >>> probs, features, hidden = model(input)
    """

    def __init__(
        self,
        n_in_features: int,
        n_out_features: int,
        enc_config: BertConfig,
        n_hidden_layers_of_classifier: int = 1,
        n_hidden_layers_of_regressor: int = 1,
        regressor_output_sigmoid: bool = False,
        use_bert_classifier: bool = False,
        use_bert_regressor: bool = False,
        concat_input_to_classifier: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = enc_config

        self.n_in_features = n_in_features
        self.input_embedding = torch.nn.Sequential(
            torch.nn.Linear(n_in_features, enc_config.hidden_size),
            torch.nn.ReLU()
        )

        self.bert_encoder: BertEncoder = BertEncoder(
            config=enc_config
        )
        self._modify_last_layer()
        self._register_last_layer_hooks()

        self._concat_input_to_classifier = concat_input_to_classifier
        self._use_bert_classifier = use_bert_classifier
        self._build_classifier(
            n_hidden_layers_of_classifier,
            use_bert_classifier
        )

        self._use_bert_regressor = use_bert_regressor
        self._build_regressor(
            n_hidden_layers_of_regressor,
            n_out_features,
            use_bert_regressor,
            regressor_output_sigmoid
        )

        self.encoder_last_hidden_state: Optional[torch.Tensor] = None
        # shape=(batch_size, sequence_length, hidden_size)
        self.encoder_last_attention: Optional[torch.Tensor] = None
        # attention `Tensor` of the last Bert encoder layer of shape=
        # (batch_size, num_heads, sequence_length, sequence_length)

    def _build_classifier(
        self,
        n_layers: int,
        use_bert: bool
    ):
        hidden_size = self.config.hidden_size

        if self._concat_input_to_classifier:
            self.point_classifier_input_emb = torch.nn.Sequential(
                torch.nn.Linear(self.n_in_features, hidden_size),
                torch.nn.ReLU()
            )
            self.point_classifier_concat_emb = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.ReLU()
            )

        if not use_bert:
            self.point_classifier = torch.nn.Sequential()
            for i in range(n_layers - 1):
                self.point_classifier += (torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU()
                ))
            self.point_classifier += torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size // 2, 2)
            )  # output logits
        else:
            config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size
            )
            self.point_classifier = BertEncoder(config)
            self.classifier_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 2)
            )  # output logits

    def _build_regressor(
        self,
        n_layers: int,
        n_out_features: int,
        use_bert: bool,
        regressor_output_sigmoid: bool
    ):
        hidden_size = self.config.hidden_size

        if not use_bert:
            self.regressor = torch.nn.Sequential()
            for i in range(n_layers - 1):
                self.regressor += (torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU()
                ))
            self.regressor += torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size // 2, n_out_features)
            )
            if regressor_output_sigmoid:
                self.regressor += torch.nn.Sigmoid()
        else:
            config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size
            )
            self.regressor = BertEncoder(config)
            self.regressor_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, n_out_features)
            )
            if regressor_output_sigmoid:
                self.regressor_head += torch.nn.Sigmoid()

    def _modify_last_layer(self):
        last_layer: BertLayer = self.bert_encoder.layer[-1]
        original_forward = getattr(last_layer.__class__, "forward")
        original_forward_sig = inspect.signature(original_forward)

        def new_forward(inst, *args, **kwargs):
            bound = original_forward_sig.bind(inst, *args, **kwargs)
            bound.apply_defaults()
            bound.arguments["output_attentions"] = True
            return original_forward(*bound.args, **bound.kwargs)

        last_layer.forward = types.MethodType(new_forward, last_layer)

    def _register_last_layer_hooks(self):
        def _get_attn_hook(module, input, output):
            if len(output) > 1:
                self.encoder_last_attention = output[1]

        def _get_hidden_hook(module, input, output):
            self.encoder_last_hidden_state = output[0]

        last_layer: BertLayer = self.bert_encoder.layer[-1]
        last_layer.attention.self.register_forward_hook(_get_attn_hook)
        last_layer.register_forward_hook(_get_hidden_hook)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through classification and regression heads.

        Args:
            input: Input point cloud features of shape (batch_size, n_points, n_in_features)

        Returns:
            Tuple containing:
                - probs: Classification probabilities of shape (batch_size, n_points, 2)
                - regressed_features: Regressed features of shape (batch_size, n_points, n_out_features)
                - hidden_state: Last encoder hidden state of shape (batch_size, n_points, hidden_size)
        """
        hidden_states = self.input_embedding(input)
        # shape=(batch_size, n_points, hidden_size)
        attention_mask = ~((input == 0).all(-1))
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        bert_output: BaseModelOutputWithPastAndCrossAttentions = self.bert_encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        _encoder_last_hidden_state = self.encoder_last_hidden_state.clone()
        self.encoder_last_hidden_state.detach_()
        if self.encoder_last_attention is not None:
            self.encoder_last_attention.detach_()

        if self._concat_input_to_classifier:
            classifier_input_emb = self.point_classifier_input_emb(input)
            classifier_input_emb_concat = torch.concat(
                (_encoder_last_hidden_state,
                 classifier_input_emb),
                dim=-1
            )
            classifier_input = self.point_classifier_concat_emb(classifier_input_emb_concat)
        else:
            classifier_input = _encoder_last_hidden_state
        if self._use_bert_classifier:
            classifier_output = self.point_classifier(
                classifier_input,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            logits: torch.Tensor = self.classifier_head(classifier_output.last_hidden_state)
        else:
            logits: torch.Tensor = self.point_classifier(classifier_input)
        probs = torch.nn.Softmax(dim=-1)(logits)

        if self._use_bert_regressor:
            regressor_output = self.regressor(
                _encoder_last_hidden_state,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            regressed_features = self.regressor_head(regressor_output.last_hidden_state)
        else:
            regressed_features = self.regressor(_encoder_last_hidden_state)

        return probs, regressed_features, _encoder_last_hidden_state

    def get_attention(self, mean_over_heads: bool = False) -> Optional[torch.Tensor]:
        """Get attention weights from the last encoder layer.

        Args:
            mean_over_heads: If True, average attention over all heads.
                             If False, return per-head attention weights.

        Returns:
            Attention weights of shape:
            - (batch_size, seq_len, seq_len) if mean_over_heads=True
            - (batch_size, num_heads, seq_len, seq_len) if mean_over_heads=False
            Returns None if no forward pass has been executed.
        """
        if self.encoder_last_attention is None:
            return None

        attn = self.encoder_last_attention
        # shape=(batch_size, num_heads, sequence_length, sequence_length)

        if mean_over_heads:
            return attn.mean(dim=1)  # (batch_size, seq_len, seq_len)

        return attn  # (batch_size, num_heads, seq_len, seq_len)

    def get_correlation(self) -> Optional[torch.Tensor]:
        """Compute correlation matrix from last hidden state.

        Computes normalized correlation between all pairs of positions:
        corr[i,j] = cosine_similarity(h[i], h[j])

        Returns:
            Correlation matrix of shape (batch_size, seq_len, seq_len).
            Returns None if no forward pass has been executed.
        """
        if self.encoder_last_hidden_state is None:
            return None

        normalized_state = (
            self.encoder_last_hidden_state /
            torch.norm(self.encoder_last_hidden_state, dim=-1, keepdim=True)
        )
        # shape=(batch_size, sequence_length, hidden_size)
        corr = torch.matmul(
            normalized_state,
            normalized_state.transpose(-1, -2)
        )
        return corr  # shape=(batch_size, seq_len, seq_len)
