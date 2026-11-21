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
from typing import Optional, Callable, Tuple, List

import torch
import torch.nn as nn
from model_association.point_cc.model.base import SetCompRequiresGradMixin, ModelParts
from transformers import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer


class PointClusterModel(nn.Module):
    """Point clustering model using BERT encoder for point cloud feature transformation.
    
    This model encodes point cloud features through a BERT encoder and maps them to
    output features using a dense network. It supports attention visualization and
    correlation analysis.
    
    Attributes:
        config (BertConfig): BERT configuration object
        input_embedding (nn.Sequential): Input feature embedding layer
        bert_encoder (BertEncoder): BERT encoder for feature transformation
        output_mapper (nn.Sequential): Output mapping network
        encoder_last_hidden_state (torch.Tensor): Last encoder hidden state
        encoder_attentions (tuple): Attention weights from all encoder layers
    
    Example:
        >>> config = BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4)
        >>> model = PointClusterModel(n_in_features=64, n_out_features=32, enc_config=config)
        >>> input = torch.randn(2, 100, 64)  # (batch, points, features)
        >>> output = model(input)  # (batch, points, 32)
    """

    def __init__(self, n_in_features: int, n_out_features: int, enc_config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = enc_config

        self.input_embedding = torch.nn.Sequential(
            torch.nn.Linear(n_in_features, enc_config.hidden_size),
            torch.nn.ReLU()
        )

        self.bert_encoder: BertEncoder = BertEncoder(
            config=enc_config
        )

        self.output_mapper = torch.nn.Sequential(
            torch.nn.Linear(enc_config.hidden_size, enc_config.hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_config.hidden_size // 2, n_out_features),
            torch.nn.Sigmoid()
        )

        self.encoder_last_hidden_state = None
        self.encoder_attentions = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input: Input point cloud features of shape (batch_size, n_points, n_in_features)
        
        Returns:
            Mapped features of shape (batch_size, n_points, n_out_features)
        """
        hidden_states = self.input_embedding(input)
        # shape=(batch_size, n_points, hidden_size)

        attention_mask = torch.ones_like(hidden_states[:, :, 0])
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
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        self.encoder_last_hidden_state = bert_output.last_hidden_state
        # shape=(batch_size, sequence_length, hidden_size)
        self.encoder_attentions = bert_output.attentions
        # tuple of `Tensor`s of each Bert encoder layer
        # each Tensor is of shape=
        # (batch_size, num_heads, sequence_length, sequence_length)

        mapped_features = self.output_mapper(self.encoder_last_hidden_state)

        return mapped_features

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
        if self.encoder_attentions is None:
            return None

        attn = self.encoder_attentions[-1]
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


class PointClassifyAndClusterModel(nn.Module, SetCompRequiresGradMixin):
    """Point classification and clustering model with BERT encoder backbone.
    
    This model extends PointClusterModel by adding separate classifier and regressor heads.
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
        concat_input_to_regressor: bool = False,
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
        self._concat_input_to_regressor = concat_input_to_regressor

        self.component_parts = ModelParts()
        self.component_parts.add_backbone_comps(
            {"input_embedding": self.input_embedding,
             "bert_encoder": self.bert_encoder}
        )

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
            self.component_parts.add_classify_comps({
                "input_emb": self.point_classifier_input_emb,
                "concat_emb": self.point_classifier_concat_emb
            })

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
            self.component_parts.add_classify_comps({
                "point_classifier": self.point_classifier
            })
        else:
            config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size
            )
            self.point_classifier = BertEncoder(config)
            self.classifier_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 2),
                # torch.nn.ReLU()
            )  # output logits
            self.component_parts.add_classify_comps({
                "point_classifier": self.point_classifier,
                "output_head": self.classifier_head
            })

    def _build_regressor(
        self,
        n_layers: int,
        n_out_features: int,
        use_bert: bool,
        regressor_output_sigmoid: bool
    ):
        hidden_size = self.config.hidden_size

        if self._concat_input_to_regressor:
            self.regressor_input_emb = torch.nn.Sequential(
                torch.nn.Linear(self.n_in_features, hidden_size),
                torch.nn.ReLU()
            )
            self.regressor_concat_emb = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.ReLU()
            )
            self.component_parts.add_regress_comps(
                {"input_emb": self.regressor_input_emb,
                 "concat_emb": self.regressor_concat_emb}
            )

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
            self.component_parts.add_regress_comps(
                {"regressor": self.regressor}
            )
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
            self.component_parts.add_regress_comps(
                {"regressor": self.regressor,
                 "output_head": self.regressor_head}
            )

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
        input_embedded = self.input_embedding(input)
        # shape=(batch_size, n_points, hidden_size)

        # In attention_mask, 1 to keep, 0 to mask
        # attention_mask = torch.ones_like(input_embedded[:, :, 0])
        attention_mask = ~((input == 0).all(-1))
        # attention_mask = (input.abs().sum(dim=-1) > 0).float()
        # shape=(batch_size, n_points)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # shape=(batch_size, 1, 1, n_points)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        bert_output: BaseModelOutputWithPastAndCrossAttentions = self.bert_encoder(
            input_embedded,
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

        if self._concat_input_to_regressor:
            regressor_input_emb = self.regressor_input_emb(input)
            regressor_input_emb_concat = torch.concat(
                (_encoder_last_hidden_state,
                 regressor_input_emb),
                dim=-1
            )
            regressor_input = self.point_classifier_concat_emb(regressor_input_emb_concat)
        else:
            regressor_input = _encoder_last_hidden_state

        if self._use_bert_regressor:
            regressor_output = self.regressor(
                regressor_input,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            regressed_features = self.regressor_head(regressor_output.last_hidden_state)
        else:
            regressed_features = self.regressor(regressor_input)

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


def expand_feature_into_multiple_precisions(
    feature: torch.Tensor,
    decimals: Tuple[int, ...],
    fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> List[torch.Tensor]:
    """Expand feature into multiple precision representations.
    
    For each decimal place, extracts the fractional part at that precision level.
    Useful for multi-scale feature representation.
    
    Args:
        feature: Input feature tensor of arbitrary shape
        decimals: Tuple of decimal places to extract (e.g., (1, 2, 3) for 0.1, 0.01, 0.01)
        fn: Optional function to apply to each expanded feature
    
    Returns:
        List of expanded feature tensors, one for each decimal place
    
    Example:
        >>> feature = torch.tensor([1.234, 5.678])
        >>> expanded = expand_feature_into_multiple_precisions(feature, (1, 2, 3))
        >>> # expanded[0] contains 0.1 precision: [0.4, 0.8]
        >>> # expanded[1] contains 0.01 precision: [0.34, 0.78]
        >>> # expanded[2] contains 0.001 precision: [0.234, 0.678]
    """
    feature_expanded = []
    for decimal in decimals:
        feature_exp = torch.fmod(feature, 0.1 ** decimal) * 10 ** decimal

        if isinstance(fn, Callable):
            feature_exp = fn(feature_exp)

        feature_expanded.append(feature_exp)

    return feature_expanded
