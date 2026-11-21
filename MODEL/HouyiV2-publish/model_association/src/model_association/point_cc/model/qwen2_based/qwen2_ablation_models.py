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

"""
Qwen2 model ablation study implementation.

Contains five ablation variants, each removing or replacing one core technology.

Adapted for transformers==4.41.2, torch==2.4.0

"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from model_association.point_cc.model.base import SetCompRequiresGradMixin, ModelParts
from transformers import Qwen2Config, Qwen2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, Qwen2RMSNorm, \
    Qwen2MLP, Qwen2RotaryEmbedding


# Traditional LayerNorm implementation (no bias, same parameter count as RMSNorm)
class TraditionalLayerNorm(nn.Module):
    """Traditional LayerNorm implementation for ablation studies.

    Differences from RMSNorm:
    - Subtracts mean before normalization (RMSNorm does not)
    - Same parameter count as RMSNorm (weight only, no bias)

    Args:
        hidden_size: Hidden dimension size
        eps: Epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # No bias to maintain same parameter count as RMSNorm
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute mean and variance for normalization.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Normalized tensor of same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, keepdim=True, unbiased=False)
        hidden_states = (hidden_states - mean) * \
                        torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        return hidden_states.to(input_dtype)


# Traditional MLP implementation (no SwiGLU, actual implementation)
class TraditionalMLP(nn.Module):
    """Traditional MLP for ablation studies, replaces SwiGLU.

    Differences from SwiGLU:
    - Single gate_proj and down_proj (SwiGLU has gate, up, down projections)
    - Uses single activation function (SwiGLU uses gating mechanism)

    Args:
        config: Qwen2Config configuration object
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # Traditional MLP has only one input projection and one output projection
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """Forward pass: Linear -> Activation -> Linear.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Traditional MLP: Linear -> Activation -> Linear
        x = self.act_fn(self.gate_proj(x))
        x = self.down_proj(x)
        return x


# Traditional position embedding implementation (sin/cos encoding, no learnable parameters)
class TraditionalPositionEmbedding(nn.Module):
    """Traditional position embedding for ablation studies, replaces RoPE.

    Differences from RoPE:
    - Does not use rotary position encoding mechanism
    - Returns None to skip RoPE application in attention layer

    Args:
        config: Qwen2Config configuration object
        device: Device type (unused)
    """

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings

    def forward(self, hidden_states, position_ids):
        """Forward pass: return None to skip RoPE application.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            position_ids: Position ID tensor of shape (batch, seq_len)

        Returns:
            (None, None): Indicates no position embeddings are used
        """
        # For traditional position embeddings, we don't use RoPE's cos/sin format
        # Instead, return two None values to let the attention layer skip position embedding application
        return None, None


# Attention layer implementation (adapted for transformers 4.41.2)
class Qwen2Attention(nn.Module):
    """Qwen2 attention layer with GQA (Grouped Query Attention) support.

    Features:
    - Q, K, V projections and attention computation
    - GQA mechanism (num_key_value_heads < num_attention_heads)
    - Integrated RoPE position embeddings
    - Optional sliding window attention

    Args:
        config: Qwen2Config configuration object
        layer_idx: Layer index for cache management
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(
            config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.hidden_size  # 添加 hidden_size 属性
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self, hidden_states, attention_mask, position_ids, past_key_value=None,
        output_attentions=False, use_cache=False, **kwargs
    ):
        """Forward pass: compute self-attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch, 1, seq_len, seq_len)
            position_ids: Position ID tensor of shape (batch, seq_len)
            past_key_value: Cached K/V states for auto-regressive generation
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cache

        Returns:
            attn_output: Attention output of shape (batch, seq_len, hidden_size)
            attn_weights: Attention weights (if output_attentions=True)
            past_key_value: Updated K/V cache (if use_cache=True)
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Apply RoPE position embeddings
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        if isinstance(self.rotary_emb, Qwen2RotaryEmbedding):
            # Qwen2RotaryEmbedding uses seq_len parameter
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            # TraditionalPositionEmbedding returns (None, None)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
                cos, sin = position_embeddings
                if cos is not None and sin is not None:
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                else:
                    cos, sin = None, None
            else:
                cos, sin = None, None

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos} if cos is not None else {}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Base decoder layer (adapted for transformers 4.41.2)
class BaseDecoderLayer(nn.Module):
    """Base decoder layer (Pre-LN structure) for Qwen2 models.

    Structure:
    - Pre-LayerNorm: Normalize before computing attention and MLP
    - Residual connections: After both attention and MLP
    - Configurable norm_class and mlp_class

    Args:
        config: Qwen2Config configuration object
        layer_idx: Layer index
        norm_class: Normalization layer class (e.g., RMSNorm or LayerNorm)
        mlp_class: MLP class (e.g., SwiGLU or TraditionalMLP)
    """

    def __init__(self, config: Qwen2Config, layer_idx: int, norm_class, mlp_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = mlp_class(config)
        self.input_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] if hasattr(
            config, "layer_types") else "full_attention"

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
        output_attentions=False, use_cache=False, **kwargs
    ):
        """Forward pass: Pre-LN decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position ID tensor
            past_key_value: Cached K/V states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cache

        Returns:
            hidden_states: Output tensor of shape (batch, seq_len, hidden_size)
            self_attn_weights: Attention weights (optional)
            present_key_value: Updated K/V cache (optional)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# Post-LN decoder layer
class PostLNDecoderLayer(nn.Module):
    """Post-LN decoder layer for ablation studies, replaces Pre-LN structure.

    Differences from Pre-LN:
    - Compute attention and MLP first, then normalize
    - Normalization applied after residual connection

    Args:
        config: Qwen2Config configuration object
        layer_idx: Layer index
        norm_class: Normalization layer class
        mlp_class: MLP class
    """

    def __init__(self, config: Qwen2Config, layer_idx: int, norm_class, mlp_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = mlp_class(config)
        self.input_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] if hasattr(
            config, "layer_types") else "full_attention"

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
        output_attentions=False, use_cache=False, **kwargs
    ):
        """Forward pass: Post-LN decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position ID tensor
            past_key_value: Cached K/V states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cache

        Returns:
            hidden_states: Output tensor of shape (batch, seq_len, hidden_size)
            self_attn_weights: Attention weights (optional)
            present_key_value: Updated K/V cache (optional)
        """
        # Compute attention first, then normalize
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        residual = hidden_states
        # Compute MLP first, then normalize
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2BaseModel(Qwen2PreTrainedModel):

    def __init__(
        self,
        config: Qwen2Config,
        norm_class,
        mlp_class,
        position_embedding_class,
        decoder_layer_class,
        is_causal: bool = True
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._attn_implementation = "eager"
        self.is_causal = is_causal

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([
            decoder_layer_class(config, layer_idx, norm_class, mlp_class)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in getattr(
            config, "layer_types", ["full_attention"] * config.num_hidden_layers)

        for layer in self.layers:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
            if position_embedding_class == Qwen2RotaryEmbedding:
                layer.self_attn.rotary_emb = Qwen2RotaryEmbedding(
                    dim=config.hidden_size // config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    base=getattr(config, "rope_theta", 1000000)
                )
            else:
                layer.self_attn.rotary_emb = position_embedding_class(config=config)
            layer.self_attn.num_heads = config.num_attention_heads
            layer.self_attn.num_key_value_heads = config.num_key_value_heads
            layer.self_attn.is_causal = is_causal

        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch_size, seq_len, _ = inputs_embeds.shape
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, dtype=torch.long, device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.is_causal:
            attention_mask_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_len),
                inputs_embeds,
                past_seen_tokens,
                sliding_window=getattr(self.config, "sliding_window", None) if self.has_sliding_layers else None
            )
        else:
            attention_mask_4d = attention_mask
            if attention_mask is not None:
                attention_mask_4d = attention_mask[:, None, None, :]
                attention_mask_4d = (1.0 - attention_mask_4d) * -10000.0

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attns] if
                v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2StructureModel(nn.Module):

    def __init__(self, config: Qwen2Config):
        super().__init__()
        config.use_cache = False

        # 导入原始 Qwen2Model
        from transformers import Qwen2Model as OriginalQwen2Model
        self.qwen2_model = OriginalQwen2Model(config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class TraditionalMultiHeadAttention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(
            config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads)
        # 传统多头注意力，每个头都有独立的 K 和 V 投影
        self.num_key_value_groups = 1
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # 传统 MHA
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self, hidden_states, attention_mask, position_ids, past_key_value=None,
        output_attentions=False, use_cache=False, **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # 应用 RoPE 位置编码
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        if isinstance(self.rotary_emb, Qwen2RotaryEmbedding):
            # Qwen2RotaryEmbedding 使用 seq_len 参数
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            # TraditionalPositionEmbedding 返回 (None, None)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
                cos, sin = position_embeddings
                if cos is not None and sin is not None:
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                else:
                    cos, sin = None, None
            else:
                cos, sin = None, None

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos} if cos is not None else {}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 传统多头注意力，不使用 repeat_kv
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class TraditionalAttentionDecoderLayer(nn.Module):

    def __init__(self, config: Qwen2Config, layer_idx: int, norm_class, mlp_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        # 使用传统多头注意力
        self.self_attn = TraditionalMultiHeadAttention(
            config=config, layer_idx=layer_idx)
        self.mlp = mlp_class(config)
        self.input_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(
            config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] if hasattr(
            config, "layer_types") else "full_attention"

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
        output_attentions=False, use_cache=False, **kwargs
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2AblationNoGQA(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 复制配置并修改：使用 MHA（num_key_value_heads = num_attention_heads）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # 关键修改：MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding)

        self.qwen2_model = Qwen2BaseModel(
            self.config, Qwen2RMSNorm, Qwen2MLP, Qwen2RotaryEmbedding, TraditionalAttentionDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class Qwen2AblationNoRMSNorm(nn.Module):

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Force traditional MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2MLP, Qwen2RotaryEmbedding)

        self.qwen2_model = Qwen2BaseModel(
            self.config, TraditionalLayerNorm, Qwen2MLP, Qwen2RotaryEmbedding, BaseDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class Qwen2AblationNoSwiGLU(nn.Module):

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Force traditional MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            hidden_act="gelu",
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RMSNorm, Qwen2RotaryEmbedding)

        self.qwen2_model = Qwen2BaseModel(
            self.config, Qwen2RMSNorm, TraditionalMLP, Qwen2RotaryEmbedding, BaseDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class Qwen2AblationNoRoPE(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 复制配置（无需修改）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Force traditional MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2MLP, Qwen2RMSNorm)

        self.qwen2_model = Qwen2BaseModel(
            self.config, Qwen2RMSNorm, Qwen2MLP, TraditionalPositionEmbedding, BaseDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class Qwen2AblationNoPreLN(nn.Module):

    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 复制配置（无需修改）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # Force traditional MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding)

        self.qwen2_model = Qwen2BaseModel(
            self.config, Qwen2RMSNorm, Qwen2MLP, Qwen2RotaryEmbedding, PostLNDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.qwen2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_gpu_memory():
    """ Get current GPU usage (MB). """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


if __name__ == "__main__":
    print("=" * 80)
    print("测试 Qwen2 消融实验模型 (transformers 4.41.2)")
    print("=" * 80)

    # 创建基础配置 (~11M 参数量)
    # 配置选择经验：hidden_size=256, layers=8, heads=4, intermediate=1152
    # 遵循 Llama/Qwen 设计：intermediate_size 通常为 hidden_size 的 4-5 倍
    base_config = Qwen2Config(
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=1152,  # 4.5x hidden_size，平衡表达能力与参数量
        max_position_embeddings=2048,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
    )

    # ========== 第一部分：参数量测试 ==========
    print("\n" + "=" * 80)
    print("第一部分：参数量测试")
    print("=" * 80)

    # 测试官方 Qwen2Model（基准）
    print("\n1. 测试 Qwen2Model (官方基准)")
    print("-" * 40)
    from transformers import Qwen2Model as OfficialQwen2Model

    model_official = OfficialQwen2Model(base_config)
    total_params, trainable_params = count_parameters(model_official)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试 Qwen2StructureModel（与 Qwen2Model 一致）
    print("\n2. 测试 Qwen2StructureModel (本实现)")
    print("-" * 40)
    model_structure = Qwen2StructureModel(base_config)
    total_params, trainable_params = count_parameters(model_structure)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试无 GQA 模型
    print("\n3. 测试无 GQA 模型")
    print("-" * 40)
    model_no_gqa = Qwen2AblationNoGQA(base_config)
    total_params, trainable_params = count_parameters(model_no_gqa)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试无 RMSNorm 模型
    print("\n4. 测试无 RMSNorm 模型")
    print("-" * 40)
    model_no_rmsnorm = Qwen2AblationNoRMSNorm(base_config)
    total_params, trainable_params = count_parameters(model_no_rmsnorm)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试无 SwiGLU 模型
    print("\n5. 测试无 SwiGLU 模型")
    print("-" * 40)
    model_no_swiglu = Qwen2AblationNoSwiGLU(base_config)
    total_params, trainable_params = count_parameters(model_no_swiglu)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试无 RoPE 模型
    print("\n6. 测试无 RoPE 模型")
    print("-" * 40)
    model_no_rope = Qwen2AblationNoRoPE(base_config)
    total_params, trainable_params = count_parameters(model_no_rope)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试无 PreLN 模型
    print("\n7. 测试无 PreLN 模型")
    print("-" * 40)
    model_no_preln = Qwen2AblationNoPreLN(base_config)
    total_params, trainable_params = count_parameters(model_no_preln)
    print(f"总参数量：{total_params:,} ({total_params / 1e6:.2f}M)")

    # 参数量对比汇总表
    print("\n" + "=" * 80)
    print("参数量对比汇总表")
    print("=" * 80)
    models_params = [
        ("Qwen2Model (官方)", model_official),
        ("Qwen2StructureModel", model_structure),
        ("无 GQA 模型", model_no_gqa),
        ("无 RMSNorm 模型", model_no_rmsnorm),
        ("无 SwiGLU 模型", model_no_swiglu),
        ("无 RoPE 模型", model_no_rope),
        ("无 PreLN 模型", model_no_preln)
    ]

    print(f"{'模型名称':<25} | {'总参数量':>15} | {'可训练参数':>15} | {'差异 (vs 官方)':>18}")
    print("-" * 80)
    official_params = sum(p.numel() for p in model_official.parameters())
    for name, model in models_params:
        total, trainable = count_parameters(model)
        diff = total - official_params
        diff_str = f"{diff:+,}" if diff != 0 else "0 (一致)"
        print(f"{name:<25} | {total:>15,} | {trainable:>15,} | {diff_str:>18}")

    # ========== 第二部分：显存占用测试 ==========
    print("\n" + "=" * 80)
    print("第二部分：显存占用测试")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n⚠️  未检测到 CUDA 设备，跳过显存占用测试")
        print("如需测试显存，请在有 GPU 的设备上运行")
    else:
        print(f"\n✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU 总显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # 测试配置
        batch_size = 32
        seq_len = 512
        print(f"\n测试条件:")
        print(f"  - batch_size: {batch_size}")
        print(f"  - seq_len: {seq_len}")
        print(f"  - use_cache: False (所有模型统一设置)")
        print(f"  - output_hidden_states: False")
        print(f"  - output_attentions: False")

        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print("\n" + "-" * 80)
        print("各模型显存占用测试")
        print("-" * 80)

        memory_results = []

        for name, model in models_params:
            # 移动到 GPU
            model.cuda()

            # 清空缓存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 创建输入
            x = torch.randint(0, 1000, (batch_size, seq_len)).cuda()

            # 前向传播
            model.eval()
            with torch.no_grad():
                outputs = model(x, use_cache=False, output_hidden_states=False, output_attentions=False)
                torch.cuda.synchronize()

            # 记录显存
            mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2

            memory_results.append((name, mem_allocated, mem_reserved))
            print(f"{name:<25} | 分配：{mem_allocated:>8.2f} MB | 预留：{mem_reserved:>8.2f} MB")

            # 移回 CPU，清空缓存
            model.cpu()
            del x
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 显存对比汇总表
        print("\n" + "-" * 80)
        print("显存占用对比汇总 (以 Qwen2Model 为基准)")
        print("-" * 80)

        baseline_mem = memory_results[0][1]  # Qwen2Model 的显存
        print(f"{'模型名称':<25} | {'分配显存 (MB)':>15} | {'与基准差异':>15} | {'差异百分比':>12}")
        print("-" * 80)

        for name, mem_alloc, mem_res in memory_results:
            diff = mem_alloc - baseline_mem
            diff_pct = (diff / baseline_mem) * 100 if baseline_mem > 0 else 0
            diff_str = f"{diff:+.2f} MB" if diff != 0 else "0.00 MB"
            print(f"{name:<25} | {mem_alloc:>15.2f} | {diff_str:>15} | {diff_pct:>11.2f}%")

        # 找出显存占用最大和最小的模型
        max_mem_model = max(memory_results, key=lambda x: x[1])
        min_mem_model = min(memory_results, key=lambda x: x[1])

        print("\n" + "=" * 80)
        print("显存占用分析")
        print("=" * 80)
        print(f"显存占用最大：{max_mem_model[0]} ({max_mem_model[1]:.2f} MB)")
        print(f"显存占用最小：{min_mem_model[0]} ({min_mem_model[1]:.2f} MB)")
        print(
            f"最大差异：{max_mem_model[1] - min_mem_model[1]:.2f} MB ({((max_mem_model[1] - min_mem_model[1]) / min_mem_model[1] * 100):.2f}%)")

        if max_mem_model[1] - min_mem_model[1] < 5:  # 差异小于 5MB 认为是一致的
            print("\n✓ 结论：所有模型显存占用基本一致，修复成功！")
        else:
            print(f"\n⚠️  注意：模型间仍存在显存差异，可能需要进一步检查")

    # ========== 第三部分：前向传播测试 ==========
    print("\n" + "=" * 80)
    print("第三部分：前向传播测试")
    print("=" * 80)

    batch_size = 2
    seq_len = 32
    x = torch.randint(0, 1000, (batch_size, seq_len))  # [batch_size, seq_len]

    print(f"\n输入形状：{x.shape}")
    print("-" * 40)

    for name, model in models_params:
        try:
            outputs = model(x)  # 返回 BaseModelOutputWithPast
            last_hidden_state = outputs.last_hidden_state
            print(f"{name:<25} | 输出形状：{last_hidden_state.shape} ✓")
        except Exception as e:
            print(f"{name:<25} | 前向传播失败：{e} ✗")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


# ============================================================================
# Qwen2AblationOnlyWithXXX 系列
# 在简化的 Qwen2 基础上逐个添加 Qwen2 的特性
#
# 设计理念：
# - 基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
# - 每个模型只添加一个 Qwen2 特性，其他保持基础架构
# - 这样可以独立测试每个特性的影响
#
# 基础架构配置（所有模型的默认设置）：
# - num_key_value_heads = num_attention_heads (传统 MHA)
# - LayerNorm (非 RMSNorm)
# - TraditionalMLP (非 SwiGLU)
# - TraditionalPositionEmbedding (非 RoPE)
# - PostLNDecoderLayer (非 Pre-LN)
# ============================================================================

class Qwen2AblationOnlyWithGQA(nn.Module):
    """
    Qwen2 消融模型：只添加 GQA 特性

    基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
    添加特性：GQA（num_key_value_heads < num_attention_heads）

    参数量影响：
    - K/V 投影从 num_attention_heads 头减少到 num_key_value_heads 头
    - 每层减少：2 * (num_attention_heads - num_key_value_heads) * head_dim * hidden_size
    - 以 8 头->2 头为例：每层减少约 0.5M 参数，8 层总计减少约 4M 参数

    Args:
        config: Qwen2Config 配置对象（应包含 num_key_value_heads < num_attention_heads）
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 只修改 num_key_value_heads，使用传入的 GQA 配置
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,  # 使用传入的 GQA 配置
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        # 使用基础组件
        self.qwen2_model = Qwen2BaseModel(
            self.config, TraditionalLayerNorm, TraditionalMLP, TraditionalPositionEmbedding, PostLNDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.qwen2_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Qwen2AblationOnlyWithRMSNorm(nn.Module):
    """
    Qwen2 消融模型：只添加 RMSNorm 特性

    基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
    添加特性：RMSNorm（替代 LayerNorm）

    参数量影响：
    - RMSNorm 与 LayerNorm 参数量相同（都只有 weight，无 bias）
    - 参数量基本无变化

    Args:
        config: Qwen2Config 配置对象
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 强制使用传统 MHA（num_key_value_heads = num_attention_heads）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # 强制使用传统 MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        # 只使用 RMSNorm，其他保持基础组件
        self.qwen2_model = Qwen2BaseModel(
            self.config, Qwen2RMSNorm, TraditionalMLP, TraditionalPositionEmbedding, PostLNDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.qwen2_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Qwen2AblationOnlyWithSwiGLU(nn.Module):
    """
    Qwen2 消融模型：只添加 SwiGLU 特性

    基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
    添加特性：SwiGLU（替代 TraditionalMLP）

    参数量影响：
    - SwiGLU (Qwen2MLP): gate_proj + up_proj + down_proj (3 个投影)
    - TraditionalMLP: gate_proj + down_proj (2 个投影)
    - 每层增加：intermediate_size * hidden_size 参数
    - 以 hidden_size=576, intermediate_size=2304 为例：每层增加约 1.3M 参数，8 层总计增加约 10.6M

    Args:
        config: Qwen2Config 配置对象
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 强制使用传统 MHA（num_key_value_heads = num_attention_heads）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # 强制使用传统 MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        # 只使用 Qwen2MLP (SwiGLU)，其他保持基础组件
        self.qwen2_model = Qwen2BaseModel(
            self.config, TraditionalLayerNorm, Qwen2MLP, TraditionalPositionEmbedding, PostLNDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.qwen2_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Qwen2AblationOnlyWithRoPE(nn.Module):
    """
    Qwen2 消融模型：只添加 RoPE 特性

    基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
    添加特性：RoPE（替代 TraditionalPositionEmbedding）

    参数量影响：
    - RoPE 无额外参数（使用正弦/余弦函数，register_buffer 不计入参数量）
    - 参数量基本无变化

    Args:
        config: Qwen2Config 配置对象
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 强制使用传统 MHA（num_key_value_heads = num_attention_heads）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # 强制使用传统 MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        # 只使用 Qwen2RotaryEmbedding (RoPE)，其他保持基础组件
        self.qwen2_model = Qwen2BaseModel(
            self.config, TraditionalLayerNorm, TraditionalMLP, Qwen2RotaryEmbedding, PostLNDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.qwen2_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class Qwen2AblationOnlyWithPreLN(nn.Module):
    """
    Qwen2 消融模型：只添加 Pre-LN 特性

    基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
    添加特性：Pre-LN（替代 Post-LN）

    参数量影响：
    - Pre-LN 与 Post-LN 只是归一化位置不同
    - 参数量基本无变化

    Args:
        config: Qwen2Config 配置对象
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        # 强制使用传统 MHA（num_key_value_heads = num_attention_heads）
        self.config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # 强制使用传统 MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        # 只使用 BaseDecoderLayer (Pre-LN)，其他保持基础组件
        self.qwen2_model = Qwen2BaseModel(
            self.config, TraditionalLayerNorm, TraditionalMLP, TraditionalPositionEmbedding, BaseDecoderLayer
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return self.qwen2_model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ============================================================================
# Qwen2CC 系列 - Qwen2 + Classifier + Regressor
# 在 Qwen2BaseModel 基础上添加分类头和回归头，与 Houyi2 模型对齐
# ============================================================================

class Qwen2CCBaseModel(nn.Module, SetCompRequiresGradMixin):
    """
    Qwen2 基类模型 - 添加分类器 (Classifier) 和回归器 (Regressor)。

    基于 Qwen2BaseModel，添加：
    - 分类器：用于二分类任务
    - 回归器：用于特征回归任务

    Args:
        config: Qwen2Config 配置对象
        n_out_features: 回归器输出特征维度
        n_hidden_layers_of_classifier: 分类器隐藏层数
        n_hidden_layers_of_regressor: 回归器隐藏层数
        regressor_output_sigmoid: 回归器输出是否使用 Sigmoid 激活
        use_qwen2_classifier: 是否使用 Qwen2 编码器作为分类器头
        use_qwen2_regressor: 是否使用 Qwen2 编码器作为回归器头
    """

    def __init__(
        self,
        config: Qwen2Config,
        n_in_features: int,
        n_out_features: int,
        n_hidden_layers_of_classifier: int = 1,
        n_hidden_layers_of_regressor: int = 1,
        regressor_output_sigmoid: bool = False,
        use_qwen2_classifier: bool = False,
        use_qwen2_regressor: bool = False,
        norm_class=TraditionalLayerNorm,
        mlp_class=TraditionalMLP,
        position_embedding_class=TraditionalPositionEmbedding,
        decoder_layer_class=PostLNDecoderLayer,
        is_causal: bool = True
    ):
        super().__init__()
        self.config = config

        self.component_parts = ModelParts()
        self._build_input_embedding(n_in_features, config.hidden_size)

        # 创建 Qwen2 主干网络
        self.qwen2_model = Qwen2BaseModel(
            config, norm_class, mlp_class, position_embedding_class, decoder_layer_class, is_causal
        )
        self.component_parts.add_backbone_comps(
            {"qwen2_model": self.qwen2_model}
        )

        # 构建分类器
        self._use_qwen2_classifier = use_qwen2_classifier
        self._build_classifier(
            n_hidden_layers_of_classifier, use_qwen2_classifier
        )

        # 构建回归器
        self._use_qwen2_regressor = use_qwen2_regressor
        self._build_regressor(
            n_out_features, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_qwen2_regressor
        )

        self.last_hidden_state = None

    def _build_input_embedding(self, n_in_features: int, hidden_size: int):
        self.input_embedding = torch.nn.Sequential(
            torch.nn.Linear(n_in_features, hidden_size),
            torch.nn.ReLU()
        )
        self.component_parts.add_backbone_comps(
            {"input_embedding": self.input_embedding}
        )

    def _build_classifier(
        self,
        n_layers: int,
        use_qwen2: bool
    ):
        """构建分类器"""
        hidden_size = self.config.hidden_size

        if not use_qwen2:
            self.point_classifier = nn.Sequential()
            for i in range(n_layers - 1):
                self.point_classifier += nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                )
            self.point_classifier += nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2),
                nn.Softmax(dim=-1)
            )
            self.component_parts.add_classify_comps({
                "point_classifier": self.point_classifier
            })
        else:
            classifier_config = Qwen2Config(
                hidden_size=self.config.hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
                num_key_value_heads=self.config.num_key_value_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                vocab_size=self.config.vocab_size,
                rms_norm_eps=self.config.rms_norm_eps,
                rope_theta=getattr(self.config, 'rope_theta', 1000000),
                layer_types=["full_attention"] * n_layers,
            )
            self.point_classifier = Qwen2BaseModel(
                classifier_config, Qwen2RMSNorm, Qwen2MLP, Qwen2RotaryEmbedding, BaseDecoderLayer, is_causal=False
            )
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_size, 2),
                nn.ReLU(),
                nn.Softmax(dim=-1)
            )
            self.component_parts.add_classify_comps({
                "point_classifier": self.point_classifier,
                "classifier_head": self.classifier_head
            })

    def _build_regressor(
        self,
        n_out_features: int,
        n_layers: int,
        use_sigmoid: bool,
        use_qwen2: bool
    ):
        """构建回归器"""
        hidden_size = self.config.hidden_size

        if not use_qwen2:
            self.regressor = nn.Sequential()
            for i in range(n_layers - 1):
                self.regressor += nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                )
            self.regressor += nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, n_out_features)
            )
            if use_sigmoid:
                self.regressor += nn.Sigmoid()

            self.component_parts.add_regress_comps({
                "regressor": self.regressor
            })
        else:
            regressor_config = Qwen2Config(
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
                num_key_value_heads=self.config.num_key_value_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                vocab_size=self.config.vocab_size,
                rms_norm_eps=self.config.rms_norm_eps,
                rope_theta=getattr(self.config, 'rope_theta', 1000000),
                layer_types=["full_attention"] * n_layers,
            )
            self.regressor = Qwen2BaseModel(
                regressor_config, Qwen2RMSNorm, Qwen2MLP, Qwen2RotaryEmbedding, BaseDecoderLayer, is_causal=False
            )
            self.regressor_head = nn.Sequential(
                nn.Linear(hidden_size, n_out_features)
            )
            if use_sigmoid:
                self.regressor_head += nn.Sigmoid()

            self.component_parts.add_regress_comps({
                "regressor": self.regressor,
                "regressor_head": self.regressor_head
            })

    def forward(self, inputs, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        前向传播。

        Args:
            inputs: 输入 token features，形状 (batch_size, seq_len, n_features)
            attention_mask: 注意力掩码
            position_ids: 位置 ID 张量
            past_key_values: 缓存的 K/V 状态
            inputs_embeds: 输入嵌入（替代 input_ids）
            use_cache: 是否使用 cache
            output_attentions: 是否返回注意力权重
            output_hidden_states: 是否返回隐藏状态
            return_dict: 是否使用返回字典格式

        Returns:
            tuple: (probs, regressed_features, last_hidden_state)
                - probs: 分类器输出概率，形状 (batch_size, seq_len, 2)
                - regressed_features: 回归器输出，形状 (batch_size, seq_len, n_out_features)
                - last_hidden_state: 主干网络最后一层输出
        """
        inputs_embeds = self.input_embedding(inputs)
        outputs = self.qwen2_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_clone = last_hidden_state.clone()

        if self._use_qwen2_classifier:
            probs_output = self.point_classifier(
                inputs_embeds=last_hidden_state_clone,
                attention_mask=attention_mask,
                return_dict=True
            )
            probs = self.classifier_head(probs_output.last_hidden_state)
        else:
            probs = self.point_classifier(last_hidden_state_clone)

        if self._use_qwen2_regressor:
            regressor_output = self.regressor(
                inputs_embeds=last_hidden_state_clone,
                attention_mask=attention_mask,
                return_dict=True
            )
            regressed_features = self.regressor_head(regressor_output.last_hidden_state)
        else:
            regressed_features = self.regressor(last_hidden_state_clone)

        self.last_hidden_state = last_hidden_state_clone.detach()

        return probs, regressed_features, last_hidden_state_clone

    def get_attention(self, mean_over_heads: bool = False):
        """
        获取最后一层注意力权重。

        Args:
            mean_over_heads: 是否对所有注意力头取平均

        Returns:
            注意力权重张量或 None
            - 如果 mean_over_heads=True: (batch_size, seq_len, seq_len)
            - 如果 mean_over_heads=False: (batch_size, num_heads, seq_len, seq_len)
        """
        if not hasattr(self, 'qwen2_model') or self.qwen2_model is None:
            return None

        # Qwen2BaseModel 不直接存储注意力权重，需要从 decoder layers 获取
        # 这里返回最后一层的注意力权重（如果有的话）
        if hasattr(self.qwen2_model, 'layers') and len(self.qwen2_model.layers) > 0:
            last_layer = self.qwen2_model.layers[-1]
            if hasattr(last_layer.self_attn, 'last_attention_weights'):
                attn = last_layer.self_attn.last_attention_weights
                if mean_over_heads and attn is not None:
                    return attn.mean(dim=1)
                return attn
        return None

    def get_correlation(self):
        if self.last_hidden_state is None:
            return None

        normalized_state = (
            self.last_hidden_state /
            torch.norm(self.last_hidden_state, dim=-1, keepdim=True)
        )
        # shape=(batch_size, sequence_length, hidden_size)
        corr = torch.matmul(
            normalized_state,
            normalized_state.transpose(-1, -2)
        )
        return corr  # shape=(batch_size, seq_len, seq_len)


# ============================================================================
# Qwen2CCModelPlain - 关闭所有 Qwen2 特性（基础架构）
# ============================================================================

class Qwen2CCModelPlain(Qwen2CCBaseModel):
    """
    Qwen2CC 基础模型 - 关闭所有 Qwen2 特性。

    基础架构：
    - 传统 MHA（num_key_value_heads = num_attention_heads）
    - LayerNorm（非 RMSNorm）
    - TraditionalMLP（非 SwiGLU）
    - 传统位置编码（非 RoPE）
    - Post-LN（非 Pre-LN）
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        # 强制使用传统 MHA
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,
            mlp_class=TraditionalMLP,
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=PostLNDecoderLayer,
        )


# ============================================================================
# Qwen2CCModelWithXXX 系列 - 在基础架构上添加单个 Qwen2 特性
# ============================================================================

class Qwen2CCModelWithGQA(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：只添加 GQA 特性。

    基础架构：传统 MHA + LayerNorm + TraditionalMLP + 传统位置编码 + Post-LN
    添加特性：GQA（num_key_value_heads < num_attention_heads）
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False,
                 num_key_value_heads: int = 2):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,
            mlp_class=TraditionalMLP,
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=PostLNDecoderLayer,
        )


class Qwen2CCModelWithRMSNorm(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：只添加 RMSNorm 特性。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=Qwen2RMSNorm,
            mlp_class=TraditionalMLP,
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=PostLNDecoderLayer,
        )


class Qwen2CCModelWithSwiGLU(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：只添加 SwiGLU 特性。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,
            mlp_class=Qwen2MLP,
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=PostLNDecoderLayer,
        )


class Qwen2CCModelWithRoPE(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：只添加 RoPE 特性。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,
            mlp_class=TraditionalMLP,
            position_embedding_class=Qwen2RotaryEmbedding,
            decoder_layer_class=PostLNDecoderLayer,
        )


class Qwen2CCModelWithPreLN(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：只添加 Pre-LN 特性。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,
            mlp_class=TraditionalMLP,
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=BaseDecoderLayer,
        )


# ============================================================================
# Qwen2CCModelNoXXX 系列 - 在全特性基础上关闭单个 Qwen2 特性
# ============================================================================

class Qwen2CCModelNoGQA(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：关闭 GQA 特性（使用传统 MHA）。

    基础架构（全特性）：GQA + RMSNorm + SwiGLU + RoPE + Pre-LN
    关闭特性：GQA → 使用传统 MHA
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # 关闭 GQA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=Qwen2RMSNorm,
            mlp_class=Qwen2MLP,
            position_embedding_class=Qwen2RotaryEmbedding,
            decoder_layer_class=BaseDecoderLayer,
        )


class Qwen2CCModelNoRMSNorm(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：关闭 RMSNorm 特性（使用 LayerNorm）。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,  # 关闭 RMSNorm
            mlp_class=Qwen2MLP,
            position_embedding_class=Qwen2RotaryEmbedding,
            decoder_layer_class=BaseDecoderLayer,
        )


class Qwen2CCModelNoSwiGLU(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：关闭 SwiGLU 特性（使用 TraditionalMLP）。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=Qwen2RMSNorm,
            mlp_class=TraditionalMLP,  # 关闭 SwiGLU
            position_embedding_class=Qwen2RotaryEmbedding,
            decoder_layer_class=BaseDecoderLayer,
        )


class Qwen2CCModelNoRoPE(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：关闭 RoPE 特性（使用传统位置编码）。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=Qwen2RMSNorm,
            mlp_class=Qwen2MLP,
            position_embedding_class=TraditionalPositionEmbedding,  # 关闭 RoPE
            decoder_layer_class=BaseDecoderLayer,
        )


class Qwen2CCModelNoPreLN(Qwen2CCBaseModel):
    """
    Qwen2CC 模型：关闭 Pre-LN 特性（使用 Post-LN）。
    """

    def __init__(self, config: Qwen2Config, n_in_features: int, n_out_features: int,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_qwen2_classifier: bool = False,
                 use_qwen2_regressor: bool = False):
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=Qwen2RMSNorm,
            mlp_class=Qwen2MLP,
            position_embedding_class=Qwen2RotaryEmbedding,
            decoder_layer_class=PostLNDecoderLayer,  # 关闭 Pre-LN
        )


# ============================================================================
# Houyi2CCModel - Houyi2CCBaseModel + SwiGLU（双向注意力）
# ============================================================================

class Qwen2CCBaselineModel(Qwen2CCBaseModel):
    """
    Qwen2CC 模型 - 基于 Qwen2 主干，添加 Pre-LN + 双向注意力。

    架构特点：
    - Qwen2 主干（双向注意力）
    - Post-LN 结构（归一化在注意力之前）

    与 Qwen2 的区别：
    - 双向注意力（Qwen2 是单向因果注意力）
    - Post-LN 结构（Qwen2 是 Pre-LN）

    适用于：
    - 序列分类任务
    - 回归任务
    - 需要双向上下文的场景

    Args:
        config: Qwen2Config 配置对象
        n_out_features: 回归器输出特征维度
        n_hidden_layers_of_classifier: 分类器隐藏层数
        n_hidden_layers_of_regressor: 回归器隐藏层数
        regressor_output_sigmoid: 回归器输出是否使用 Sigmoid
        use_qwen2_classifier: 是否使用 Qwen2 编码器作为分类器头
        use_qwen2_regressor: 是否使用 Qwen2 编码器作为回归器头
    """

    def __init__(
        self,
        config: Qwen2Config,
        n_in_features: int,
        n_out_features: int,
        n_hidden_layers_of_classifier: int = 1,
        n_hidden_layers_of_regressor: int = 1,
        regressor_output_sigmoid: bool = False,
        use_qwen2_classifier: bool = False,
        use_qwen2_regressor: bool = False,
    ):
        # 配置：使用 GQA + RMSNorm + SwiGLU + Pre-LN + 双向注意力
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # use MHA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers,
            hidden_act="gelu"
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=TraditionalLayerNorm,
            mlp_class=TraditionalMLP,
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=PostLNDecoderLayer,  # use Post-LN
            is_causal=False,  # use bi-direc. attention
        )


# ============================================================================
# Qwen2CCModel - Qwen2CCBaseModel + PreLN + SwiGLU + 双向注意力
# ============================================================================

class Qwen2CCModel(Qwen2CCBaseModel):
    """
    Qwen2CC 模型 - 基于 Qwen2 主干，添加 Pre-LN + SwiGLU + 双向注意力。

    架构特点：
    - Qwen2 主干（双向注意力）
    - Pre-LN 结构（归一化在注意力之前）
    - SwiGLU 激活函数
    - RMSNorm 归一化

    与 Qwen2 的区别：
    - 双向注意力（Qwen2 是单向因果注意力）
    - Pre-LN 结构（Qwen2 是 Post-LN）

    适用于：
    - 序列分类任务
    - 回归任务
    - 需要双向上下文的场景

    Args:
        config: Qwen2Config 配置对象
        n_out_features: 回归器输出特征维度
        n_hidden_layers_of_classifier: 分类器隐藏层数
        n_hidden_layers_of_regressor: 回归器隐藏层数
        regressor_output_sigmoid: 回归器输出是否使用 Sigmoid
        use_qwen2_classifier: 是否使用 Qwen2 编码器作为分类器头
        use_qwen2_regressor: 是否使用 Qwen2 编码器作为回归器头
    """

    def __init__(
        self,
        config: Qwen2Config,
        n_in_features: int,
        n_out_features: int,
        n_hidden_layers_of_classifier: int = 1,
        n_hidden_layers_of_regressor: int = 1,
        regressor_output_sigmoid: bool = False,
        use_qwen2_classifier: bool = False,
        use_qwen2_regressor: bool = False,
    ):
        # 配置：使用 GQA + RMSNorm + SwiGLU + Pre-LN + 双向注意力
        config = Qwen2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,  # use GQA
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.0,
            use_cache=False,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, 'rope_theta', 1000000),
            layer_types=["full_attention"] * config.num_hidden_layers,
        )

        super().__init__(
            config, n_in_features, n_out_features, n_hidden_layers_of_classifier,
            n_hidden_layers_of_regressor, regressor_output_sigmoid,
            use_qwen2_classifier, use_qwen2_regressor,
            norm_class=Qwen2RMSNorm,  # use RMSNorm
            mlp_class=Qwen2MLP,  # use SwiGLU MLP
            position_embedding_class=TraditionalPositionEmbedding,
            decoder_layer_class=BaseDecoderLayer,  # use Pre-LN
            is_causal=False,  # use bi-direc. attention
        )
