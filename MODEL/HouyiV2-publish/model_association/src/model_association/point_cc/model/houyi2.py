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
Houyi2 model implementation with customizable architecture components.

This module implements the Houyi2 architecture with support for various ablation studies:
- Grouped Query Attention (GQA)
- RMSNorm vs LayerNorm
- SwiGLU vs traditional MLP
- RoPE vs traditional position embeddings
- Pre-LN vs Post-LN normalization

Adapted for transformers==4.41.2, torch==2.4.0

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


# ============================================================================
# Component Definitions
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) implementation.
    
    GQA reduces memory and computation by having multiple query heads share
    fewer key-value heads. When num_key_value_heads equals num_attention_heads,
    this becomes standard multi-head attention.
    
    Args:
        config: BERT configuration object
        is_causal: Whether to use causal masking
        layer_idx: Layer index for cache management
    
    Attributes:
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        num_key_value_groups: Ratio of query heads to key-value heads
        attention_head_size: Dimension per attention head
    
    Example:
        >>> config = BertConfig(hidden_size=512, num_attention_heads=8, num_key_value_heads=2)
        >>> attn = GroupedQueryAttention(config)
        >>> hidden = torch.randn(2, 10, 512)
        >>> output, attn_weights = attn(hidden)
    """

    def __init__(self, config: BertConfig, is_causal: bool = False, layer_idx: Optional[int] = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size ** -0.5
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_causal = is_causal

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for grouped query attention.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_embeddings: Position embeddings as (cos, sin, position_ids) tuple
        
        Returns:
            Tuple of:
                - context_layer: Attention output of shape (batch_size, seq_len, hidden_size)
                - attention_probs: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(
            hidden_shape[0], -1, self.num_key_value_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(
            hidden_shape[0], -1, self.num_key_value_heads, self.attention_head_size).transpose(1, 2)

        # Repeat K and V to match number of query heads
        if self.num_key_value_groups > 1:
            key_layer = key_layer.repeat_interleave(self.num_key_value_groups, dim=1)
            value_layer = value_layer.repeat_interleave(self.num_key_value_groups, dim=1)

        # Apply RoPE position embeddings
        if position_embeddings is not None and len(position_embeddings) >= 3:
            cos, sin, position_ids = position_embeddings[:3]
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids,
                                                          unsqueeze_dim=1)
        elif position_embeddings is not None and len(position_embeddings) == 2:
            cos, sin = position_embeddings
            batch_size, seq_len = query_layer.shape[0], query_layer.shape[2]
            position_ids = torch.arange(seq_len, device=query_layer.device).unsqueeze(0).expand(batch_size, -1)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids,
                                                          unsqueeze_dim=1)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(*input_shape, -1)

        return context_layer, attention_probs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Unlike standard LayerNorm, RMSNorm does not subtract the mean.
    It only normalizes by the root mean square, which can be more efficient
    while maintaining similar performance.
    
    Args:
        hidden_size: Hidden dimension size
        eps: Epsilon for numerical stability (default: 1e-6)
    
    Attributes:
        weight: Learnable scaling parameter of shape (hidden_size,)
        variance_epsilon: Epsilon value for numerical stability
    
    Example:
        >>> norm = RMSNorm(hidden_size=512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = norm(x)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for RMSNorm.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Normalized tensor of same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU (Swish Gated Linear Unit) MLP.
    
    Implements the SwiGLU activation: SwiGLU(x) = Swish(W_g @ x) ⊗ (W_u @ x)
    where ⊗ denotes element-wise multiplication.
    
    Args:
        config: BERT configuration object
    
    Attributes:
        gate_proj: Linear projection for gate
        up_proj: Linear projection for up
        down_proj: Linear projection for down
    
    Example:
        >>> config = BertConfig(hidden_size=512, intermediate_size=2048)
        >>> mlp = SwiGLUMLP(config)
        >>> x = torch.randn(2, 10, 512)
        >>> output = mlp(x)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return x


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    RoPE encodes positional information by rotating query and key vectors
    in the attention mechanism. This provides relative positional encoding
    with good extrapolation properties.
    
    Args:
        config: BERT configuration object
    
    Attributes:
        inv_freq: Inverse frequency buffer for RoPE computation
        _cos_cache: Cached cosine values
        _sin_cache: Cached sine values
    
    Example:
        >>> config = BertConfig(hidden_size=512, num_attention_heads=8)
        >>> rope = RotaryPositionEmbedding(config)
        >>> x = torch.randn(2, 10, 512)
        >>> cos, sin = rope(x)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache = None
        self._sin_cache = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cached cosine and sine values if needed."""
        if self._cos_cache is not None and self._cos_cache.shape[0] >= seq_len:
            return self._cos_cache, self._sin_cache

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        return self._cos_cache, self._sin_cache

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for RoPE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_ids: Position IDs (not used, kept for API compatibility)
        
        Returns:
            Tuple of (cos, sin) tensors for RoPE application
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        device, dtype = x.device, x.dtype

        cos, sin = self._update_cos_sin_cache(seq_len, device, dtype)
        return cos, sin


# ============================================================================
# LayerConfig - Layer Configuration
# ============================================================================

class LayerConfig:
    """Layer configuration for controlling architecture components.
    
    This dataclass controls which architectural features are enabled in the
    Houyi2 model layers.
    
    Args:
        use_gqa: Enable Grouped Query Attention (default: False)
        use_rmsnorm: Enable RMSNorm instead of LayerNorm (default: False)
        use_swiglu: Enable SwiGLU instead of traditional MLP (default: False)
        use_rope: Enable RoPE position embeddings (default: False)
        use_preln: Enable Pre-LN instead of Post-LN (default: False)
        num_key_value_heads: Number of key-value heads for GQA (default: 8)
    
    Example:
        >>> config = LayerConfig(
        ...     use_gqa=True,
        ...     use_rmsnorm=True,
        ...     use_swiglu=True,
        ...     use_preln=True
        ... )
    """

    def __init__(
        self,
        use_gqa: bool = False,
        use_rmsnorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_preln: bool = False,
        num_key_value_heads: int = 8,
    ):
        self.use_gqa = use_gqa
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        self.use_rope = use_rope
        self.use_preln = use_preln
        self.num_key_value_heads = num_key_value_heads


# ============================================================================
# Houyi2EncoderLayer - Encoder Layer
# ============================================================================

class Houyi2EncoderLayer(nn.Module):
    """Houyi2 encoder layer with customizable architecture.
    
    This layer supports various architectural configurations through LayerConfig:
    - GQA vs standard multi-head attention
    - RMSNorm vs LayerNorm
    - SwiGLU vs traditional MLP
    - RoPE vs no position embeddings
    - Pre-LN vs Post-LN normalization
    
    Args:
        config: BERT configuration object
        layer_cfg: Layer configuration object
        layer_idx: Layer index
    
    Attributes:
        attention: Grouped query attention module
        mlp: Feed-forward network (SwiGLU or traditional)
        norm1, norm2: Normalization layers
        dropout: Dropout layer
    
    Example:
        >>> config = BertConfig(hidden_size=512, num_attention_heads=8)
        >>> layer_cfg = LayerConfig(use_swiglu=True, use_preln=True)
        >>> layer = Houyi2EncoderLayer(config, layer_cfg)
        >>> x = torch.randn(2, 10, 512)
        >>> output, attn = layer(x)
    """

    def __init__(self, config: BertConfig, layer_cfg: LayerConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_cfg = layer_cfg
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        # Set num_key_value_heads
        if layer_cfg.use_gqa:
            config.num_key_value_heads = layer_cfg.num_key_value_heads
        else:
            config.num_key_value_heads = config.num_attention_heads  # Standard MHA

        # Attention layer
        self.attention = GroupedQueryAttention(
            config, is_causal=config.is_decoder, layer_idx=layer_idx
        )

        # MLP layer
        if layer_cfg.use_swiglu:
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU()
            )
            self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

        # Normalization layers
        norm_class = RMSNorm if layer_cfg.use_rmsnorm else nn.LayerNorm
        if layer_cfg.use_preln:
            # Pre-LN: Normalization before attention and MLP
            self.norm1 = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        else:
            # Post-LN: Normalization after residual connection
            self.norm1 = norm_class(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for encoder layer.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_embeddings: Position embeddings as (cos, sin) tuple
        
        Returns:
            Tuple of:
                - layer_output: Output of shape (batch_size, seq_len, hidden_size)
                - attention_probs: Attention weights
        """
        if self.layer_cfg.use_preln:
            return self._forward_preln(hidden_states, attention_mask, position_embeddings, **kwargs)
        elif self.layer_cfg.use_swiglu:
            return self._forward_swiglu(hidden_states, attention_mask, position_embeddings, **kwargs)
        elif self.layer_cfg.use_rmsnorm:
            return self._forward_rmsnorm(hidden_states, attention_mask, position_embeddings, **kwargs)
        else:
            return self._forward_standard(hidden_states, attention_mask, position_embeddings, **kwargs)

    def _forward_standard(self, hidden_states, attention_mask, position_embeddings, **kwargs):
        """Standard BERT Post-LN architecture."""
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, position_embeddings, **kwargs
        )
        attention_output = self.dropout(attention_output)
        attention_output = self.norm1(attention_output + hidden_states)

        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output, attention_probs

    def _forward_rmsnorm(self, hidden_states, attention_mask, position_embeddings, **kwargs):
        """Post-LN with RMSNorm architecture."""
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, position_embeddings, **kwargs
        )
        attention_output = self.dropout(attention_output)
        attention_output = self.norm1(attention_output + hidden_states)

        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output, attention_probs

    def _forward_swiglu(self, hidden_states, attention_mask, position_embeddings, **kwargs):
        """Post-LN with SwiGLU architecture."""
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, position_embeddings, **kwargs
        )
        attention_output = self.dropout(attention_output)
        attention_output = self.norm1(attention_output + hidden_states)

        residual = attention_output
        mlp_output = self.mlp(self.norm2(attention_output))
        layer_output = residual + mlp_output

        return layer_output, attention_probs

    def _forward_preln(self, hidden_states, attention_mask, position_embeddings, **kwargs):
        """Pre-LN architecture."""
        residual = hidden_states
        hidden_states_norm = self.norm1(hidden_states)
        attention_output, attention_probs = self.attention(
            hidden_states_norm, attention_mask, position_embeddings, **kwargs
        )
        attention_output = self.dropout(attention_output)
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states_norm = self.norm2(hidden_states)
        mlp_output = self.feed_forward_chunk(hidden_states_norm)
        layer_output = residual + mlp_output

        return layer_output, attention_probs

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.
        
        Args:
            attention_output: Input from attention layer
        
        Returns:
            Output after feed-forward transformation
        """
        if self.layer_cfg.use_swiglu:
            mlp_output = self.mlp(attention_output)
            return mlp_output
        else:
            intermediate_output = self.mlp(attention_output)
            output = self.output_dense(intermediate_output)
            return output


# ============================================================================
# Houyi2Encoder - Encoder
# ============================================================================

class Houyi2Encoder(nn.Module):
    """Houyi2 encoder stack.
    
    Stacks multiple Houyi2EncoderLayer modules to form the complete encoder.
    
    Args:
        config: BERT configuration object
        layer_cfg: Layer configuration object
    
    Attributes:
        layer: ModuleList of encoder layers
    
    Example:
        >>> config = BertConfig(hidden_size=512, num_hidden_layers=6)
        >>> layer_cfg = LayerConfig(use_swiglu=True)
        >>> encoder = Houyi2Encoder(config, layer_cfg)
        >>> x = torch.randn(2, 10, 512)
        >>> output, attn = encoder(x)
    """

    def __init__(self, config: BertConfig, layer_cfg: LayerConfig):
        super().__init__()
        self.config = config
        self.layer_cfg = layer_cfg
        self.layer = nn.ModuleList([
            Houyi2EncoderLayer(config, layer_cfg, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through encoder stack.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_embeddings: Position embeddings as (cos, sin) tuple
        
        Returns:
            Tuple of:
                - last_hidden_state: Output of last layer (batch_size, seq_len, hidden_size)
                - last_attention_probs: Attention weights from last layer
        """
        last_attention_probs = None

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states, attention_mask, position_embeddings
            )
            hidden_states = layer_outputs[0]

            if len(layer_outputs) > 1 and layer_outputs[1] is not None:
                last_attention_probs = layer_outputs[1]

        return hidden_states, last_attention_probs


# ============================================================================
# Houyi2BaseModel - Base Model
# ============================================================================

class Houyi2BaseModel(nn.Module):
    """Houyi2 base model with BERT-style encoder.
    
    Provides the base encoder architecture that can be extended for specific tasks.
    This corresponds to Qwen2BaseModel in the Qwen2 implementation.
    
    Args:
        n_in_features: Input feature dimension
        n_out_features: Output feature dimension (not used in base model)
        enc_config: BERT configuration object
        layer_cfg: Layer configuration object
    
    Attributes:
        input_embedding: Input feature embedding network
        bert_encoder: Houyi2 encoder
        rotary_emb: RoPE position embedding (if enabled)
        encoder_last_hidden_state: Cached last hidden state
        encoder_last_attention: Cached last attention weights
    
    Example:
        >>> config = BertConfig(hidden_size=512, num_hidden_layers=6)
        >>> model = Houyi2BaseModel(n_in_features=64, n_out_features=32, enc_config=config)
        >>> x = torch.randn(2, 10, 64)
        >>> hidden, attn = model(x)
    """

    def __init__(
        self,
        n_in_features: int,
        n_out_features: int,
        enc_config: BertConfig,
        layer_cfg: Optional[LayerConfig] = None,
    ):
        super().__init__()
        self.config = enc_config
        self.layer_cfg = layer_cfg if layer_cfg is not None else LayerConfig()

        self.input_embedding = nn.Sequential(
            nn.Linear(n_in_features, enc_config.hidden_size),
            nn.ReLU()
        )

        self.bert_encoder = Houyi2Encoder(enc_config, self.layer_cfg)

        if self.layer_cfg.use_rope:
            self.rotary_emb = RotaryPositionEmbedding(enc_config)
        else:
            self.rotary_emb = None

        self.encoder_last_hidden_state: Optional[torch.Tensor] = None
        self.encoder_last_attention: Optional[torch.Tensor] = None

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through base model.
        
        Args:
            input: Input tensor of shape (batch_size, seq_len, n_in_features)
        
        Returns:
            Tuple of:
                - last_hidden_state: Encoder output (batch_size, seq_len, hidden_size)
                - last_attention_probs: Last layer attention weights
        """
        hidden_states = self.input_embedding(input)

        attention_mask = torch.ones_like(hidden_states[:, :, 0])
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        position_embeddings = None
        if self.rotary_emb is not None:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
            position_embeddings = (cos, sin, position_ids)

        encoder_last_hidden_state, encoder_last_attention = self.bert_encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_embeddings=position_embeddings,
        )

        self.encoder_last_hidden_state = encoder_last_hidden_state.detach()
        self.encoder_last_attention = encoder_last_attention.detach() if encoder_last_attention is not None else None

        return encoder_last_hidden_state, encoder_last_attention


# ============================================================================
# Houyi2CCBaseModel - Classification and Clustering Base Model
# ============================================================================

class Houyi2CCBaseModel(nn.Module):
    """Houyi2 classification and clustering model.
    
    Extends Houyi2BaseModel by adding classifier and regressor heads.
    Both heads can use either dense networks or BERT encoders.
    
    Architecture:
        - Input embedding: Linear + ReLU
        - Houyi2 encoder: Customizable attention/MLP/normalization
        - Classifier head: Point-wise classification (dense or BERT encoder)
        - Regressor head: Feature regression (dense or BERT encoder)
    
    Args:
        n_in_features: Input feature dimension
        n_out_features: Regressor output feature dimension
        enc_config: BERT configuration object
        n_hidden_layers_of_classifier: Number of classifier layers (default: 1)
        n_hidden_layers_of_regressor: Number of regressor layers (default: 1)
        regressor_output_sigmoid: Apply sigmoid to regressor output (default: False)
        use_bert_classifier: Use BERT encoder for classifier (default: False)
        use_bert_regressor: Use BERT encoder for regressor (default: False)
        layer_cfg: Layer configuration object
    
    Attributes:
        input_embedding: Input feature embedding
        bert_encoder: Main Houyi2 encoder
        point_classifier: Classification head
        regressor: Regression head
        encoder_last_hidden_state: Cached encoder output
        encoder_last_attention: Cached attention weights
    
    Example:
        >>> config = BertConfig(hidden_size=512, num_hidden_layers=6)
        >>> model = Houyi2CCBaseModel(
        ...     n_in_features=64,
        ...     n_out_features=32,
        ...     enc_config=config,
        ...     use_bert_classifier=True
        ... )
        >>> x = torch.randn(2, 10, 64)
        >>> probs, features, hidden = model(x)
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
        layer_cfg: Optional[LayerConfig] = None,
    ):
        super().__init__()
        self.config = enc_config
        self.layer_cfg = layer_cfg if layer_cfg is not None else LayerConfig()

        self.input_embedding = nn.Sequential(
            nn.Linear(n_in_features, enc_config.hidden_size),
            nn.ReLU()
        )

        self.bert_encoder = Houyi2Encoder(enc_config, self.layer_cfg)

        if self.layer_cfg.use_rope:
            self.rotary_emb = RotaryPositionEmbedding(enc_config)
        else:
            self.rotary_emb = None

        self._use_bert_classifier = use_bert_classifier
        self._build_classifier(
            n_hidden_layers_of_classifier, use_bert_classifier
        )

        self._use_bert_regressor = use_bert_regressor
        self._build_regressor(
            n_out_features, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_regressor
        )

        self.encoder_last_hidden_state: Optional[torch.Tensor] = None
        self.encoder_last_attention: Optional[torch.Tensor] = None

    def _build_classifier(self, n_layers: int, use_bert: bool) -> None:
        """Build classifier head.
        
        Args:
            n_layers: Number of layers in classifier
            use_bert: Whether to use BERT encoder
        """
        hidden_size = self.config.hidden_size

        if not use_bert:
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
        else:
            classifier_config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size
            )
            self.point_classifier = BertEncoder(classifier_config)
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_size, 2),
                nn.ReLU(),
                nn.Softmax(dim=-1)
            )

    def _build_regressor(
        self,
        n_out_features: int,
        n_layers: int,
        use_sigmoid: bool,
        use_bert: bool
    ) -> None:
        """Build regressor head.
        
        Args:
            n_out_features: Output feature dimension
            n_layers: Number of layers in regressor
            use_sigmoid: Apply sigmoid activation
            use_bert: Whether to use BERT encoder
        """
        hidden_size = self.config.hidden_size

        if not use_bert:
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
        else:
            regressor_config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=n_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size
            )
            self.regressor = BertEncoder(regressor_config)
            self.regressor_head = nn.Sequential(
                nn.Linear(hidden_size, n_out_features)
            )
            if use_sigmoid:
                self.regressor_head += nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through classification and regression heads.
        
        Args:
            input: Input tensor of shape (batch_size, seq_len, n_in_features)
        
        Returns:
            Tuple of:
                - probs: Classification probabilities (batch_size, seq_len, 2)
                - regressed_features: Regressed features (batch_size, seq_len, n_out_features)
                - hidden_state: Last encoder hidden state (batch_size, seq_len, hidden_size)
        """
        hidden_states = self.input_embedding(input)

        attention_mask = torch.ones_like(hidden_states[:, :, 0])
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        position_embeddings = None
        if self.rotary_emb is not None:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
            position_embeddings = (cos, sin, position_ids)

        encoder_last_hidden_state, encoder_last_attention = self.bert_encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_embeddings=position_embeddings,
        )

        self.encoder_last_hidden_state = encoder_last_hidden_state.detach()
        self.encoder_last_attention = encoder_last_attention.detach() if encoder_last_attention is not None else None

        encoder_last_hidden_state_clone = encoder_last_hidden_state.clone()

        if self._use_bert_classifier:
            classifier_output = self.point_classifier(
                encoder_last_hidden_state_clone,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            probs = self.classifier_head(classifier_output.last_hidden_state)
        else:
            probs = self.point_classifier(encoder_last_hidden_state_clone)

        if self._use_bert_regressor:
            regressor_output = self.regressor(
                encoder_last_hidden_state_clone,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            regressed_features = self.regressor_head(regressor_output.last_hidden_state)
        else:
            regressed_features = self.regressor(encoder_last_hidden_state_clone)

        return probs, regressed_features, encoder_last_hidden_state_clone

    def get_attention(self, mean_over_heads: bool = False) -> Optional[torch.Tensor]:
        """Get attention weights from last encoder layer.
        
        Args:
            mean_over_heads: Average over heads if True
        
        Returns:
            Attention weights of shape:
            - (batch_size, seq_len, seq_len) if mean_over_heads=True
            - (batch_size, num_heads, seq_len, seq_len) if mean_over_heads=False
        """
        if self.encoder_last_attention is None:
            return None

        attn = self.encoder_last_attention
        if mean_over_heads:
            return attn.mean(dim=1)
        return attn

    def get_correlation(self) -> Optional[torch.Tensor]:
        """Compute correlation matrix from last hidden state.
        
        Returns:
            Correlation matrix of shape (batch_size, seq_len, seq_len)
        """
        if self.encoder_last_hidden_state is None:
            return None

        normalized_state = (
            self.encoder_last_hidden_state /
            torch.norm(self.encoder_last_hidden_state, dim=-1, keepdim=True)
        )
        corr = torch.matmul(
            normalized_state,
            normalized_state.transpose(-1, -2)
        )
        return corr


# ============================================================================
# Houyi2CCModelPlain - Base Architecture (All Qwen2 Features Disabled)
# ============================================================================

class Houyi2CCModelPlain(Houyi2CCBaseModel):
    """Houyi2CC base model with all Qwen2 features disabled.
    
    Base architecture:
    - Standard MHA (num_key_value_heads = num_attention_heads)
    - LayerNorm (not RMSNorm)
    - GELU MLP (not SwiGLU)
    - No RoPE
    - Post-LN (not Pre-LN)
    """

    def __init__(self, n_in_features: int, n_out_features: int = 17,
                 enc_config: Optional[BertConfig] = None,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_bert_classifier: bool = False,
                 use_bert_regressor: bool = False):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )

        layer_cfg = LayerConfig(
            use_gqa=False,
            use_rmsnorm=False,
            use_swiglu=False,
            use_rope=False,
            use_preln=False,
            num_key_value_heads=8,
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )


# ============================================================================
# Houyi2CCModelWithXXX Series - Individual Qwen2 Features
# ============================================================================

class Houyi2CCModelWithGQA(Houyi2CCBaseModel):
    """Houyi2CC model with only GQA enabled."""

    def __init__(self, n_in_features: int, n_out_features: int = 17,
                 enc_config: Optional[BertConfig] = None,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_bert_classifier: bool = False,
                 use_bert_regressor: bool = False,
                 num_key_value_heads: int = 2):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )
        enc_config.num_key_value_heads = num_key_value_heads

        layer_cfg = LayerConfig(
            use_gqa=True,
            use_rmsnorm=False,
            use_swiglu=False,
            use_rope=False,
            use_preln=False,
            num_key_value_heads=num_key_value_heads,
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )


class Houyi2CCModelWithRMSNorm(Houyi2CCBaseModel):
    """Houyi2CC model with only RMSNorm enabled."""

    def __init__(self, n_in_features: int, n_out_features: int = 17,
                 enc_config: Optional[BertConfig] = None,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_bert_classifier: bool = False,
                 use_bert_regressor: bool = False):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )
        enc_config.num_key_value_heads = 8  # Standard MHA

        layer_cfg = LayerConfig(
            use_gqa=False,
            use_rmsnorm=True,
            use_swiglu=False,
            use_rope=False,
            use_preln=False,
            num_key_value_heads=8,
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )


class Houyi2CCModelWithSwiGLU(Houyi2CCBaseModel):
    """Houyi2CC model with only SwiGLU enabled."""

    def __init__(self, n_in_features: int, n_out_features: int = 17,
                 enc_config: Optional[BertConfig] = None,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_bert_classifier: bool = False,
                 use_bert_regressor: bool = False):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )
        enc_config.num_key_value_heads = 8  # Standard MHA

        layer_cfg = LayerConfig(
            use_gqa=False,
            use_rmsnorm=False,
            use_swiglu=True,
            use_rope=False,
            use_preln=False,
            num_key_value_heads=8,
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )


class Houyi2CCModelWithRoPE(Houyi2CCBaseModel):
    """Houyi2CC model with only RoPE enabled."""

    def __init__(self, n_in_features: int, n_out_features: int = 17,
                 enc_config: Optional[BertConfig] = None,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_bert_classifier: bool = False,
                 use_bert_regressor: bool = False):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )
        enc_config.num_key_value_heads = 8  # Standard MHA

        layer_cfg = LayerConfig(
            use_gqa=False,
            use_rmsnorm=False,
            use_swiglu=False,
            use_rope=True,
            use_preln=False,
            num_key_value_heads=8,
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )


class Houyi2CCModelWithPreLN(Houyi2CCBaseModel):
    """Houyi2CC model with only Pre-LN enabled."""

    def __init__(self, n_in_features: int, n_out_features: int = 17,
                 enc_config: Optional[BertConfig] = None,
                 n_hidden_layers_of_classifier: int = 1,
                 n_hidden_layers_of_regressor: int = 1,
                 regressor_output_sigmoid: bool = False,
                 use_bert_classifier: bool = False,
                 use_bert_regressor: bool = False):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )
        enc_config.num_key_value_heads = 8  # Standard MHA

        layer_cfg = LayerConfig(
            use_gqa=False,
            use_rmsnorm=False,
            use_swiglu=False,
            use_rope=False,
            use_preln=True,
            num_key_value_heads=8,
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )


# ============================================================================
# Houyi2CCModel - Full Model with SwiGLU (Bidirectional Attention)
# ============================================================================

class Houyi2CCModel(Houyi2CCBaseModel):
    """Houyi2CC model with SwiGLU activation and bidirectional attention.
    
    Architecture features:
    - BERT backbone (bidirectional attention)
    - SwiGLU activation function
    - Post-LN structure
    - RMSNorm normalization
    
    Differs from Qwen2:
    - Bidirectional attention (Qwen2 uses unidirectional causal attention)
    - Post-LN structure (Qwen2 uses Pre-LN)
    
    Suitable for:
    - Sequence classification tasks
    - Regression tasks
    - Scenarios requiring bidirectional context
    
    Args:
        n_in_features: Input feature dimension
        n_out_features: Regressor output feature dimension
        enc_config: BERT configuration object
        n_hidden_layers_of_classifier: Classifier layers (default: 1)
        n_hidden_layers_of_regressor: Regressor layers (default: 1)
        regressor_output_sigmoid: Apply sigmoid to output (default: False)
        use_bert_classifier: Use BERT encoder for classifier (default: False)
        use_bert_regressor: Use BERT encoder for regressor (default: False)
    """

    def __init__(
        self,
        n_in_features: int,
        n_out_features: int = 17,
        enc_config: Optional[BertConfig] = None,
        n_hidden_layers_of_classifier: int = 1,
        n_hidden_layers_of_regressor: int = 1,
        regressor_output_sigmoid: bool = False,
        use_bert_classifier: bool = False,
        use_bert_regressor: bool = False,
    ):
        if enc_config is None:
            enc_config = BertConfig(
                hidden_size=576,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2304,
            )
        enc_config.num_key_value_heads = 2  # Use GQA

        layer_cfg = LayerConfig(
            use_gqa=False,
            use_rmsnorm=True,
            use_swiglu=True,
            use_rope=False,
            use_preln=False
        )

        super().__init__(
            n_in_features, n_out_features, enc_config,
            n_hidden_layers_of_classifier, n_hidden_layers_of_regressor,
            regressor_output_sigmoid, use_bert_classifier, use_bert_regressor,
            layer_cfg=layer_cfg,
        )
