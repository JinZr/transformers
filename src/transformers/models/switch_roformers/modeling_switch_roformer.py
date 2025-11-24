# coding=utf-8
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from transformers.cache_utils import EncoderDecoderCache
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersAttention,
    SwitchTransformersEncoderModel,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersModel,
)
from transformers.utils import logging

from .configuration_switch_roformer import SwitchRoFormerConfig

logger = logging.get_logger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwitchRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_position_embeddings: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_seq_len_cached = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        seq_len_to_cache = max(seq_len, self.max_seq_len_cached)
        self.max_seq_len_cached = seq_len_to_cache
        t = torch.arange(seq_len_to_cache, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(dtype=dtype)
        self._sin_cached = emb.sin().to(dtype=dtype)

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(position_ids.max()) + 1 if position_ids.numel() > 0 else 1
        if (
            self._cos_cached is None
            or seq_len > self.max_seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._set_cos_sin_cache(seq_len, device, dtype)

        position_ids = position_ids.to(device)
        flat_position_ids = position_ids.reshape(-1)
        cos = self._cos_cached.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
        sin = self._sin_cached.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
        return cos, sin


class SwitchRoFormerAttention(SwitchTransformersAttention):
    def __init__(
        self,
        config: SwitchRoFormerConfig,
        has_relative_attention_bias: bool = False,
        layer_idx: Optional[int] = None,
    ):
        use_rel_bias = has_relative_attention_bias and getattr(config, "use_relative_attention_bias", False)
        super().__init__(config, has_relative_attention_bias=use_rel_bias, layer_idx=layer_idx)

        self.rotary_dim = int(self.key_value_proj_dim * getattr(config, "rope_partial_rotary_factor", 1.0))
        if self.rotary_dim <= 0 or self.rotary_dim % 2 != 0:
            raise ValueError("`rope_partial_rotary_factor` must leave an even, positive rotary dimension.")

        self.apply_rotary_to_cross_attention = getattr(config, "rope_on_cross_attention", False)
        self.rotary_emb = SwitchRotaryEmbedding(
            dim=self.rotary_dim,
            base=getattr(config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(config, "rope_max_position_embeddings", 4096),
        )

    def _apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cache_position: Optional[torch.Tensor],
        is_cross_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if is_cross_attention and not self.apply_rotary_to_cross_attention:
            return query_states, key_states

        if cache_position is None:
            position_ids = torch.arange(
                key_states.shape[-2], device=key_states.device, dtype=torch.long
            ).unsqueeze(0)
        else:
            position_ids = cache_position
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        cos, sin = self.rotary_emb(position_ids, device=query_states.device, dtype=query_states.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot, q_pass = query_states[..., : self.rotary_dim], query_states[..., self.rotary_dim :]
        k_rot, k_pass = key_states[..., : self.rotary_dim], key_states[..., self.rotary_dim :]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        query_states = torch.cat([q_rot, q_pass], dim=-1)
        key_states = torch.cat([k_rot, k_pass], dim=-1)
        return query_states, key_states

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                curr_past_key_values = past_key_values.cross_attention_cache
            else:
                curr_past_key_values = past_key_values.self_attention_cache
        else:
            curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if not is_cross_attention:
                query_states, key_states = self._apply_rotary(query_states, key_states, cache_position, False)

            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        if is_cross_attention and self.apply_rotary_to_cross_attention:
            query_states, key_states = self._apply_rotary(query_states, key_states, cache_position, True)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


def _upgrade_stack_to_rope(stack) -> None:
    """
    Swap the SwitchTransformers self-attention modules on a stack for rotary-aware variants.
    """
    use_rel_bias = getattr(stack.config, "use_relative_attention_bias", False)
    for block in stack.block:
        old_self_attention = block.layer[0].SelfAttention
        new_self_attention = SwitchRoFormerAttention(
            stack.config,
            has_relative_attention_bias=old_self_attention.has_relative_attention_bias and use_rel_bias,
            layer_idx=old_self_attention.layer_idx,
        )
        attention_state_dict = old_self_attention.state_dict()
        if not new_self_attention.has_relative_attention_bias:
            attention_state_dict.pop("relative_attention_bias.weight", None)
        _, unexpected_keys = new_self_attention.load_state_dict(attention_state_dict, strict=False)
        if unexpected_keys:
            logger.warning(
                "Unexpected keys when porting SwitchTransformers attention to RoPE: %s", ", ".join(unexpected_keys)
            )
        block.layer[0].SelfAttention = new_self_attention


class SwitchRoFormerModel(SwitchTransformersModel):
    config_class = SwitchRoFormerConfig

    def __init__(self, config: SwitchRoFormerConfig):
        super().__init__(config)
        _upgrade_stack_to_rope(self.encoder)
        _upgrade_stack_to_rope(self.decoder)


class SwitchRoFormerForConditionalGeneration(SwitchTransformersForConditionalGeneration):
    config_class = SwitchRoFormerConfig

    def __init__(self, config: SwitchRoFormerConfig):
        super().__init__(config)
        _upgrade_stack_to_rope(self.encoder)
        _upgrade_stack_to_rope(self.decoder)


class SwitchRoFormerEncoderModel(SwitchTransformersEncoderModel):
    config_class = SwitchRoFormerConfig

    def __init__(self, config: SwitchRoFormerConfig):
        super().__init__(config)
        _upgrade_stack_to_rope(self.encoder)


__all__ = [
    "SwitchRoFormerAttention",
    "SwitchRoFormerConfig",
    "SwitchRoFormerEncoderModel",
    "SwitchRoFormerForConditionalGeneration",
    "SwitchRoFormerModel",
]
