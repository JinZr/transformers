# coding=utf-8
from __future__ import annotations

from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig


class SwitchRoFormerConfig(SwitchTransformersConfig):
    """
    Drop-in configuration for a Switch Transformer variant that swaps T5-style relative position bias for rotary
    position embeddings (RoPE).

    Args:
        rope_theta (`float`, *optional*, defaults to `10000.0`):
            Base used to compute rotary frequencies.
        rope_scaling (`dict`, *optional*):
            Placeholder for custom RoPE scaling strategies. Kept for parity with other rotary-enabled configs.
        rope_max_position_embeddings (`int`, *optional*, defaults to `4096`):
            Length up to which rotary caches are preallocated. Automatically grows if longer positions are seen.
        rope_partial_rotary_factor (`float`, *optional*, defaults to `1.0`):
            Portion of each head dimension rotated by RoPE. Must be in the interval `(0, 1]`.
        use_relative_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to retain SwitchTransformers' relative position bias alongside RoPE.
        **kwargs:
            All base [`SwitchTransformersConfig`] arguments.
    """

    model_type = "switch_roformer"

    def __init__(
        self,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        rope_max_position_embeddings: int = 4096,
        rope_partial_rotary_factor: float = 1.0,
        use_relative_attention_bias: bool = False,
        **kwargs,
    ):
        if rope_partial_rotary_factor <= 0.0 or rope_partial_rotary_factor > 1.0:
            raise ValueError("`rope_partial_rotary_factor` must be in the interval (0, 1].")

        super().__init__(**kwargs)
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_max_position_embeddings = rope_max_position_embeddings
        self.rope_partial_rotary_factor = rope_partial_rotary_factor
        self.use_relative_attention_bias = use_relative_attention_bias


__all__ = ["SwitchRoFormerConfig"]
