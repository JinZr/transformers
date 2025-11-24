Switch RoFormer
===============

This folder keeps a RoPE-enabled variant of Switch Transformers without touching the core library.

- `SwitchRoFormerConfig` extends `SwitchTransformersConfig` with `rope_theta`, `rope_max_position_embeddings`, `rope_partial_rotary_factor`, and an opt-in `use_relative_attention_bias`.
- `SwitchRoFormerAttention` swaps T5-style relative position bias for rotary position embeddings on self-attention. Cross-attention stays unchanged by default.
- `SwitchRoFormerModel` / `SwitchRoFormerForConditionalGeneration` / `SwitchRoFormerEncoderModel` mirror the original classes but patch their stacks to use the rotary-aware attention.

Example
-------

```python
from transformers.models.switch_roformers import (
    SwitchRoFormerConfig,
    SwitchRoFormerForConditionalGeneration,
)

config = SwitchRoFormerConfig(
    d_model=768,
    num_layers=12,
    num_experts=8,
    rope_theta=10000.0,
    rope_max_position_embeddings=4096,
)
model = SwitchRoFormerForConditionalGeneration(config)
```
