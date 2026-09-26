"""Restore ROPE_INIT_FUNCTIONS['default'] removed in transformers 5.x."""
from __future__ import annotations

from pathlib import Path

p = Path("/home/shreyaspatel/.local/lib/python3.10/site-packages/transformers/modeling_rope_utils.py")
t = p.read_text()
if '"default": _compute_default_rope_parameters' in t:
    print("already patched")
else:
    helper = '''
def _compute_default_rope_parameters(
    config=None,
    device=None,
    seq_len=None,
    layer_type=None,
    **kwargs,
):
    """Plain RoPE inverse-frequency init (transformers<=4.x "default")."""
    import torch
    if config is not None and hasattr(config, "standardize_rope_params"):
        try:
            config.standardize_rope_params()
            rope_parameters_dict = (
                config.rope_parameters[layer_type]
                if layer_type is not None
                else config.rope_parameters
            )
            base = rope_parameters_dict["rope_theta"]
            partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
        except Exception:
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = 1.0
    else:
        base = getattr(config, "rope_theta", 10000.0) if config is not None else 10000.0
        partial_rotary_factor = 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0

'''
    marker = "ROPE_INIT_FUNCTIONS:"
    idx = t.find(marker)
    if idx < 0:
        raise SystemExit("ROPE_INIT_FUNCTIONS not found")
    # insert helper just before the dict
    t = t[:idx] + helper + t[idx:]
    # register in dict after opening brace
    brace = t.find("{", t.find(marker))
    t = t[: brace + 1] + '\n    "default": _compute_default_rope_parameters,' + t[brace + 1 :]
    p.write_text(t)
    print("patched", p)

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

print("keys", sorted(ROPE_INIT_FUNCTIONS))
assert "default" in ROPE_INIT_FUNCTIONS
