```python
"""
Swaps GPSA in for standard attention in a pretrained phikon backbone.

phikon is just a normal ViT loaded through HuggingFace transformers.
This file replaces the first `local_layers` attention blocks with GPSA
instead, but the important part is that it COPIES the original
pretrained Q/K/V/output weights into the new GPSA layers first - so
we're not throwing away everything phikon learned, only changing how
the attention itself is computed (adding the positional part on top of
what's already there).

Had to check the actual HuggingFace ViT source to get this right, since
each layer's attention module looks like:
    ViTAttention(q_proj, k_proj, v_proj, o_proj)
and ViTLayer.forward() calls it like:
    hidden_states, _ = self.attention(hidden_states, attention_mask, **kwargs)
so the replacement has to match that same call or the rest of
the model breaks.

How to use it:
    from transformers import ViTModel
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    inject_gpsa(model, local_layers=10, locality_strength=1.0, gating_init=1.0)
    # first 10 layers now use GPSA (with pretrained weights kept),
    # the rest are untouched normal attention
"""

import math
import torch
import torch.nn as nn
from gpsa import GPSA


class GPSAAttentionWrapper(nn.Module):
    """
    Wraps a GPSA layer so it can be dropped straight into
    layer.attention without needing to touch anything else in the ViT
    code - just has to match HF's expected input/output shape.
    """

    def __init__(self, gpsa_module):
        super().__init__()
        self.gpsa = gpsa_module

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, C = hidden_states.shape
        # -1 because of the CLS token at the front
        num_patches = N - 1
        H = W = int(math.isqrt(num_patches))
        assert H * W == num_patches, (
            f"expected a square patch grid but got {num_patches} patches "
            f"(N={N}) - if the input isn't square this needs H, W passed in manually"
        )
        out = self.gpsa(hidden_states, H, W)
        return out, None  # HF expects (output, attention_weights), we don't use the second one


def _transplant_qkv(gpsa, old_attn):
    """Copies the pretrained q/k/v/output weights over into GPSA's layout."""
    dim = old_attn.q_proj.in_features
    with torch.no_grad():
        # GPSA combines q and k into one linear layer, so split it into
        # the first half / second half
        gpsa.qk.weight[:dim].copy_(old_attn.q_proj.weight)
        gpsa.qk.weight[dim:].copy_(old_attn.k_proj.weight)
        if old_attn.q_proj.bias is not None:
            gpsa.qk.bias[:dim].copy_(old_attn.q_proj.bias)
            gpsa.qk.bias[dim:].copy_(old_attn.k_proj.bias)

        gpsa.v.weight.copy_(old_attn.v_proj.weight)
        if old_attn.v_proj.bias is not None:
            gpsa.v.bias.copy_(old_attn.v_proj.bias)

        gpsa.proj.weight.copy_(old_attn.o_proj.weight)
        if old_attn.o_proj.bias is not None:
            gpsa.proj.bias.copy_(old_attn.o_proj.bias)


def inject_gpsa(model, local_layers=10, locality_strength=1.0, gating_init=1.0, use_local_init=True):
    """
    Changes model in place - swaps the first local_layers attention
    blocks over to GPSA, keeping everything after that as normal
    attention.

    Args:
        model: a ViTModel already loaded from pretrained weights
        local_layers: how many of the first layers get converted to GPSA
        locality_strength: how local the positional attention starts out
        gating_init: starting value for the gate (before sigmoid). 1.0
            matches the paper (~73% positional at init). Lower values
            (e.g. 0.0) start closer to 50/50, which might make sense if
            we're only fine-tuning for a short time and don't want to
            disrupt the pretrained attention too much right away.
        use_local_init: whether to actually do the convolution-like init
            on the positional part, or just leave it randomly initialized

    Returns model (already modified in place, just returned for convenience).
    """
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size

    for i in range(local_layers):
        layer = model.layers[i]
        old_attn = layer.attention

        gpsa = GPSA(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            locality_strength=locality_strength,
            gating_init=gating_init,
            use_local_init=use_local_init,
            class_token=True,
        )
        _transplant_qkv(gpsa, old_attn)
        layer.attention = GPSAAttentionWrapper(gpsa)

    return model


def get_gating_params(model, local_layers=10):
    """Returns sigmoid(gate) per head for every converted layer - useful
    for checking whether the gates actually moved away from their
    starting value during training (i.e. did the model learn to prefer
    content over positional attention, or the other way round)."""
    out = {}
    for i in range(local_layers):
        attn = model.layers[i].attention
        if isinstance(attn, GPSAAttentionWrapper):
            out[i] = torch.sigmoid(attn.gpsa.gating_param).detach().cpu()
    return out


if __name__ == "__main__":
    # quick check that this actually runs end to end - using a randomly
    # initialized model with the same config as phikon since we don't
    # need real pretrained weights just to check the plumbing works
    from transformers import ViTModel, ViTConfig

    cfg = ViTConfig(hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
                     image_size=224, patch_size=16)
    model = ViTModel(cfg, add_pooling_layer=False)

    inject_gpsa(model, local_layers=10, locality_strength=1.0)

    x = torch.randn(2, 3, 224, 224)
    out = model(x).last_hidden_state
    print("output shape:", out.shape)
    assert out.shape == (2, 197, 768)

    loss = out.mean()
    loss.backward()
    gates = get_gating_params(model, local_layers=10)
    print("GPSA layers converted:", list(gates.keys()))
    print("layer 0 gate (per head):", gates[0])
    print("layer 0 gate grad is not None:",
          model.layers[0].attention.gpsa.gating_param.grad is not None)
```
