```python
"""
Adding GPSA to a pretrained Phikon ViT.

The pretrained attention weights are copied into the GPSA layer so
that we do not have to train the whole model from scratch.

The first few ViT layers use GPSA and the remaining layers keep
the original self-attention.
"""

import math
import torch
import torch.nn as nn

from gpsa import GPSA


class GPSAAttention(nn.Module):
    """
    Wrapper around GPSA so that it can be used in the HuggingFace ViT.
    """

    def __init__(self, gpsa):
        super().__init__()

        self.gpsa = gpsa

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        **kwargs
    ):

        B, N, C = hidden_states.shape

        # Phikon has one class token, so the rest are image patches
        num_patches = N - 1

        # The images use a square patch grid
        H = int(math.sqrt(num_patches))
        W = H

        assert H * W == num_patches

        output = self.gpsa(
            hidden_states,
            H,
            W
        )

        # HuggingFace expects two outputs
        # We don't need the attention weights here
        return output, None


def copy_pretrained_weights(gpsa, old_attention):
    """
    Copy the pretrained ViT attention weights into GPSA.
    """

    dim = old_attention.q_proj.in_features

    with torch.no_grad():

        # Q and K are stored together in GPSA
        gpsa.qk.weight[:dim].copy_(
            old_attention.q_proj.weight
        )

        gpsa.qk.weight[dim:].copy_(
            old_attention.k_proj.weight
        )

        # Copy Q and K biases if they exist
        if old_attention.q_proj.bias is not None:

            gpsa.qk.bias[:dim].copy_(
                old_attention.q_proj.bias
            )

            gpsa.qk.bias[dim:].copy_(
                old_attention.k_proj.bias
            )

        # Copy V
        gpsa.v.weight.copy_(
            old_attention.v_proj.weight
        )

        if old_attention.v_proj.bias is not None:
            gpsa.v.bias.copy_(
                old_attention.v_proj.bias
            )

        # Copy the final attention projection
        gpsa.output.weight.copy_(
            old_attention.o_proj.weight
        )

        if old_attention.o_proj.bias is not None:
            gpsa.output.bias.copy_(
                old_attention.o_proj.bias
            )


def add_gpsa(
    model,
    local_layers=10,
    locality_strength=1.0
):
    """
    Replace the first few attention layers with GPSA.
    """

    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size

    for i in range(local_layers):

        # Get the original attention layer
        layer = model.encoder.layer[i]
        old_attention = layer.attention.attention

        # Create a GPSA layer with the same dimensions
        gpsa = GPSA(
            dim=dim,
            num_heads=num_heads,
            locality_strength=locality_strength
        )

        # Keep the pretrained Q/K/V/output weights
        copy_pretrained_weights(
            gpsa,
            old_attention
        )

        # Replace the original attention
        layer.attention.attention = GPSAAttention(gpsa)

    return model


def get_gates(model, local_layers=10):
    """
    Get the current GPSA gate values.

    A higher value means the head is using more positional attention.
    """

    gates = {}

    for i in range(local_layers):

        attention = model.encoder.layer[
            i
        ].attention.attention

        if isinstance(attention, GPSAAttention):

            gates[i] = torch.sigmoid(
                attention.gpsa.gate
            ).detach().cpu()

    return gates


if __name__ == "__main__":

    from transformers import ViTModel, ViTConfig

    # Small ViT with the same basic structure as Phikon
    config = ViTConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        image_size=224,
        patch_size=16
    )

    model = ViTModel(
        config,
        add_pooling_layer=False
    )

    # Replace the first 10 attention layers
    model = add_gpsa(
        model,
        local_layers=10,
        locality_strength=1.0
    )

    # Test image
    x = torch.randn(
        2,
        3,
        224,
        224
    )

    output = model(x).last_hidden_state

    print("Output shape:", output.shape)

    # 224 / 16 = 14 patches per side
    # 14 * 14 = 196 patches + 1 class token
    assert output.shape == (2, 197, 768)

    # Check that gradients can pass through GPSA
    loss = output.mean()
    loss.backward()

    gates = get_gates(
        model,
        local_layers=10
    )

    print(
        "GPSA layers:",
        list(gates.keys())
    )

    print(
        "Layer 0 gates:",
        gates[0]
    )

    print(
        "Gate has gradient:",
        model.encoder.layer[0]
        .attention.attention
        .gpsa.gate.grad is not None
    )
```
