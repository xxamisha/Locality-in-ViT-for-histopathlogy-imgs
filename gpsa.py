```python
"""
Gated Positional Self-Attention (GPSA)

GPSA combines normal self-attention with positional attention.
The positional part encourages the model to look at nearby patches,
similar to the locality that CNNs have.

Based on:
d'Ascoli et al. (2021)
"ConViT: Improving Vision Transformers with Soft Convolutional
Inductive Biases"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPSA(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=4,
        locality_strength=1.0,
        dropout=0.0
    ):
        super().__init__()

        # Make sure the embedding can be split between the heads
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Scaling used in normal attention
        self.scale = self.head_dim ** -0.5

        # Used to create Q, K and V
        self.qkv = nn.Linear(dim, dim * 3)

        # Used to turn the relative position of two patches
        # into an attention score for each head
        self.position = nn.Linear(3, num_heads)

        # Each head learns how much positional attention to use.
        # sigmoid keeps this value between 0 and 1.
        self.gate = nn.Parameter(torch.zeros(num_heads))

        self.output = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.locality_strength = locality_strength

        self._setup_position_weights()

    def _setup_position_weights(self):
        """
        Give the positional attention a simple local starting point.
        """

        # Start the position layer with small values
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)

        # Make the third input (distance) negative so that
        # nearby patches get higher scores.
        self.position.weight.data[:, 2] = -self.locality_strength

    def get_positions(self, H, W, device):
        """
        Create the relative position information for every pair
        of image patches.

        Each pair has:
        dx = difference in x position
        dy = difference in y position
        distance = squared distance between them
        """

        # Create coordinates for every patch
        rows, cols = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )

        coordinates = torch.stack(
            [rows, cols],
            dim=-1
        ).reshape(-1, 2).float()

        # Difference between every pair of patches
        difference = (
            coordinates[:, None, :]
            - coordinates[None, :, :]
        )

        dy = difference[..., 0]
        dx = difference[..., 1]

        distance = dx ** 2 + dy ** 2

        positions = torch.stack(
            [dx, dy, distance],
            dim=-1
        )

        return positions

    def get_attention(self, x, H, W):

        B, N, C = x.shape

        # Create Q, K and V
        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        # -----------------------------
        # Normal self-attention
        # -----------------------------

        content_attention = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        content_attention = F.softmax(
            content_attention,
            dim=-1
        )

        # -----------------------------
        # Positional attention
        # -----------------------------

        positions = self.get_positions(
            H,
            W,
            x.device
        )

        # Shape:
        # (N, N, num_heads)
        position_attention = self.position(
            positions
        )

        # Change to:
        # (num_heads, N, N)
        position_attention = position_attention.permute(
            2, 0, 1
        )

        # Add batch dimension
        position_attention = position_attention.unsqueeze(0)

        position_attention = F.softmax(
            position_attention,
            dim=-1
        )

        # -----------------------------
        # Combine the two attentions
        # -----------------------------

        gate = torch.sigmoid(self.gate)

        # Shape so it can be applied to every image
        gate = gate.view(
            1,
            self.num_heads,
            1,
            1
        )

        attention = (
            (1 - gate) * content_attention
            + gate * position_attention
        )

        attention = self.dropout(attention)

        return attention, v

    def forward(self, x, H, W):

        B, N, C = x.shape

        # Number of patches should match the image grid
        assert N == H * W

        attention, v = self.get_attention(
            x,
            H,
            W
        )

        # Apply attention to the values
        output = attention @ v

        # Put all of the heads back together
        output = output.transpose(1, 2)

        output = output.reshape(
            B,
            N,
            C
        )

        output = self.output(output)

        return output

    def get_attention_map(self, x, H, W):
        """
        Returns the average attention across all heads.
        This is mainly useful for visualising the attention.
        """

        attention, _ = self.get_attention(
            x,
            H,
            W
        )

        return attention.mean(dim=1)


if __name__ == "__main__":

    # Small test to check that GPSA works

    batch_size = 2
    H = 14
    W = 14
    embedding_dim = 192
    heads = 4

    x = torch.randn(
        batch_size,
        H * W,
        embedding_dim
    )

    gpsa = GPSA(
        dim=embedding_dim,
        num_heads=heads
    )

    output = gpsa(
        x,
        H,
        W
    )

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Input and output should have the same shape
    assert output.shape == x.shape

    print(
        "Gate values:",
        torch.sigmoid(gpsa.gate)
    )
```
