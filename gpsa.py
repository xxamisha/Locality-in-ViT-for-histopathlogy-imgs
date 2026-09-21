```python

"""
GPSA (Gated Positional Self-Attention) layer, based on ConViT.

Paper: "ConViT: Improving Vision Transformers with Soft Convolutional
Inductive Biases" (d'Ascoli et al., 2021) - https://arxiv.org/abs/2103.10697

normal self attention only looks at how similar are two tokens but gpsa also computes a second attention map based on relative position, and lets each head 
learn how much to trust each one via a gate. At intialisation the positional part is set up to mimic a small convolution (nearby patches get more attention), which 
is supposed to give the model a locality bias similar to a CNN while still letting it fall back to normal global attention if it's more useful.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPSA(nn.Module):
    """
    One GPSA attention layer, meant to be dropped in place of a normal
    multi-head self-attention block.

    Args:
    dim: embedding dimension
    num_heads: number of attention heads
    qkv_bias: whether to use bias in the q/k/v linear layers
    attn_drop / proj_drop: standard dropout on the attention and output
    locality_strength: how "local" the positional attention starts out.
    Bigger = more concentrated on nearby patches at init.
    use_local_init: if False, skips the convolution-like init and just
    leaves the positional projection randomly initialized (mostly useful for testing/ablations)
class_token: set True if a CLS token is prepended to the input(see forward() below)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        locality_strength=1.0,
        gating_init=1.0,
        use_local_init=True,
        class_token=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "num_heads has to divide dim evenly"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.locality_strength = locality_strength
        self.use_local_init = use_local_init

        # if we have a CLS token, it doesn't really have a "position" the
        # way patch tokens do, so we handle it separately below (basically
        # give it a relative position of 0 to everything)
        self.class_token = class_token

        # q and k are combined into one linear layer just to save a call,
        # v is separate. this is the "normal" content-based attention part
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # this turns a relative position (dx, dy, distance^2) into a score
        # per head - this is the part that lets the model learn positional
        # attention patterns
        self.pos_proj = nn.Linear(3, num_heads)

        # gate that decides, per head, how much to trust positional vs
        # content attention. passed through sigmoid so it's always in
        # [0, 1]. starting value matters - see gating_init below
        self.gating_param = nn.Parameter(torch.full((num_heads,), float(gating_init)))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # caching the relative position tensor so we don't recompute it
        # every single forward pass - it only depends on the grid size,
        # not on the actual input values
        self.rel_indices = None
        self.current_grid_size = None

        self.apply(self._init_weights)
        if self.use_local_init:
            self._local_init()

    @staticmethod
    def _init_weights(m):
        # basic weight init for the linear layers, nothing fancy
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _local_init(self):
        """
        sets up pos_proj so before any training happesn each head is already biased toward attending to a particular nearby offset - trying
        to mimic what a small convolution kernal would look like so the model can get a headstart on looking at your neighbours rather than learning 
        it from nothing. 
        
        note: the dist^2 term HAS to be negative here. If it were positive,
        attention score would go UP the further away a patch is, which is
        backwards - we want nearby patches to score higher, not lower.
        Got this wrong on a first pass (used +locality_strength instead of
        -locality_strength) and it took a while to catch since the model
        still worked but just performed worse than it should have.
        """
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1
        kernel_size = int(math.sqrt(self.num_heads))
        center = (kernel_size - 1) / 2 if kernel_size > 1 else 0

        for h in range(self.num_heads):
            if kernel_size > 1:
                h1 = h // kernel_size
                h2 = h % kernel_size
            else:
                h1, h2 = 0, 0
            # each head gets assigned a different "preferred" direction
            # based on its index, so different heads end up looking at
            # different neighbours
            self.pos_proj.weight.data[h, 0] = -1 * self.locality_strength * (h1 - center) * locality_distance
            self.pos_proj.weight.data[h, 1] = -1 * self.locality_strength * (h2 - center) * locality_distance
            self.pos_proj.weight.data[h, 2] = -1 * self.locality_strength  # negative = closer is better
        self.pos_proj.bias.data.zero_()

    def _get_rel_indices(self, H, W, device):
        """
        Builds a tensor of relative positions between every pair of
        patches in the H x W grid: for each pair (i, j) we store
        (dx, dy, dx^2 + dy^2). Cached after the first call since it's
        always the same for a given grid size.
        """
        if self.rel_indices is not None and self.current_grid_size == (H, W):
            return self.rel_indices.to(device)

        Np = H * W
        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
        ).reshape(Np, 2).float()

        rel = coords[None, :, :] - coords[:, None, :]
        dy, dx = rel[..., 0], rel[..., 1]
        dist2 = dx ** 2 + dy ** 2
        rel_indices = torch.stack([dx, dy, dist2], dim=-1)

        if self.class_token:
            # pad with an extra row/col of zeros for the CLS token - since
            # it doesn't have a real spatial position, this basically
            # treats it as equally "close" to every patch and vice versa
            N = Np + 1
            padded = torch.zeros(N, N, 3)
            padded[1:, 1:] = rel_indices
            rel_indices = padded

        rel_indices = rel_indices.unsqueeze(0)
        self.rel_indices = rel_indices
        self.current_grid_size = (H, W)
        return rel_indices.to(device)

    def get_attention(self, x, H, W):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]

        # standard scaled dot-product attention
        attn_content = (q @ k.transpose(-2, -1)) * self.scale

        # positional attention - same for every image in the batch since
        # it only depends on the grid layout, not the actual pixel values
        rel_indices = self._get_rel_indices(H, W, x.device)
        attn_pos = self.pos_proj(rel_indices)
        attn_pos = attn_pos.permute(0, 3, 1, 2)

        attn_content = F.softmax(attn_content, dim=-1)
        attn_pos = F.softmax(attn_pos, dim=-1)

        # blend the two using the per-head gate
        gating = torch.sigmoid(self.gating_param).view(1, -1, 1, 1)
        attn = (1.0 - gating) * attn_content + gating * attn_pos
        # since we're mixing two things that already sum to 1 each, the
        # result should already sum to ~1, but renormalizing anyway just
        # to be safe with floating point stuff
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.attn_drop(attn)
        return attn

    def forward(self, x, H, W):
        """
        x: (B, N, C) - the tokens
        H, W: the patch grid dimensions (so N = H*W, or H*W + 1 if there's
              a CLS token at the front)
        """
        B, N, C = x.shape
        expected_N = H * W + 1 if self.class_token else H * W
        assert N == expected_N, f"got N={N}, expected {expected_N} (class_token={self.class_token})"

        attn = self.get_attention(x, H, W)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def get_attention_map(self, x, H, W, return_map=True):
        """Mostly just for visualizing/debugging - lets you see what the
        attention actually looks like (averaged across heads)."""
        attn = self.get_attention(x, H, W)
        if return_map:
            return attn.mean(dim=1)
        return attn


if __name__ == "__main__":
    # quick sanity check that everything runs and shapes line up
    B, H, W, C, heads = 2, 14, 14, 192, 4
    x = torch.randn(B, H * W, C)
    gpsa = GPSA(dim=C, num_heads=heads, locality_strength=1.0)
    out = gpsa(x, H, W)
    print("output shape:", out.shape)
    assert out.shape == x.shape
    print("gating (sigmoid):", torch.sigmoid(gpsa.gating_param))
