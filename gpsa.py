
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        locality_strength=1.0,
        gating_init=1.0,
        class_token=False,
        use_local_init=True
    ):
        super().__init__()
        #Each attention head gets an equal part of the embedding dimension 
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 #scaling used in  normal attention: QK^T / sqrt(head__dim)
        self.class_token = class_token #true when the input has  a CLS token before the image patches

        #used to calculate the normal content-based attention map (QK^T).
        #qk produces both Q and K while v produces V
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        #takes the relative position between two patches and produces one positonal attention score for each attention head. 
        self.pos_proj = nn.Linear(3, num_heads)

        #One learnable gating value for each attention head. Sigmoid() later converts this into a value between 0 and 1.
        self.gating_param = nn.Parameter(
            torch.full((num_heads,), float(gating_init))
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #we cache the relative position because the image grid normally stays the same for every batch
        self.rel_indices = None
        self.current_grid_size = None
        self.apply(self._init_weights)

        #intilialise the positional attention so that it starts out roughly looking like a local convolution.
        self._local_init(locality_strength)

    def _init_weights(self, m):
        #standard weight intilisation for the linear layers.
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _local_init(self, locality_strength):
        #start V as an identity transformation. this keeps the intial positional attention closer to input features.
        self.v.weight.data.copy_(torch.eye(self.dim))

        #we use the heads to represent different nearby sptial offsets. for example, with 4 heads this gives 2x2 arrangemnet for offsets
        kernel_size = int(math.sqrt(self.num_heads))
        center = (kernel_size - 1) / 2 if kernel_size > 1 else 0

        for h in range(self.num_heads):
            #works out which spatial offset this head represents 
            if kernel_size > 1:
                h1 = h // kernel_size
                h2 = h % kernel_size
            else:
                h1, h2 = 0, 0
        #these weights make nearby patches recieve larger positoinal attention, giving the attention a convolution like starting point. 
            self.pos_proj.weight.data[h, 0] = (
                -locality_strength * (h1 - center)
            )
            self.pos_proj.weight.data[h, 1] = (
                -locality_strength * (h2 - center)
            )
            self.pos_proj.weight.data[h, 2] = -locality_strength

        self.pos_proj.bias.data.zero_()

    def _get_rel_indices(self, H, W, device):
        #if we already calculated the relative positions for this. image size, reuse them instead of recalculating. 
        if self.rel_indices is not None and self.current_grid_size == (H, W):
            return self.rel_indices.to(device)


#create (row,column) coords for every image patch.

#for 3x3 image grid, this gives:
# [[0,0], [0,1], [0,2],
#  [1,0], [1,1], [1,2],
#  [2,0], [2,1], [2,2]]

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(H),
                torch.arange(W),
                indexing="ij"
            ),
            dim=-1
        ).reshape(H * W, 2).float()
#calc the relative position between every pair of patches. 

# if patch is at 2,2 and patch j is ar 1,2 then their relative position says theyre one row apart. 
        rel = coords[None, :, :] - coords[:, None, :]

        dy = rel[..., 0]
        dx = rel[..., 1]
        #squared dist between two patches
        dist2 = dx ** 2 + dy ** 2

#each pair of patches is represented by [x difference, y difference, squared distance]. This is the input to the positional attention projection.
        rel_indices = torch.stack([dx, dy, dist2], dim=-1)

#if theres a CLS token it doesnt have a spatial position so we add a row/column of zeros for it
        if self.class_token:
            N = H * W + 1
            padded = torch.zeros(
                N, N, 3,
                device=rel_indices.device
            )
            padded[1:, 1:] = rel_indices
            rel_indices = padded

#add a batch dimension

#the positional information is the same for every img in the batch so we only need one copy. 
        self.rel_indices = rel_indices.unsqueeze(0)
        self.current_grid_size = (H, W)

        return self.rel_indices.to(device)

    def get_attention(self, x, H, W):
        B, N, C = x.shape

        # produce the Q and K and spliy them into separate attention heads. 
        #before reshape: (B,N,2*C) 
        #after reshape: (2, B, heads, N, head_dim)
        qk = self.qk(x).reshape(
            B, N, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k = qk[0], qk[1]
        #1. normal content attention 
        #this is the usual transformer attention: QK^T it tells the model which patches have smilar feature information. 
        attn_content = (q @ k.transpose(-2, -1)) * self.scale

        #2. positional attention
        #instead of looking at the img feature, this attention is based on where the patches are located
        rel_indices = self._get_rel_indices(H, W, x.device)

    #converts relative ositions into an attention score for each head. this is the convolution-like part of GPSA.
        attn_pos = self.pos_proj(rel_indices)

        #change from (batch, N, N, heads) to: (batch, heads, N, N) so we can combine it with the content attention.
        attn_pos = attn_pos.permute(0, 3, 1, 2)

        #convert both types of scores into probabilities distributions. 
        attn_content = F.softmax(attn_content, dim=-1)
        attn_pos = F.softmax(attn_pos, dim=-1)

        #3. mix the two attentions
        #sigmoid makes the gate range from 0 to 1. 

        #gate close to 0: mostly content attention
        #gate close to 1: mostly positional attention
        gating = torch.sigmoid(self.gating_param).view(1, self.num_heads, 1, 1)

        attn = (
            (1 - gating) * attn_content
            + gating * attn_pos
        )
# Keep the attention rows normalised after mixing
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return self.attn_drop(attn)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # check that the number of tokens matches the img grid. 

        #without cls: N = H*W
        #with CLS: N = H*W+1
        expected_N = H * W + 1 if self.class_token else H * W
        assert N == expected_N

        #calculate the GPSA attention weights 
        attn = self.get_attention(x, H, W)

        #create the V values and split them into attention heads 
        v = self.v(x).reshape(
            B, N, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        #apply attention weights to V
        out = attn @ v
        #put the heads back together
        out = out.transpose(1, 2).reshape(B, N, C)
        #final linear projection like normal transformer attention. 
        out = self.proj(out)
        return self.proj_drop(out)
