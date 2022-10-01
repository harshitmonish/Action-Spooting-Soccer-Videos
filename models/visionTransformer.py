import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class Attention(nn.Module):
    """
    Attention mechanism

    paramteres:
    -----------
    dim: int
        The input and output dimension per token features.

    n_heads : int
        Number of attention heads

    qkv_bias : bool
        Dropout probability applied to the query, key and value tensor

    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale: float
        Normalizing constant for the dot product.

    qkv: nn.Linear
        Linear projection for the query, key and value

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention heads
        and maps it into a new space.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads  #Define dimensionality of each head.
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias =qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        :param x:
            shape (n_samples, n_patches + 1, dim) +1 is for class token as 1st token in the sequence
        :return: Torch.Tensor
            shape(n_samples, n_patches + 1, dim)
        """
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # n_samples, n_patches + 1, * 3 * dim
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # n_samples, n_patches  + 1,  3 , n_heads, head_dim
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dot_p = (
            q @ k_t
        ) * self.scale
        attn = dot_p.softmax(dim=1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose( 1, 2) #(n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # n_samples, n_patches + 1, dim

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    """
    Multilayer perceptron

    :parameters
    -----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer

    out_features : int
        Number of output features.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        :Returns
        ---------
        torch.Tensor
            Shape (n_samples, n_patches + 1, out_features)
        """
        x = self.fc1(
            x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    """
    Transformer block

    :parameter:
    dim : int
        Embedding dimension

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the MLP module with respect to dim.

    qkv_bias : bool
        if True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability


    Attributes
    ----------
    norm1, norm2 : LayerNorm
        LayerNormalization
    attn : Attention
        attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0, attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p = attn_p,
            proj_p = p,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )
    def forward(self, x):
        """
        Run forward pass
        :param x: torch.Tensor, Shape - (n_samples, n_patches + 1, dim)
        :return:
        torch.Tensor, Shape - (n_samples, n_patches + 1, dim)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class visionTransformer(nn.Module):
    """
    Simplified implementation of the vision transformer

    :parameter
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token embeddings
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the MLP module
    qkv_bias : bool
        if True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability

    Attributes:
    patch_embed : PatchEmbed
        Instance of PatchEmbed Layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has embed_dim elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has (n_patches + 1) * embed_dim elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of Block modules.

    norm : nn.LayerNorm
        Layer Normalization
    """
    def __init__(self,
                 n_classes=18,
                 embed_dim=512,
                 n_frames=20,
                 depth=4,
                 n_heads=4,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 p=0.,
                 attn_p=0.,
                 ):
        super(visionTransformer, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+ n_frames, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim = embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Run the forward pass.
        :param self: x : torch.Tensor
        Shape (n_samples, num_frames, features_size

        :return:
        logits : torch.Tensor
            Logits over all the classes - (n_samples, n_classes)
        """
        n_samples = x.shape[0]
        #x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        ) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)

        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        cls_token_final = x[:, 0] # just CLS token
        #x = torch.nn.functional.softmax(self.head(cls_token_final), dim=1)
        x = self.head(cls_token_final)

        return x
