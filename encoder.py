from transformers import AutoConfig, AutoTokenizer

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def scaled_dot_product_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Attention similarity measure, from "Attention Is All You Need" original paper.
    """

    # Norm factor
    embed_dim = k.shape[-1]

    # (Q dot K) / sqrt(n)
    scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(embed_dim)

    # softmax(weights)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)

class Attention(nn.Module):
    """
    A single attention which uses scaled dot product to calculate an updated set of embeddings for inputs to a Transformer encoder.
    """
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.Q = nn.Linear(embed_dim, head_dim)
        self.K = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attn(
            self.Q(x),
            self.K(x),
            self.V(x),
        )

class MultiHeadAttention(nn.Module):
    """
    Multihead attention from Transformer. Concatentes outputs from multiple attention heads into a single item.
    """
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        head_dim = embed_dim // n_heads
        self.heads = nn.ModuleList([
            Attention(embed_dim=embed_dim, head_dim=head_dim) for _ in range(n_heads)
        ])
        self.linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # Concatenate all of the values together.
        result = torch.cat([h(xs) for h in self.heads], dim=-1)
        result = self.linear(result)
        return result
    
class FeedForward(nn.Module):
    """
    Feed forward network that stacks two linear layers with a nonlinearity in the middle. Allows learning
    extra information in the encoder about how the attention-updated outputs map to hidden representation.
    """
    
    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x
    
class EncoderBlock(nn.Module):
    """
    Single block in the Transformer encoder
    """
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.ff = FeedForward(embed_dim=embed_dim, ff_dim=embed_dim*4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ff(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers: int, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(embed_dim=embed_dim, n_heads=n_heads) for _ in range(n_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    # Create a new field, and download it all
    inps = torch.rand((1, 15, 768))
    # mha = MultiHeadAttention(embed_dim=768, n_heads=12)
    # res = mha(inps)
    # print(res.shape)
    encoder = Encoder(embed_dim=768, n_heads=12, n_layers=12)
    res = encoder(inps)
    print(res.shape)
