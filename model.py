import torch
import torch.nn as nn
import numpy as np
import math
import einops
import einsum
from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head = 64
    d_mlp = 3072
    n_heads = 12


cfg = Config()


# used for testing layer modules
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg)
    random_input = torch.randn(shape)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg)
    random_input = torch.randint(100, 1000, shape)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch_size, position]
        if self.cfg.debug:
            print("Input:", tokens.shape)
        embed = self.W_E[tokens, :]
        if self.cfg.debug:
            print("Output:", embed.shape)
        return embed


# batch size of 2, meaning 2 sentences
# position/ seq_lenq of 4, meaning 4 words in each sentence
# you will get out a tensor of shape [2, 4, 768], because each word is represented by a 768-dimensional vector
rand_int_test(Embed, [2, 4])


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x):
        # x: [batch_size, position, d_model]
        if self.cfg.debug:
            print("Input:", x.shape)

        # Calculate mean and variance along the last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + cfg.layer_norm_eps)

        # Apply learnable parameters
        y = x_norm * self.w + self.b

        if self.cfg.debug:
            print("Normalized:", y.shape)

        return y


rand_float_test(LayerNorm, [2, 4, 768])


# acts in parallel to the embedding layer; they are not dependent on each other
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch_size, position]
        if self.cfg.debug:
            print("Input:", tokens.shape)
        # Select the positional embeddings up to the max sequence length in 'tokens'
        pos_embed = self.W_pos[: tokens.size(1), :]
        # Expand position embeddings to match the batch size in the first dimension
        pos_embed = pos_embed.unsqueeze(0).repeat(
            tokens.size(0), 1, 1
        )  # Shape: [batch_size, position, d_model]
        if self.cfg.debug:
            print("Output:", pos_embed.shape)
        return pos_embed


rand_int_test(PosEmbed, [2, 4])


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # key, query, value projections for all heads, but in a batch
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head)
        self.W_K = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head)
        self.W_V = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head)
        self.W_O = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).view(
                1, 1, cfg.n_ctx, cfg.n_ctx
            ),
        )

    def forward(self, normalized_resid_pre):
        if self.cfg.debug:
            print("Normalized_resid_pre:", normalized_resid_pre.shape)

        batch_size, seq_len, d_model = normalized_resid_pre.shape
        n_heads, d_head = self.cfg.n_heads, self.cfg.d_head

        # Linear projections for Q, K, V
        Q = self.W_Q(
            normalized_resid_pre
        )  # Shape: [batch_size, seq_len, n_heads * d_head]
        K = self.W_K(
            normalized_resid_pre
        )  # Shape: [batch_size, seq_len, n_heads * d_head]
        V = self.W_V(
            normalized_resid_pre
        )  # Shape: [batch_size, seq_len, n_heads * d_head]

        # Reshape and permute to [batch_size, n_heads, seq_len, d_head]
        Q = Q.view(batch_size, seq_len, n_heads, d_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, n_heads, d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, n_heads, d_head).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d_head**0.5)
        attn = attn.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # Weighted sum of values
        z = torch.matmul(attn, V)  # Shape: [batch_size, n_heads, seq_len, d_head]

        # Concatenate heads and apply final linear projection
        z = (
            z.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, n_heads * d_head)
        )
        output = self.W_O(z)

        if self.cfg.debug:
            print("Output shape:", output.shape)

        return output


rand_float_test(SelfAttention, [2, 4, 768])


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.gelu = nn.GELU()
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x):
        if self.cfg.debug:
            print("Input:", x.shape)
        x = self.W_in(x)
        x = self.gelu(x)
        x = self.W_out(x)
        if self.cfg.debug:
            print("Output:", x.shape)
        return x


rand_float_test(MLP, [2, 4, 768])


class Block(mm.Moduele):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm1 = LayerNorm(cfg)
        self.attn = SelfAttention(cfg)
        self.norm2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        if self.cfg.debug:
            print("Input:", x.shape)
        x = self.norm1(x)
        x = self.attn(x)
        x = x + x  # Residual connection
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x  # Residual connection
        if self.cfg.debug:
            print("Output:", x.shape)
        return x
