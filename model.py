import torch
import torch.nn as nn
import numpy as np
import math
import einops
import einsum
from dataclasses import dataclass

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
text = "abcde"
encoded_input = tokenizer(text, return_tensors="pt")
tokens = encoded_input["input_ids"]  # Extract token ids


@dataclass
class Config:
    debug: bool = False
    d_embd: int = 768
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 50257
    initializer_range: float = 0.02
    n_ctx: int = 1024
    d_head = 64
    d_inner = (
        3072  # typically set to 4 times the size of n_embd in the GPT-2 architecture
    )
    n_heads = 12
    n_layer = 12


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
    cfg = Config(debug=False)
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
        self.wte = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_embd))
        nn.init.normal_(self.wte, std=self.cfg.initializer_range)

    def forward(self, tokens):
        # tokens: [batch_size, position]
        if self.cfg.debug:
            print("Input:", tokens.shape)
        embed = self.wte[tokens, :]
        if self.cfg.debug:
            print("Output:", embed.shape)
        return embed


# batch size of 2, meaning 2 sentences
# position/ seq_lenq of 4, meaning 4 words in each sentence
# you will get out a tensor of shape [2, 4, 768], because each word is represented by a 768-dimensional vector
# rand_int_test(Embed, [2, 4])


# acts in parallel to the embedding layer; they are not dependent on each other
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wpe = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_embd)))
        nn.init.normal_(self.wpe, std=self.cfg.initializer_range)

    def forward(self, tokens):
        # tokens: [batch_size, position]
        if self.cfg.debug:
            print("Input:", tokens.shape)
        # Select the positional embeddings up to the max sequence length in 'tokens'
        pos_embed = self.wpe[: tokens.size(1), :]
        # Expand position embeddings to match the batch size in the first dimension
        pos_embed = pos_embed.unsqueeze(0).repeat(
            tokens.size(0), 1, 1
        )  # Shape: [batch_size, position, d_embd]
        if self.cfg.debug:
            print("Output:", pos_embed.shape)
        return pos_embed


# rand_int_test(PosEmbed, [2, 4])


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.ones(cfg.d_embd))
        self.bias = nn.Parameter(torch.zeros(cfg.d_embd))

    def forward(self, x):
        # x: [batch_size, position, d_embd]
        if self.cfg.debug:
            print("Input:", x.shape)

        # Calculate mean and variance along the last dimension (d_embd)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + cfg.layer_norm_epsilon)

        # Apply learnable parameters
        y = x_norm * self.weight + self.bias

        if self.cfg.debug:
            print("Normalized:", y.shape)

        # return y
        return torch.nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, 1e-5
        )


# rand_float_test(LayerNorm, [2, 4, 768])


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # key, query, value projections for all heads, but in a single linear layer
        self.c_attn = nn.Linear(cfg.d_embd, 3 * cfg.d_embd)
        # output projection
        self.c_proj = nn.Linear(cfg.d_embd, cfg.d_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).view(
                1, 1, cfg.n_ctx, cfg.n_ctx
            ),
        )

    def forward(self, x):
        if self.cfg.debug:
            print("Normalized_resid_pre:", x.shape)

        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.cfg.d_embd, dim=2)

        k = k.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


# rand_float_test(SelfAttention, [2, 4, 768])


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_fc = nn.Linear(cfg.d_embd, cfg.d_inner)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(cfg.d_inner, cfg.d_embd)

    def forward(self, x):
        if self.cfg.debug:
            print("Input:", x.shape)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        if self.cfg.debug:
            print("Output:", x.shape)
        return x


# rand_float_test(MLP, [2, 4, 768])


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_1 = LayerNorm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        if self.cfg.debug:
            print("Input:", x.shape)
        x_norm = self.ln_1(x)
        z = self.attn(x_norm)
        z = z + x  # Residual connection
        z_norm = self.ln_2(z)
        z_proj = self.mlp(z_norm)
        z_proj = z_proj + x  # Residual connection
        if self.cfg.debug:
            print("Output:", x.shape)
        return z_proj


# rand_float_test(Block, [2, 4, 768])


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.d_embd),
                wpe=nn.Embedding(cfg.n_ctx, cfg.d_embd),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg),
            )
        )
        self.lm_head = nn.Linear(cfg.d_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        b, t = idx.size()
        assert (
            t <= self.cfg.n_ctx
        ), f"Cannot forward sequence of length {t}, block size is only {self.cfg.n_ctx}"
        pos = torch.arange(0, t, dtype=torch.long)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        # inference-time mini-optimization: only forward the lm_head on the very last position
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# rand_int_test(Transformer, [2, 4])


def from_pretrained(target_model, pretrained_model):
    sd = target_model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [
        k for k in sd_keys if not k.endswith(".attn.bias")
    ]  # discard this mask / buffer, not a param

    model_hf = pretrained_model
    sd_hf = model_hf.state_dict()
    # copy while ensuring all of the parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [
        k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
    ]  # ignore these, just a buffer
    sd_keys_hf = [
        k for k in sd_keys_hf if not k.endswith(".attn.bias")
    ]  # same, just the mask (buffer)
    should_transpose = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    #  print("target model keys")
    #  for key in sd_keys:
    #      print(key)
    #  print()
    #  print("pretrained model keys")
    #  for key in sd_keys_hf:
    #      print(key)
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(
        sd_keys
    ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in should_transpose):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert (
                sd_hf[k].shape == sd[k].shape
            ), f"mismatched dims: {sd_hf[k].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])


model = Transformer(cfg)
from_pretrained(model, gpt2)
# for key in model.state_dict().keys():
# print(key)

output = model(tokens)
# output = gpt2(tokens)
output = torch.softmax(output, dim=-1)
predicted_token_ids = torch.argmax(
    output, dim=-1
)  # Get the most likely next token ID for each position
print("Predicted tokens:", predicted_token_ids)
output_text = tokenizer.decode(predicted_token_ids[-1][-1])
print(output_text)

for i in range(10):
    encoded_input = tokenizer(text, return_tensors="pt")
    tokens = encoded_input["input_ids"]  # Extract token ids
    output = model(tokens)
    output = torch.softmax(output, dim=-1)
    predicted_token_ids = torch.argmax(
        output, dim=-1
    )  # Get the most likely next token ID for each position
    output_token = tokenizer.decode(predicted_token_ids[-1][-1])
    text += output_token
    print(text)
