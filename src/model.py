# -*- coding: utf-8 -*-
"""
Grokking model (Power et al. 2022, arXiv:2201.02177, Appendix A.1.2).

A small decoder-only Transformer built directly on PyTorch's built-in
`nn.TransformerEncoder` (no hand-rolled attention blocks needed).

Architecture (paper config):
  - 2 layers, width 128, 4 attention heads, causal masking
  - input sequence: <x> <op> <y> <=> <answer>  (5 tokens in total)
  - loss / accuracy are computed ONLY at the answer position

Vocabulary layout:
  - token 0 .. p-1 : residue symbols (p abstract symbols)
  - token p        : <op>
  - token p + 1    : <=>

NOTE: we deliberately keep PyTorch's default (Xavier/GPT-agnostic) weight
initialization of `nn.TransformerEncoderLayer`. Do NOT override it with
GPT-style std=0.02 init -- small init makes generalization arrive almost
simultaneously with memorization, and grokking disappears.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 5          # <x> <op> <y> <=> <answer>
ANSWER_POS = 4       # answer position in the sequence (predicted from pos 3)


class GrokTransformer(nn.Module):
    """Decoder-only Transformer assembled from PyTorch built-in modules."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # token embedding + learned positional embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # built-in Transformer encoder stack with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN, as in the paper's standard transformer
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # unembedding

        # fixed causal mask template (max_len x max_len), sliced per sequence
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) -> logits: (B, T, vocab_size)"""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(pos)[None, :, :]
        # causal self-attention mask matching the current sequence length
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tokens.device),
            diagonal=1)
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        return self.head(self.ln_f(x))

    def loss_and_acc(self, tokens: torch.Tensor):
        """
        tokens: (B, 5) = <x> <op> <y> <=> <answer>
        Cross-entropy is computed only at the answer position:
        the logits at position 3 predict the answer token at position 4.
        Returns (loss, n_correct, n_total).
        """
        logits = self.forward(tokens[:, :-1])            # (B, 4, V)
        answer_logits = logits[:, ANSWER_POS - 1, :]     # (B, V)
        answers = tokens[:, ANSWER_POS]                  # (B,)
        loss = F.cross_entropy(answer_logits, answers)
        pred = answer_logits.argmax(dim=-1)
        correct = (pred == answers).sum()
        return loss, correct, answers.numel()


def make_vocab(p: int) -> int:
    """p residue symbols + <op> + <=>  ->  p + 2 tokens."""
    return p + 2


def encode_batch(a, b, c, p: int, op_token: int = None, eq_token: int = None):
    """Encode (x, y, answer) batches into token sequences of shape (B, 5)."""
    if op_token is None:
        op_token = p
    if eq_token is None:
        eq_token = p + 1
    a = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    b = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
    c = c if isinstance(c, torch.Tensor) else torch.as_tensor(c)
    B = a.shape[0]
    op = torch.full((B,), op_token, dtype=torch.long)
    eq = torch.full((B,), eq_token, dtype=torch.long)
    return torch.stack([a, op, b, eq, c], dim=1).long()


if __name__ == "__main__":
    # minimal self-test
    torch.manual_seed(0)
    p = 97
    model = GrokTransformer(make_vocab(p))
    n_params = sum(par.numel() for par in model.parameters())
    print(f"total parameters: {n_params:,}  (paper: ~4e5)")
    toks = encode_batch([1, 2, 3], [5, 6, 7], [8, 9, 10], p)
    loss, correct, total = model.loss_and_acc(toks)
    print(f"forward OK, initial loss ~= ln(97) = {math.log(97):.3f}, "
          f"got {loss.item():.3f}")
