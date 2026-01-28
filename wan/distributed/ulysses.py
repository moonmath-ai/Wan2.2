# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import flash_attention
from .util import all_to_all

# Import lite_attention for optimized attention
try:
    from lite_attention import LiteAttention
    LITE_ATTENTION_AVAILABLE = True
except ImportError:
    LITE_ATTENTION_AVAILABLE = False


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        lite_attention=None,
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
        lite_attention: LiteAttention instance (optional)

    Notes on the communication pattern:
      - This function assumes sequence-parallel (a.k.a. "context-parallel") inputs:
        each rank holds only a shard of the *sequence length* (L // p), but initially
        still holds all attention heads (N).
      - The key idea is to compute attention with *global* (full-length) K/V by doing
        an all-to-all that trades:
          - heads partition  <->  sequence partition
        so each rank temporarily holds the full sequence length but only (N // p) heads.
      - The two `all_to_all` calls are typically the primary communication bottleneck
        in this attention implementation.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    b = q.shape[0]

    # gather q/k/v sequence
    #
    # all_to_all(x, scatter_dim, gather_dim) (see `wan/distributed/util.py`) does:
    #   1) Split x into `world_size` chunks along `scatter_dim`
    #   2) dist.all_to_all: send chunk i to rank i and receive one chunk from every rank
    #   3) Concatenate received chunks along `gather_dim`
    #
    # Here we call: all_to_all(., scatter_dim=2=heads, gather_dim=1=sequence)
    #
    # If inputs are:
    #   q,k,v: [B, L_local, N, D]   where L_local = L // p and p = world_size
    # then after all-to-all:
    #   q,k,v: [B, L,       N/p, D]
    #
    # Intuition:
    #   - We "scatter" (split) the heads across ranks (each rank keeps only N/p heads)
    #   - We "gather" (concatenate) the sequence shards from every rank (each rank sees full L)
    #
    # Communication volume:
    #   - Each rank sends/receives ~sizeof(q) + sizeof(k) + sizeof(v) worth of data
    #     across the fabric in this phase (amortized across ranks, but still heavy).
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    # Use LiteAttention if available, otherwise fall back to flash_attention
    if lite_attention is not None and LITE_ATTENTION_AVAILABLE:
        # LiteAttention expects (batch, seq_len, heads, head_dim) format
        # and returns (batch, seq_len, heads * head_dim) format
        # Convert to bfloat16 for memory efficiency
        x = lite_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
        # Convert result back to float32 to maintain consistency with model expectations
        x = x.float()
    else:
        x = flash_attention(
            q,
            k,
            v,
            k_lens=seq_lens,
            window_size=window_size,
        )

    # scatter q/k/v sequence
    # Reverse the earlier exchange to return outputs back to sequence-parallel layout.
    #
    # If attention output is:
    #   x: [B, L, N/p, D]
    # then all_to_all(x, scatter_dim=1=sequence, gather_dim=2=heads) produces:
    #   x: [B, L/p, N,   D]
    #
    # This is the second major communication step (another all-to-all).
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x
