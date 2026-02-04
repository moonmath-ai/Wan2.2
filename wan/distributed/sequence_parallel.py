# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.distributed as dist

from ..modules.model import sinusoidal_embedding_1d
from .util import gather_forward, get_rank, get_world_size, all_to_all, gather_forward
from ..modules.attention import flash_attention

# Import lite_attention for optimized attention / sequence parallelism
try:
    from lite_attention import LiteAttention
except ImportError:
    LiteAttention = None

try:
    from lite_attention import SeqParallelLiteAttention
except ImportError:
    SeqParallelLiteAttention = None

LITE_ATTENTION_AVAILABLE = LiteAttention is not None
SEQ_PAR_LITE_ATTENTION_AVAILABLE = SeqParallelLiteAttention is not None


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
        lite_attention=getattr(self, 'lite_attention', None),
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x

def _get_seq_parallel_attn(lite_attention, world_size, batch_size):
    """
    Returns a `SeqParallelLiteAttention` instance.

    - If `lite_attention` is provided (typically per-layer), we attach a cached
      `SeqParallelLiteAttention` instance to it to avoid sharing skip-state across layers.
    - If `lite_attention` is None, we use a module-level cached instance with skipping disabled.
    """
    if SeqParallelLiteAttention is None:
        raise RuntimeError("SeqParallelLiteAttention is not available.")

    # Defaults (match LiteAttention README).
    enable_skipping = True
    threshold = -10.0
    use_int8 = False

    if lite_attention is not None:
        # If caller already passed a SeqParallelLiteAttention instance, just use it.
        if isinstance(lite_attention, SeqParallelLiteAttention):
            return lite_attention

        # Mirror LiteAttention config when possible.
        enable_skipping = bool(getattr(lite_attention, "enable_skipping", enable_skipping))
        threshold = float(getattr(lite_attention, "threshold", threshold))
        use_int8 = bool(getattr(lite_attention, "use_int8", use_int8))

        cfg = (world_size, enable_skipping, threshold, int(batch_size), use_int8)
        cached = getattr(lite_attention, "_seq_parallel_attn", None)
        cached_cfg = getattr(lite_attention, "_seq_parallel_attn_cfg", None)
        if cached is None or cached_cfg != cfg:
            cached = SeqParallelLiteAttention(
                num_nodes=world_size,
                enable_skipping=enable_skipping,
                threshold=threshold,
                max_batch_size=int(batch_size),
                use_int8=use_int8,
            )
            lite_attention._seq_parallel_attn = cached
            lite_attention._seq_parallel_attn_cfg = cfg
        return cached

    # No per-layer object to stash state on; keep a single cached instance with skipping disabled.
    global _GLOBAL_SEQ_PAR_ATTN
    global _GLOBAL_SEQ_PAR_ATTN_CFG
    if "_GLOBAL_SEQ_PAR_ATTN" not in globals():
        _GLOBAL_SEQ_PAR_ATTN = None
        _GLOBAL_SEQ_PAR_ATTN_CFG = None

    cfg = (world_size, False, -10.0, int(batch_size), False)
    if _GLOBAL_SEQ_PAR_ATTN is None or _GLOBAL_SEQ_PAR_ATTN_CFG != cfg:
        _GLOBAL_SEQ_PAR_ATTN = SeqParallelLiteAttention(
            num_nodes=world_size,
            enable_skipping=False,
            threshold=-10.0,
            max_batch_size=int(batch_size),
            use_int8=False,
        )
        _GLOBAL_SEQ_PAR_ATTN_CFG = cfg
    return _GLOBAL_SEQ_PAR_ATTN


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        lite_attention=None,
):
    """
    Performs distributed attention for context-parallel (sequence-parallel) inputs.

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
        lite_attention: Optional LiteAttention/SeqParallelLiteAttention instance. When provided,
                       it's used as the per-layer place to cache SeqParallelLiteAttention so
                       skip-state isn't shared across layers.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # ----------------------------
    # Single-rank fast path
    # ----------------------------
    if world_size == 1:
        if lite_attention is not None and LITE_ATTENTION_AVAILABLE:
            x = lite_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
            return x.float()
        return flash_attention(
            q,
            k,
            v,
            k_lens=seq_lens,
            window_size=window_size,
        )

    # ----------------------------
    # Preferred: LiteAttention sequence-parallel (KV-sharded) + LSE merge
    # ----------------------------
    b, lq_local, hq, dq = q.shape
    lk_local = k.size(1)
    hk = k.size(2)
    hv = v.size(2)
    dv = v.size(-1)

    use_seq_par = (
        SEQ_PAR_LITE_ATTENTION_AVAILABLE
        and window_size == (-1, -1)
        and (hq == hk == hv)
    )

    # SeqParallelLiteAttention does not currently accept per-sample k_lens masking.
    # If we're using padding (seq_lens < full length), fall back to the original path.
    if use_seq_par and seq_lens is not None:
        expected_l_full = int(lq_local * world_size)
        if int(seq_lens.max().item()) != expected_l_full:
            use_seq_par = False

    if use_seq_par:
        # 1) All-gather Q so every rank has the full query sequence.
        q_full = gather_forward(q, dim=1)  # [B, L_full, H, D]
        l_full = q_full.size(1)

        # Ensure this is the expected equal-shard layout.
        if l_full == lq_local * world_size and lk_local == lq_local:
            scale = 1.0 / (dq**0.5)
            attn = _get_seq_parallel_attn(
                lite_attention=lite_attention,
                world_size=world_size,
                batch_size=b,
            )

            out_local_flat, lse_local = attn(
                q_full.bfloat16(),
                k.bfloat16(),
                v.bfloat16(),
                split_idx=rank,
                scale=scale,
                return_softmax_lse=True,
            )

            # out_local_flat: [B, L_full, H*Dv] -> [B, L_full, H, Dv]
            out_local = out_local_flat.view(b, l_full, hq, dv)

            # lse_local expected: [B, H, L_full] or [B, L_full, H] -> [B, L_full, H]
            if lse_local.dim() != 3:
                raise ValueError(
                    f"Unexpected lse_local shape from SeqParallelLiteAttention: {tuple(lse_local.shape)}"
                )
            if lse_local.size(1) == hq and lse_local.size(2) == l_full:
                lse_local = lse_local.permute(0, 2, 1).contiguous()
            elif lse_local.size(1) == l_full and lse_local.size(2) == hq:
                lse_local = lse_local.contiguous()
            else:
                raise ValueError(
                    f"Unexpected lse_local shape from SeqParallelLiteAttention: {tuple(lse_local.shape)}"
                )
            lse_local = lse_local.to(torch.float32)

            # Merge partial results across ranks using softmax LSE.
            m = lse_local.clone()
            dist.all_reduce(m, op=dist.ReduceOp.MAX)  # m = max_r LSE_r
            s = torch.exp(lse_local - m)
            dist.all_reduce(s, op=dist.ReduceOp.SUM)  # s = sum_r exp(LSE_r - m)
            lse_global = m + torch.log(s)

            w = torch.exp(lse_local - lse_global).to(out_local.dtype)  # [B, L_full, H]
            out_weighted = out_local * w.unsqueeze(-1)                 # [B, L_full, H, Dv]

            # TODO: Finish reading this

            # Avoid replicating the full output: reduce-scatter along the sequence dimension.
            if hasattr(dist, "reduce_scatter_tensor"):
                in_rs = out_weighted.permute(1, 0, 2, 3).contiguous()  # [L_full, B, H, Dv]
                out_rs = torch.empty(
                    (lq_local, b, hq, dv),
                    device=in_rs.device,
                    dtype=in_rs.dtype,
                )
                dist.reduce_scatter_tensor(out_rs, in_rs, op=dist.ReduceOp.SUM)
                x = out_rs.permute(1, 0, 2, 3).contiguous()  # [B, L_local, H, Dv]
            else:
                # Older PyTorch: fall back to all-reduce then slice (replicates full output).
                dist.all_reduce(out_weighted, op=dist.ReduceOp.SUM)
                start = rank * lq_local
                end = start + lq_local
                x = out_weighted[:, start:end].contiguous()

            # Preserve previous LiteAttention behavior of returning float32 when a LiteAttention
            # object is provided (pipelines expect this), otherwise keep half dtype.
            return x.float() if lite_attention is not None else x
    else: 
        raise NotImplementedError("Non-distributed attention is not implemented in this context.")

