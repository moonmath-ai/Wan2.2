# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        if not FLASH_ATTN_2_AVAILABLE:
            # Fallback to PyTorch SDPA when flash-attn isn't installed.
            # This keeps the project runnable (at the cost of potential perf differences),
            # and works well when PyTorch can use its own fused SDPA kernels.
            if window_size != (-1, -1):
                warnings.warn(
                    'window_size is not supported without flash-attn; falling back to full attention.'
                )

            # q is either [B*Lq, Hq, D] (q_lens is None) or [sum(q_lens), Hq, D].
            # k/v are either [B*Lk, Hk, D] (k_lens is None) or [sum(k_lens), Hk, D].
            hq, dq = q.size(1), q.size(2)
            hk, dv = k.size(1), v.size(2)

            # Compute per-sample offsets for varlen k/v (and q if needed)
            q_offsets = torch.arange(b + 1, device=q.device, dtype=torch.int64) * lq
            if q_lens is not None:
                q_offsets = torch.cat(
                    [q_lens.new_zeros([1], dtype=torch.int64),
                     q_lens.to(torch.int64).cumsum(0)],
                    dim=0,
                )
            k_offsets = torch.arange(b + 1, device=k.device, dtype=torch.int64) * lk
            if k_lens is not None:
                k_offsets = torch.cat(
                    [k_lens.new_zeros([1], dtype=torch.int64),
                     k_lens.to(torch.int64).cumsum(0)],
                    dim=0,
                )

            # Allocate padded output [B, Lq, Hq, Dv]
            out = q.new_zeros((b, lq, hq, dv))

            enable_gqa = (hq != hk) and (hq % hk == 0)
            for i in range(b):
                qs, qe = int(q_offsets[i].item()), int(q_offsets[i + 1].item())
                ks, ke = int(k_offsets[i].item()), int(k_offsets[i + 1].item())
                if qe <= qs or ke <= ks:
                    continue

                q_i = q[qs:qe]  # [Lqi, Hq, Dq]
                k_i = k[ks:ke]  # [Lki, Hk, Dq]
                v_i = v[ks:ke]  # [Lki, Hk, Dv]

                # SDPA expects [B, H, L, D]
                q_i = q_i.permute(1, 0, 2).unsqueeze(0)
                k_i = k_i.permute(1, 0, 2).unsqueeze(0)
                v_i = v_i.permute(1, 0, 2).unsqueeze(0)

                out_i = torch.nn.functional.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    scale=softmax_scale,
                    enable_gqa=enable_gqa,
                )
                out_i = out_i.squeeze(0).permute(1, 0, 2)  # [Lqi, Hq, Dv]

                # Place back into padded output. When q_lens is None, qe-qs == lq.
                out[i, :out_i.size(0)] = out_i

            x = out
        else:
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
