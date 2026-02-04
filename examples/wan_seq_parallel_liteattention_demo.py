import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from lite_attention import SeqParallelLiteAttention
except ImportError as e:
    raise ImportError(
        "This demo requires `lite_attention` (SeqParallelLiteAttention). "
        "Install from https://github.com/moonmath-ai/LiteAttention (hopper)."
    ) from e


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ----------------------------
    # Wan-like attention shapes
    # ----------------------------
    # Typical Wan2.2-14B head config: num_heads=40, head_dim=128 (dim=5120)
    B = 2       # batch (e.g. CFG often runs 2x)
    S = 2048    # full sequence length (must be divisible by world_size)
    H = 40      # heads
    D = 128     # head dim

    assert S % world_size == 0, "S must be divisible by world_size for this demo."
    shard = S // world_size

    # Create deterministic full Q/K/V on every rank, then shard like Wan sequence-parallel inputs.
    torch.manual_seed(0)
    q_full = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

    start = rank * shard
    end = start + shard

    # This matches the "sequence-parallel input" layout used by Wan:
    # each rank holds only a shard of the sequence dimension.
    q_local = q_full[:, start:end].contiguous()  # [B, shard, H, D]
    k_local = k_full[:, start:end].contiguous()  # [B, shard, H, D]
    v_local = v_full[:, start:end].contiguous()  # [B, shard, H, D]

    # ----------------------------
    # KV-sharded sequence-parallel attention
    # ----------------------------
    # Everybody needs FULL Q; all-gather local shards.
    q_parts = [torch.empty_like(q_local) for _ in range(world_size)]
    dist.all_gather(q_parts, q_local)
    q_full_gathered = torch.cat(q_parts, dim=1).contiguous()

    scale = 1.0 / math.sqrt(D)

    # IMPORTANT: keep skipping disabled for this demo (matches LiteAttention's seq-par example).
    attn = SeqParallelLiteAttention(
        num_nodes=world_size,
        enable_skipping=False,
        threshold=-10.0,
        max_batch_size=B,
        use_int8=False,
    )

    out_partial, lse_partial = attn(
        q_full_gathered,
        k_local,
        v_local,
        split_idx=rank,
        scale=scale,
        return_softmax_lse=True,
    )

    # LiteAttention returns:
    # - out_partial: [B, S, H, D]
    # - lse_partial: [B, H, S]
    if lse_partial.size(1) == H and lse_partial.size(2) == S:
        lse_partial = lse_partial.permute(0, 2, 1).contiguous()  # [B, S, H]
    else:
        raise ValueError(f"Unexpected lse shape: {tuple(lse_partial.shape)}")

    # ----------------------------
    # Merge across ranks using LSE, then reduce-scatter (no full output gather)
    # ----------------------------
    lse_partial = lse_partial.float()
    m = lse_partial.clone()
    dist.all_reduce(m, op=dist.ReduceOp.MAX)
    s = torch.exp(lse_partial - m)
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    lse_global = m + torch.log(s)

    w = torch.exp(lse_partial - lse_global).to(out_partial.dtype)  # [B, S, H]
    out_weighted = out_partial * w.unsqueeze(-1)                   # [B, S, H, D]

    # Reduce-scatter along sequence dimension so each rank gets [B, shard, H, D]
    in_rs = out_weighted.permute(1, 0, 2, 3).contiguous()          # [S, B, H, D]
    out_rs = torch.empty((shard, B, H, D), device=device, dtype=out_weighted.dtype)
    dist.reduce_scatter_tensor(out_rs, in_rs, op=dist.ReduceOp.SUM)
    out_local = out_rs.permute(1, 0, 2, 3).contiguous()            # [B, shard, H, D]

    # ----------------------------
    # (Optional) correctness sanity check vs full attention (local compute only)
    # ----------------------------
    q_ref = q_full.permute(0, 2, 1, 3)
    k_ref = k_full.permute(0, 2, 1, 3)
    v_ref = v_full.permute(0, 2, 1, 3)
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False, scale=scale)
    out_ref = out_ref.permute(0, 2, 1, 3).contiguous()             # [B, S, H, D]
    err = (out_local.float() - out_ref[:, start:end].float()).abs().max()

    if rank == 0:
        print(f"world_size={world_size}, S={S}, shard={shard}, B={B}, H={H}, D={D}")
    print(f"[rank {rank}] max_abs_error(local shard): {err.item():.6g}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

