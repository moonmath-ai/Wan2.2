# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    Perform an all-to-all exchange that *re-partitions* a tensor.

    Conceptually:
      - Split (`scatter`) the local tensor into `world_size` chunks along `scatter_dim`
      - Exchange those chunks so each rank receives one chunk from every other rank
      - Concatenate (`gather`) the received chunks along `gather_dim`

    This is commonly used to trade one type of parallelism for another, e.g.:
      - scatter heads, gather sequence (Ulysses attention)
      - scatter sequence, gather heads (reverse operation)

    Preconditions / gotchas:
      - `x.size(scatter_dim)` must be divisible by `world_size` because `Tensor.chunk`
        is used to split into equal-sized chunks.
      - All ranks must call this with tensors whose chunk shapes match exactly,
        otherwise `dist.all_to_all` will error or hang.
    """
    world_size = get_world_size()
    if world_size > 1:
        # Split local tensor into per-destination chunks.
        # `inputs[i]` is the chunk this rank will SEND to rank i.
        #
        # Note: `.contiguous()` avoids any non-contiguous-stride pitfalls in NCCL
        # collectives and can reduce unexpected overhead from implicit copies.
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]

        # Allocate receive buffers for chunks that will be RECEIVED from every rank.
        # After `dist.all_to_all`, `outputs[i]` holds the chunk received from rank i.
        outputs = [torch.empty_like(u) for u in inputs]

        # Collective exchange:
        #   - every rank sends `inputs[i]` to rank i
        #   - every rank receives one chunk from every other rank into `outputs`
        #
        # Communication cost: this is an all-to-all, typically bandwidth-heavy
        # and often a top bottleneck at scale.
        dist.all_to_all(outputs, inputs, group=group, **kwargs)

        # Reassemble the new layout by concatenating received chunks.
        # This changes the "partitioned" dimension from `scatter_dim` to `gather_dim`.
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
