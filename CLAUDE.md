# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wan2.2 is an advanced video generation model from Alibaba Wan Team. It supports multiple generation tasks:
- **T2V-A14B**: Text-to-Video (14B MoE)
- **I2V-A14B**: Image-to-Video (14B MoE)
- **TI2V-5B**: Text+Image-to-Video (5B efficient model)
- **S2V-14B**: Speech-to-Video
- **Animate-14B**: Character Animation

## Development Environment

### Hardware Requirements
- Requires 80GB GPU for running the 14B models
- Available servers: `nebius-8-e00xty10vh289wtwn7`, `nebius-61-e00g00j5ctcrdb3b45`, `nebius-144-e00zy1ys26yhk70rry`

### Pixi Setup (Recommended)
```bash
./setup_with_pixi.sh
```
This script installs dependencies via pixi, sets up lite-attention, and downloads model weights to `~/weights/`.

After setup, test with:
```bash
pixi run test-i2v   # Image-to-video test
pixi run test-t2v   # Text-to-video test
```

### Working with Remote Servers
The code is developed locally but must run on GPU servers:
- **Code**: `~/code/Wan2.2` (same structure local and remote)
- **Weights**: `~/weights/` on remote servers

Sync changes and run remotely:
```bash
# Always exclude .git (works for both main repo and worktrees)
# Submodules should be updated locally first; setup_with_pixi.sh handles the remote side
rsync -av --exclude='.pixi' --exclude='__pycache__' --exclude='*.egg-info' --exclude='.git' \
    ./ nebius-8-e00xty10vh289wtwn7:~/code/Wan2.2/

# Run setup on remote (pixi needs PATH set)
ssh nebius-8-e00xty10vh289wtwn7 "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/code/Wan2.2 && ./setup_with_pixi.sh"

# Run pixi task on remote
ssh nebius-8-e00xty10vh289wtwn7 "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/code/Wan2.2 && pixi run test-i2v"
```

**Note**: Always set `PATH=$HOME/.pixi/bin:$PATH` when running pixi commands via SSH.

LiteAttention is installed as editable (`pip install -e`), so Python changes take effect immediately. CUDA kernel changes require reinstall: `pixi run install-lite-attention`

### Syncing outputs back from remote
After a remote run, rsync results back to local `~/outputs/`:
```bash
rsync -av nebius-8-e00xty10vh289wtwn7:~/code/Wan2.2/output/<run_dir>/ ~/outputs/<run_dir>/
# Include the log too:
rsync -av nebius-8-e00xty10vh289wtwn7:~/code/Wan2.2/run_sweep.log ~/outputs/<run_dir>/
```

## Common Commands

### Installation (pip)
```bash
pip install .                    # Basic install
pip install ".[s2v]"             # With speech-to-video deps
pip install ".[animate]"         # With animation deps
pip install ".[lite]"            # With LiteAttention deps
```

### Installation (pixi)
```bash
pixi install
pixi run install-lite-attention  # If lite-attention not installed
```

### Running Generation
```bash
# Single GPU (with pixi) - use 480*832 for faster test runs
pixi run python generate.py --task t2v-A14B --size 480*832 \
    --ckpt_dir ~/weights/Wan2.2-T2V-A14B --offload_model True \
    --frame_num 41 --sample_steps 20 --prompt "your prompt"

# Single GPU (without pixi)
python generate.py --task t2v-A14B --size 480*832 --ckpt_dir /path/to/Wan2.2-T2V-A14B

# Multi-GPU with FSDP and sequence parallelism
torchrun --nproc_per_node=4 generate.py --task t2v-A14B --ckpt_dir /path/to/model \
    --size 1280*720 --dit_fsdp --t5_fsdp --ulysses_size 4
```

### Testing
```bash
bash tests/test.sh <model_dir> <gpu_count>
```

### Formatting
```bash
make format   # Runs isort and yapf
```

## Architecture

### Pipeline Classes (`wan/`)
Each task has a dedicated pipeline class that orchestrates the generation:
- `WanT2V` (text2video.py) - Text-to-video generation
- `WanI2V` (image2video.py) - Image-to-video generation
- `WanTI2V` (textimage2video.py) - Combined text+image conditioning
- `WanS2V` (speech2video.py) - Speech-driven generation
- `WanAnimate` (animate.py) - Character animation

### Core Components (`wan/modules/`)
- **model.py**: `WanModel` - Main DiT transformer with RoPE embeddings and MoE support
- **attention.py**: Flash attention implementations with optional LiteAttention
- **t5.py**: T5 text encoder wrapper (google/umt5-xxl)
- **vae2_1.py / vae2_2.py**: VAE decoders (4×8×8 and 4×16×16 compression)

### Mixture-of-Experts (MoE)
The 14B models use two expert networks (27B total params, 14B active):
- **High-noise expert**: Used for early denoising phases (t > t_moe)
- **Low-noise expert**: Used for detail refinement (t < t_moe)
- Selection based on Signal-to-Noise Ratio (SNR)

### Distributed Inference (`wan/distributed/`)
- **fsdp.py**: Model sharding across GPUs (reduces per-GPU memory)
- **ulysses.py**: DeepSpeed Ulysses sequence parallelism
- Use `--dit_fsdp`, `--t5_fsdp`, `--ulysses_size N` flags

### Configuration (`wan/configs/`)
Model configs registered in `WAN_CONFIGS` dict. Size presets in `SIZE_CONFIGS`:
- 720P: 1280×720, 720×1280
- 480P: 480×832, 832×480
- TI2V-5B specific: 1280×704, 704×1280

### Key Flags for generate.py
- `--task`: Model variant (t2v-A14B, i2v-A14B, ti2v-5B, s2v-14B, animate-14B)
- `--size`: Resolution (e.g., 1280*720)
- `--frame_num`: Number of frames (default 81, must be 4k+1)
- `--offload_model`: Move model to CPU after use (saves VRAM)
- `--t5_cpu`: Keep T5 on CPU
- `--lite_attention_threshold`: Enable LiteAttention for memory efficiency
