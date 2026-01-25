#!/bin/bash
set -e

echo "Installing dependencies..."
uv sync --all-extras --python 3.12 --managed-python

echo "Downloading model weights..."
mkdir -p weights
uv run hf download Wan-AI/Wan2.2-T2V-A14B --local-dir ./weights/Wan2.2-T2V-A14B

echo "Setup complete!"
read -p "Test generation? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv --preview-features extra-build-dependencies run --managed-python --python 3.12 --all-extras \
        python generate.py --task t2v-A14B --size 1280*720 \
        --ckpt_dir weights/Wan2.2-T2V-A14B --offload_model True \
        --convert_model_dtype \
        --frame_num 41 --sample_steps 20 \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
fi
