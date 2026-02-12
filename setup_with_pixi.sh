#!/bin/bash
set -e

# Check for pixi command
if ! command -v pixi &> /dev/null; then
    echo "Error: 'pixi' command not found."
    echo "Please install pixi first: https://pixi.prefix.dev/latest/installation/"
    echo "and verify that it is in your PATH."
    exit 1
fi

echo "Installing dependencies..."
pixi clean
if [ -d ".git" ]; then
    git clean -Xf
    if [ -d "LiteAttention/.git" ]; then
        git -C LiteAttention clean -Xf
    else
        git submodule update --init --recursive
    fi
fi
pixi install

pixi run ninja --version || exit 1
pixi run python -c "import flash_attn" || exit 1

echo "Installing lite-attention..."
pixi run install-lite-attention
pixi run python -c "import lite_attention" || exit 1

echo "Downloading model weights..."
pixi run hf download Wan-AI/Wan2.2-T2V-A14B
pixi run hf download Wan-AI/Wan2.2-I2V-A14B

echo "Setup complete!"
echo "Run 'pixi run test-i2v' or 'pixi run test-t2v' to test generation."
