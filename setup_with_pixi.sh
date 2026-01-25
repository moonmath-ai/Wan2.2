#!/bin/bash
set -e

echo "Installing dependencies..."
pixi clean
git submodule update --init --recursive
pixi install

echo "Checking ninja..."
if pixi run ninja --version && [ $? -eq 0 ]; then
    echo "ninja is working"
else
    echo "WARNING: ninja is not working correctly, build may be slow"
fi

echo "Verifying flash-attn..."
if ! pixi run python -c "import flash_attn"; then
    echo "ERROR: flash-attn not working"
    exit 1
fi

echo "Installing lite-attention..."
pixi run install-lite-attention
if ! pixi run python -c "import lite_attention"; then
    echo "ERROR: lite-attention installation failed"
    exit 1
fi

echo "Downloading model weights..."
mkdir -p ~/weights
pixi run hf download Wan-AI/Wan2.2-T2V-A14B --local-dir ~/weights/Wan2.2-T2V-A14B
pixi run hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ~/weights/Wan2.2-I2V-A14B

echo "Setup complete!"
echo "Run 'pixi run test-i2v' or 'pixi run test-t2v' to test generation."
