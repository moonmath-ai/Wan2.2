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

read -p "Test image2video? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pixi run \
        python generate.py --task i2v-A14B --size 480*832 \
        --ckpt_dir ~/weights/Wan2.2-I2V-A14B --offload_model True \
        --convert_model_dtype \
        --frame_num 41 --sample_steps 20 \
        --image examples/i2v_input.JPG \
        --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
fi
echo

read -p "Test text2video? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pixi run \
        python generate.py --task t2v-A14B --size 480*832 \
        --ckpt_dir ~/weights/Wan2.2-T2V-A14B --offload_model True \
        --convert_model_dtype \
        --frame_num 41 --sample_steps 20 \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
fi
