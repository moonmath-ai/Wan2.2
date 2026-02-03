#!/usr/bin/env python3
"""Test LiteAttention with fixed threshold (no calibration)."""

import os
from pathlib import Path
import time
from datetime import datetime
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from lite_attention import ModuleRegistry, LiteAttention, LiteAttentionRunConfig

generate_kwargs_default = {
    "frame_num": 81,
    "sampling_steps": 50,
}
generate_kwargs_fast = {
    "frame_num": 21,
    "sampling_steps": 10,
}
generate_kwargs = generate_kwargs_default

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/fixed_th_test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.as_posix()}")

    config = WAN_CONFIGS["i2v-A14B"]
    device = torch.device("cuda:0")

    print("Loading models (this happens only once)...")
    start_load = time.time()

    wan_i2v = wan.WanI2V(
        config=config,
        checkpoint_dir=Path.home()/"weights/Wan2.2-I2V-A14B",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
        lite_attention_threshold=-10.0,
    )

    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.2f}s\n")

    image = Image.open("examples/i2v_input.JPG").convert("RGB")
    prompt = """Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. \
        The fluffy-furred feline gazes directly at the camera with a relaxed expression. \
        Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. \
        The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. \
        A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."""

    threshold = -3.0

    print(f"\n{'='*60}")
    print(f"Running with fixed threshold={threshold} (no calibration)")
    print('='*60)

    # Set fixed threshold for all LiteAttention modules
    registry_low = ModuleRegistry(wan_i2v.low_noise_model.named_modules())
    registry_high = ModuleRegistry(wan_i2v.high_noise_model.named_modules())
    for registry in [registry_low, registry_high]:
        registry.set_bulk_config(LiteAttentionRunConfig(threshold=threshold))

    for model in [wan_i2v.low_noise_model, wan_i2v.high_noise_model]:
        for _name, module in model.named_modules():
            if isinstance(module, LiteAttention):
                module.reset_skip_state()

    sample_la = wan_i2v.low_noise_model.blocks[0].self_attn.lite_attention
    print(f"Verified threshold in module: {sample_la.config.threshold}")

    start_gen = time.time()

    video = wan_i2v.generate(
        input_prompt=prompt,
        img=image,
        max_area=480*832,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=(3.5, 3.5),
        seed=42,
        offload_model=False,
        **generate_kwargs,
    )

    gen_time = time.time() - start_gen

    output_file = output_dir / f"th_{threshold}_output.mp4"
    save_video(tensor=video[None], save_file=output_file, fps=16)

    print(f"\nGeneration time: {gen_time:.2f}s")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
