#!/usr/bin/env python3
"""Test LiteAttention with different thresholds, loading weights only once."""

import os
from pathlib import Path
import time
from datetime import datetime
import torch
import numpy as np
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
    thresholds = [-10.0, -3.0, 0.0]

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/threshold_test_{timestamp}")
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
    )

    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.2f}s\n")

    # Load input image
    image = Image.open("examples/i2v_input.JPG").convert("RGB")
    prompt = """Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. \
        The fluffy-furred feline gazes directly at the camera with a relaxed expression. \
        Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. \
        The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. \
        A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."""

    results = []

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Running with threshold={threshold}")
        print('='*60)

        registries: list[ModuleRegistry] = []

        # Update threshold for all LiteAttention modules
        registry_low = ModuleRegistry(wan_i2v.low_noise_model.named_modules())
        registry_high = ModuleRegistry(wan_i2v.high_noise_model.named_modules())
        for registry in [registry_low, registry_high]:
            registry.set_bulk_config(LiteAttentionRunConfig(threshold=threshold))

        for model in [wan_i2v.low_noise_model, wan_i2v.high_noise_model]:
            for _name, module in model.named_modules():
                if isinstance(module, LiteAttention):
                    module.reset_skip_state()

        # Verify LiteAttention is available and threshold is set correctly
        sample_la = wan_i2v.low_noise_model.blocks[0].self_attn.lite_attention
        if sample_la is None:
            print("ERROR: LiteAttention not available! Check import.")
            return
        print(f"Verified threshold in module: {sample_la.config.threshold}")
        print(f"LiteAttention class: {type(sample_la).__module__}.{type(sample_la).__name__}")

        start_gen = time.time()

        video = wan_i2v.generate(
            input_prompt=prompt,
            img=image,
            max_area=480*832,  # 480p (720p failed OOM)
            shift=5.0,
            sample_solver="unipc",
            guide_scale=(3.5, 3.5),
            seed=42,  # fixed seed for comparison
            offload_model=False,  # keep on GPU for speed
            **generate_kwargs,
        )

        gen_time = time.time() - start_gen

        output_file = output_dir / f"threshold_{threshold}_output.mp4"
        save_video(tensor=video[None], save_file=output_file, fps=16)

        registry_low.config_output.save(output_dir/f"config_{threshold}_low.toml")
        registry_high.config_output.save(output_dir/f"config_{threshold}_high.toml")

        # Save first frame as PNG for comparison (no video encoding artifacts)
        for frame in [0, 10, 20]:
            first_frame = video[:, frame, :, :]  # [C, H, W]
            first_frame_np = ((first_frame.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype('uint8')
            frame_file = output_dir / f"threshold_{threshold}_frame0.png"
            Image.fromarray(first_frame_np).save(frame_file)


        # Verify LiteAttention was actually used by checking skip stats
        total_skipped = 0
        total_calls = 0
        for model in [wan_i2v.low_noise_model, wan_i2v.high_noise_model]:
            for block in model.blocks:
                la = block.self_attn.lite_attention
                if la is not None and hasattr(la, 'tiles_skipped'):
                    total_skipped += la.tiles_skipped
                    total_calls += la.tiles_total

        if total_calls > 0:
            skip_pct = 100.0 * total_skipped / total_calls
            print(f"LiteAttention VERIFIED: {total_skipped}/{total_calls} tiles skipped ({skip_pct:.1f}%)")
        else:
            print("WARNING: LiteAttention may not have been used (no tile stats)")

        # Compute video hash for comparison
        video_hash = hash(video.cpu().numpy().tobytes())

        print(f"Generation time: {gen_time:.2f}s")
        print(f"Saved to: {output_file}")
        print(f"First frame: {frame_file}")
        print(f"Video hash: {video_hash}")

        skip_info = f"{total_skipped}/{total_calls}" if total_calls > 0 else "N/A"
        results.append((threshold, gen_time, output_file, video_hash, skip_info))
        del video

    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"{'Threshold':<12} {'Time (s)':<12} {'Tiles Skipped':<18} {'Hash':<22}")
    print('-'*80)
    for threshold, gen_time, output_file, video_hash, skip_info in results:
        print(f"{threshold:<12} {gen_time:<12.2f} {skip_info:<18} {video_hash:<22}")

    # Check if all hashes are the same
    hashes = [r[3] for r in results]
    if len(set(hashes)) == 1:
        print("\nWARNING: All video hashes are identical! Thresholds may not be affecting output.")
    else:
        print("\nVideo hashes differ - thresholds are affecting output.")

    # Write summary to file
    summary_file = output_dir / "summary.txt"
    with summary_file.open("w") as f:
        f.write(f"LiteAttention Threshold Comparison Test\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Threshold':<12} {'Time (s)':<12} {'Tiles Skipped':<18} {'Hash':<22}\n")
        f.write(f"{'-'*80}\n")
        for threshold, gen_time, output_file, video_hash, skip_info in results:
            f.write(f"{threshold:<12} {gen_time:<12.2f} {skip_info:<18} {video_hash:<22}\n")
        f.write(f"\n")
        if len(set(hashes)) == 1:
            f.write("WARNING: All video hashes are identical!\n")
        else:
            f.write("SUCCESS: Video hashes differ - thresholds are affecting output.\n")
    print(f"\nSummary written to: {summary_file}")

if __name__ == "__main__":
    main()
