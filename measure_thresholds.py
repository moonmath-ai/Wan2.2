#!/usr/bin/env python3
"""Measure LiteAttention errors for different thresholds.

Loads model once, runs a single generation with LITE_ATTENTION_MEASURE set.
The hack in lite_attention.py will print errors for all thresholds on the
first LiteAttention forward call, then continue with normal generation.

Usage:
    pixi run python measure_thresholds.py [--size 480*832] [--frame_num 21]
"""

import os
import sys
import time
from pathlib import Path

# Set the measurement env var BEFORE importing anything that loads LiteAttention
THRESHOLDS = "0,-1,-3,-10,-15,-20,-30"
os.environ["LITE_ATTENTION_MEASURE"] = THRESHOLDS
os.environ["LITE_ATTENTION_DEBUG"] = "TRUE"  # allow threshold=0

import argparse

import torch
from huggingface_hub import snapshot_download
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from lite_attention import LiteAttentionRegistry

IMAGE_PATH = "examples/i2v_input.JPG"
PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds."
)
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="480*832",
                        help="Resolution, e.g. 480*832 or 1280*720")
    parser.add_argument("--frame_num", type=int, default=21,
                        help="Number of frames (must be 4k+1)")
    parser.add_argument("--sample_steps", type=int, default=10)
    args = parser.parse_args()

    h, w = [int(x) for x in args.size.split("*")]
    max_area = h * w

    print(f"Settings: size={args.size}, frame_num={args.frame_num}, steps={args.sample_steps}")
    print(f"Thresholds to measure: {THRESHOLDS}")
    print()

    # Load model once
    config = WAN_CONFIGS["i2v-A14B"]
    print("Loading models...")
    start = time.time()
    wan_i2v = wan.WanI2V(
        config=config,
        checkpoint_dir=snapshot_download("Wan-AI/Wan2.2-I2V-A14B"),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )
    print(f"Model loaded in {time.time() - start:.1f}s\n")

    image = Image.open(IMAGE_PATH).convert("RGB")

    # Set up LiteAttention with any const threshold (doesn't matter, measurement uses its own)
    reg_low = LiteAttentionRegistry.from_model(
        wan_i2v.low_noise_model, mode='const', threshold=-10.0)
    reg_high = LiteAttentionRegistry.from_model(
        wan_i2v.high_noise_model, mode='const', threshold=-10.0)

    print("Starting generation (will print threshold measurements on first attention call)...\n")
    video = wan_i2v.generate(
        input_prompt=PROMPT,
        img=image,
        max_area=max_area,
        shift=5.0,
        sample_solver="unipc",
        guide_scale=(3.5, 3.5),
        seed=SEED,
        offload_model=False,
        frame_num=args.frame_num,
        sampling_steps=args.sample_steps,
    )
    print("\nDone. (Video generation completed but we only needed the first attention call)")


if __name__ == "__main__":
    main()
