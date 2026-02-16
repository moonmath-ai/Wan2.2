#!/usr/bin/env python3
"""Unified LiteAttention test script.

Edit RUNS below to control what gets tested. Each entry is either a CalibRun
(calibration mode with target_error) or a ConstRun (fixed threshold mode).
The model is loaded once; each run reuses it.

Each run can specify its own size and frame_num for different resolutions.

Output structure: output/{YYYYMMDD}_{mode}_{index}/
"""

import io
import os
import sys
# Disable the measurement hack in lite_attention.py
os.environ.pop("LITE_ATTENTION_MEASURE", None)

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import subprocess
import time

import torch
from huggingface_hub import snapshot_download
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from lite_attention import LiteAttentionRegistry, LiteAttention

# ---------------------------------------------------------------------------
# Configure runs here
# ---------------------------------------------------------------------------

@dataclass
class Run:
    mode: str
    la_kwargs: dict[str, Any]
    size: str = "480*832"
    frame_num: int = 21
    sampling_steps: int = 10

# Huge sweep: 480x832, 41 frames, 40 sampling steps
_SIZE, _FRAMES, _STEPS = '480*832', 41, 40
_C = lambda th: Run('const', {'threshold': th}, _SIZE, _FRAMES, _STEPS)
_L1 = lambda te: Run('calib', {'calib_config': {'target_error': te, 'metric': 'L1'}}, _SIZE, _FRAMES, _STEPS)
_RMSE = lambda te: Run('calib', {'calib_config': {'target_error': te, 'metric': 'RMSE'}}, _SIZE, _FRAMES, _STEPS)
_COS = lambda te: Run('calib', {'calib_config': {'target_error': te, 'metric': 'Cossim'}}, _SIZE, _FRAMES, _STEPS)

RUNS: list[Run] = [
    # Constant thresholds
    _C(0), _C(-2), _C(-4), _C(-8), _C(-10),
    # L1 calibration
    _L1(0.1), _L1(0.3), _L1(0.1), _L1(0.03), _L1(0.01), _L1(0.003), _L1(0.001),
    # RMSE calibration
    _RMSE(0.1), _RMSE(0.3), _RMSE(0.1), _RMSE(0.03), _RMSE(0.01), _RMSE(0.003), _RMSE(0.001),
    # Cossim calibration
    _COS(0.1), _COS(0.03), _COS(0.01), _COS(0.003), _COS(0.001),
]

# ---------------------------------------------------------------------------

IMAGE_PATH = "examples/i2v_input.JPG"
PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing atmosphere of the seaside."
)
SEED = 42
SAVE_FRAMES = [0, 10, 20]


def get_git_info() -> dict[str, str]:
    """Collect git commit hash, summary, dirty status, and diff."""
    def _run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()

    commit_hash = _run(["git", "rev-parse", "HEAD"])
    commit_short = _run(["git", "rev-parse", "--short", "HEAD"])
    commit_summary = _run(["git", "log", "-1", "--format=%s"])
    is_dirty = _run(["git", "status", "--porcelain"]) != ""
    diff = _run(["git", "diff", "HEAD"]) if is_dirty else ""
    return {
        "hash": commit_hash,
        "short": commit_short,
        "summary": commit_summary,
        "dirty": is_dirty,
        "diff": diff,
    }


def format_git_info(git_info: dict[str, str]) -> str:
    dirty_marker = " (dirty)" if git_info["dirty"] else ""
    header = f"Commit: {git_info['short']}{dirty_marker} — {git_info['summary']}"
    if not git_info["diff"]:
        return header
    return f"{header}\n\nUncommitted changes:\n{git_info['diff']}"


def run_label(run: Run) -> str:
    if run.mode == 'calib':
        cc = run.la_kwargs['calib_config']
        metric = cc.get('metric', 'L1')
        return f"{run.size}x{run.frame_num}f_{run.sampling_steps}s_{metric}={cc['target_error']}"
    return f"{run.size}x{run.frame_num}f_{run.sampling_steps}s_const={run.la_kwargs['threshold']}"


def setup_registries(wan_i2v, run: Run, output_dir: Path):
    """Configure LiteAttention registries for a single run."""
    label = run_label(run)
    filename = output_dir / f"config_low_{label}.toml" if run.mode == 'calib' else None
    reg_low = LiteAttentionRegistry.from_model(
        wan_i2v.low_noise_model, mode=run.mode,
        filename=filename,
        **run.la_kwargs,
    )
    filename = output_dir / f"config_high_{label}.toml" if run.mode == 'calib' else None
    reg_high = LiteAttentionRegistry.from_model(
        wan_i2v.high_noise_model, mode=run.mode,
        filename=filename,
        **run.la_kwargs,
    )
    return reg_low, reg_high


def reset_lite_attention(wan_i2v):
    for model in [wan_i2v.low_noise_model, wan_i2v.high_noise_model]:
        for _name, module in model.named_modules():
            if isinstance(module, LiteAttention):
                module.reset_skip_state()


def collect_skip_stats(wan_i2v):
    """Compute average skip percentage from LiteAttention skip lists."""
    percentages = []
    for model in [wan_i2v.low_noise_model, wan_i2v.high_noise_model]:
        for _name, module in model.named_modules():
            if isinstance(module, LiteAttention) and module._skip_list is not None:
                # _skip_list shape: [2, batch, heads, qtiles, ktiles+1]
                # read buffer is at current _phase index
                read_list = module._skip_list[module._phase]
                pct_computed = LiteAttention.calc_percentage(read_list).item()
                percentages.append(1.0 - pct_computed)  # skip %
    if percentages:
        avg_skip = sum(percentages) / len(percentages)
        return avg_skip, len(percentages)
    return None, 0


class _Tee:
    """Write to both the original stream and a capture buffer."""
    def __init__(self, original, buffer):
        self._original = original
        self._buffer = buffer

    def write(self, text):
        self._original.write(text)
        self._buffer.write(text)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class TeeCapture:
    """Context manager that captures stdout+stderr while still printing."""
    def __init__(self):
        self.buffer = io.StringIO()
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _Tee(self._old_stdout, self.buffer)
        sys.stderr = _Tee(self._old_stderr, self.buffer)
        return self

    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    @property
    def text(self):
        return self.buffer.getvalue()


def convergence_suffix(captured_text: str) -> str:
    """Return filename suffix based on calibration convergence warnings."""
    suffixes = []
    if "using the high limit" in captured_text:
        suffixes.append("_or_lower")
    if "using the low limit" in captured_text:
        suffixes.append("_or_higher")
    if "binary search did not converge" in captured_text:
        suffixes.append("_non_converge")
    return "".join(suffixes)


def save_frames(video, output_dir, prefix, frame_indices=None):
    if frame_indices is None:
        frame_indices = SAVE_FRAMES
    for frame_idx in frame_indices:
        frame = video[:, frame_idx, :, :]
        frame_np = ((frame.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype("uint8")
        Image.fromarray(frame_np).save(output_dir / f"{prefix}_frame{frame_idx}.png")


def main():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_info = get_git_info()

    print(f"Planned runs ({len(RUNS)}):")
    for i, run in enumerate(RUNS):
        print(f"  [{i}] {run_label(run)}")

    # Load model once
    config = WAN_CONFIGS["i2v-A14B"]
    print("\nLoading models (once)...")
    start_load = time.time()
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
    print(f"Model loaded in {time.time() - start_load:.2f}s\n")

    image = Image.open(IMAGE_PATH).convert("RGB")

    results = []
    mode_all = RUNS[0].mode if all(run.mode == RUNS[0].mode for run in RUNS) else 'mixed'
    output_dir = Path(f"output/{date_str}_{mode_all}_{len(RUNS)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, run in enumerate(RUNS):
        h, w = [int(x) for x in run.size.split("*")]
        max_area = h * w

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(RUNS)}] {run_label(run)}")
        print(f"Output: {output_dir}")
        print("=" * 60)

        reg_low, reg_high = setup_registries(wan_i2v, run, output_dir)
        reset_lite_attention(wan_i2v)

        start_gen = time.time()
        with TeeCapture() as capture:
            video = wan_i2v.generate(
                input_prompt=PROMPT,
                img=image,
                max_area=max_area,
                shift=5.0,
                sample_solver="unipc",
                guide_scale=(3.5, 3.5),
                seed=SEED,
                offload_model=False,
                frame_num=run.frame_num,
                sampling_steps=run.sampling_steps,
            )
        gen_time = time.time() - start_gen

        # Detect calibration convergence issues
        suffix = convergence_suffix(capture.text) if run.mode == 'calib' else ""
        label = run_label(run) + suffix

        # Save video
        video_file = output_dir / f"output_{label}.mp4"
        save_video(tensor=video[None], save_file=video_file, fps=16)

        # Save calib configs if applicable
        reg_low.save_if_calib()
        reg_high.save_if_calib()

        # Save frames (pick first, middle, last)
        n_frames = video.shape[1]
        frame_indices = [0, n_frames // 2, n_frames - 1]
        save_frames(video, output_dir, f"frame_{label}", frame_indices)

        # Collect stats
        avg_skip, n_layers = collect_skip_stats(wan_i2v)
        if avg_skip is not None:
            skip_info = f"{avg_skip:.1%} avg ({n_layers} layers)"
            print(f"Tiles skipped: {skip_info}")
        else:
            skip_info = "N/A"
            print("WARNING: no skip lists found")

        video_hash = hash(video.cpu().numpy().tobytes())
        print(f"Time: {gen_time:.2f}s | Hash: {video_hash}")

        results.append({
            "index": i,
            "label": label,
            "time": gen_time,
            "skip_info": skip_info,
            "hash": video_hash,
            "output_dir": str(output_dir),
        })
        del video

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(format_git_info(git_info))
    print()
    print(f"{'#':<4} {'Run':<55} {'Time (s)':<12} {'Tiles Skipped':<24} {'Hash'}")
    print("-" * 105)
    for r in results:
        print(f"{r['index']:<4} {r['label']:<55} {r['time']:<12.2f} {r['skip_info']:<24} {r['hash']}")

    hashes = [r["hash"] for r in results]
    if len(set(hashes)) == 1:
        print("\nWARNING: All video hashes identical — parameters may not be taking effect.")
    else:
        print("\nVideo hashes differ — parameters are affecting output.")

    # Write summary to first output dir's parent
    summary_file = output_dir / "summary.txt"
    with summary_file.open("w") as f:
        f.write(f"LiteAttention Test — {mode_all}\n")
        f.write(f"Date: {date_str}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(format_git_info(git_info))
        f.write("\n\n")
        f.write(f"{'#':<4} {'Run':<45} {'Time (s)':<12} {'Tiles Skipped':<24} {'Hash'}\n")
        f.write(f"{'-' * 95}\n")
        for r in results:
            f.write(f"{r['index']:<4} {r['label']:<55} {r['time']:<12.2f} {r['skip_info']:<24} {r['hash']}\n")
        f.write("\n")
        if len(set(hashes)) == 1:
            f.write("WARNING: All video hashes identical.\n")
        else:
            f.write("SUCCESS: Video hashes differ.\n")
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
