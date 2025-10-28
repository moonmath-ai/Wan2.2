#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
elif command -v nvidia-smi >/dev/null 2>&1; then
  NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
  NPROC=$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
fi

if [ "$NPROC" -gt 1 ] && [ $(($NPROC % 2)) -ne 0 ]; then
  echo "When using more than 1 GPU, the number must be even (found $NPROC)"
  exit 1
fi


CUR_DIR=$(pwd)

TASK="ti2v-5B"
SIZE="1280*704"
CKPT_DIR="./Wan2.2-TI2V-5B"
IMAGE="dragon-warrior.jpg"
PROMPT="The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character. At the last moment, the dragon warriors head slowly turns to look directly at the camera, its eyes opening wide."
SEED=6599114657078355688
STEPS=50
FRAMES=121
LA1_THRESHOLD=-3.75
OUTPUT="$CUR_DIR/output/dragon-warrior-$TASK-$SEED-$STEPS.mp4"

# Run command
torchrun --nproc_per_node=${NPROC} generate.py \
  --task "$TASK" \
  --size "$SIZE" \
  --ckpt_dir "$CKPT_DIR" \
  --image "$IMAGE" \
  --ulysses_size ${NPROC} \
  --prompt "$PROMPT" \
  --save_file "$OUTPUT" \
  --la1_threshold $LA1_THRESHOLD  \
  --frame_num $FRAMES \
  --base_seed $SEED \
  --sample_steps $STEPS \
  --t5_fsdp \
  --dit_fsdp \
  --convert_model_dtype