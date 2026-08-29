#!/usr/bin/env bash
# Full-data sweeps for both GenImage++ detectors (CLIP baseline, CLIP-LoRA)
# across all 5 generators, balanced the same way as UniversalFakeDetect/NPR
# (max_sample = fake-pool size, applied per class, so the 162k shared real
# pool isn't redundantly run in full for flux/sd3/sdxl).
set -e
cd /home/student/data/genimagepp_detectors
source /home/student/qwen_env/bin/activate

BATCH_SIZE=16

BIGGAN_REAL="/home/student/Downloads/imagenet_ai_0419_biggan/val/nature"
BIGGAN_FAKE="/home/student/Downloads/imagenet_ai_0419_biggan/val/ai"
BIGGAN_TRAIN_REAL="/home/student/Downloads/imagenet_ai_0419_biggan/train/nature"
GLIDE_REAL="/home/student/Downloads/imagenet_glide/val/nature"
GLIDE_FAKE="/home/student/Downloads/imagenet_glide/val/ai"
FLUX_FAKE="/home/student/data/genimagepp/flux/flux/val/1_fake"
SD3_FAKE="/home/student/data/genimagepp/sd3/stable_diffusion_v_3_0/val/1_fake"
SDXL_FAKE="/home/student/data/genimagepp/sdxl/sdxl_style/val/1_fake"

mkdir -p run_logs_full

run_one() {
  script="$1"; ckpt="$2"; tag="$3"; name="$4"; real="$5"; fake="$6"; max_sample="$7"
  echo "=================================================="
  echo "$tag: $name (max_sample=$max_sample)"
  echo "=================================================="
  python3 "$script" \
    --real_path "$real" --fake_path "$fake" \
    --ckpt "$ckpt" \
    --out "predictions_${tag}_${name}_full.csv" \
    --batch_size="$BATCH_SIZE" \
    --max_sample="$max_sample" \
    2>&1 | tee "run_logs_full/${tag}_${name}.log"
}

BASELINE_CKPT="attack_code/weights/clip_epoch_20.pth"
LORA_CKPT="attack_code/weights/best_model_low_rank.pt"

for cfg in \
  "run_clip_baseline.py|$BASELINE_CKPT|clipbaseline" \
  "run_clip_lora.py|$LORA_CKPT|cliplora" \
; do
  IFS='|' read -r script ckpt tag <<< "$cfg"
  run_one "$script" "$ckpt" "$tag" BigGan "$BIGGAN_REAL" "$BIGGAN_FAKE" 6000
  run_one "$script" "$ckpt" "$tag" Glide  "$GLIDE_REAL"  "$GLIDE_FAKE"  6000
  run_one "$script" "$ckpt" "$tag" flux   "$BIGGAN_TRAIN_REAL" "$FLUX_FAKE" 6000
  run_one "$script" "$ckpt" "$tag" sd3    "$BIGGAN_TRAIN_REAL" "$SD3_FAKE"  6000
  run_one "$script" "$ckpt" "$tag" sdxl   "$BIGGAN_TRAIN_REAL" "$SDXL_FAKE" 18301
done

echo "=================================================="
echo "ALL GENIMAGEPP DETECTOR FULL-DATA RUNS COMPLETE"
echo "=================================================="
