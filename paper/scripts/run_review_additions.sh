#!/usr/bin/env bash
# Remaining GPU work for the review response, run sequentially to avoid
# contending for the single shared 8GB GPU.
set -e
source /home/student/qwen_env/bin/activate

BG_R="/home/student/Downloads/imagenet_ai_0419_biggan/val/nature"
BG_F="/home/student/Downloads/imagenet_ai_0419_biggan/val/ai"
GL_R="/home/student/Downloads/imagenet_glide/val/nature"
GL_F="/home/student/Downloads/imagenet_glide/val/ai"
SHARED_R="/home/student/Downloads/imagenet_ai_0419_biggan/train/nature"
FLUX_F="/home/student/data/genimagepp/flux/flux/val/1_fake"
SD3_F="/home/student/data/genimagepp/sd3/stable_diffusion_v_3_0/val/1_fake"
SDXL_F="/home/student/data/genimagepp/sdxl/sdxl_style/val/1_fake"

# wait for the GLIDE padding chain launched earlier
until grep -q "ALL_GLIDE_PAD_DONE" /home/student/data/genimagepp_detectors/glide_pad_chain.out 2>/dev/null; do sleep 20; done
echo "=== STAGE 1 COMPLETE (GLIDE padding) ==="

cd /home/student/data/genimagepp_detectors
mkdir -p run_logs_full

# ---- STAGE 2: ResNet-50 fifth detector, full sweep ----
run_rn () {
  name="$1"; real="$2"; fake="$3"; cap="$4"
  python3 run_resnet_baseline.py --real_path "$real" --fake_path "$fake" \
    --ckpt attack_code/weights/resnet_epoch_20.pth \
    --out "predictions_resnet_${name}_full.csv" \
    --batch_size 32 --normalize imagenet --max_sample "$cap" \
    > "run_logs_full/resnet_${name}.log" 2>&1
  echo "  resnet $name done"
}
run_rn BigGan "$BG_R" "$BG_F" 6000
run_rn Glide  "$GL_R" "$GL_F" 6000
run_rn flux   "$SHARED_R" "$FLUX_F" 6000
run_rn sd3    "$SHARED_R" "$SD3_F"  6000
run_rn sdxl   "$SHARED_R" "$SDXL_F" 18301
echo "=== STAGE 2 COMPLETE (ResNet-50 sweep) ==="

# ---- STAGE 3: JPEG re-encoding robustness on FLUX, paired clean vs q=75 ----
N=2000
for Q in clean 75; do
  if [ "$Q" = "clean" ]; then JFLAG=""; SUF="clean"; else JFLAG="--jpeg_quality 75"; SUF="jpeg75"; fi

  python3 run_clip_baseline.py --real_path "$SHARED_R" --fake_path "$FLUX_F" \
    --ckpt attack_code/weights/clip_epoch_20.pth \
    --out "predictions_clipbaseline_flux_${SUF}${N}.csv" --batch_size 16 \
    --max_sample $N $JFLAG > "run_logs_full/jpeg_clipbaseline_${SUF}.log" 2>&1
  echo "  clipbaseline $SUF done"

  python3 run_clip_lora.py --real_path "$SHARED_R" --fake_path "$FLUX_F" \
    --ckpt attack_code/weights/best_model_low_rank.pt \
    --out "predictions_cliplora_flux_${SUF}${N}.csv" --batch_size 16 \
    --max_sample $N $JFLAG > "run_logs_full/jpeg_cliplora_${SUF}.log" 2>&1
  echo "  cliplora $SUF done"

  python3 run_resnet_baseline.py --real_path "$SHARED_R" --fake_path "$FLUX_F" \
    --ckpt attack_code/weights/resnet_epoch_20.pth \
    --out "predictions_resnet_flux_${SUF}${N}.csv" --batch_size 32 \
    --normalize imagenet --max_sample $N $JFLAG > "run_logs_full/jpeg_resnet_${SUF}.log" 2>&1
  echo "  resnet $SUF done"

  cd /home/student/data/NPR-DeepfakeDetection
  python3 run_inference.py --real_path "$SHARED_R" --fake_path "$FLUX_F" \
    --out "predictions_flux_${SUF}${N}.csv" --batch_size 16 --max_sample $N $JFLAG \
    > "run_logs_full/jpeg_${SUF}.log" 2>&1
  echo "  npr $SUF done"

  cd /home/student/data/UniversalFakeDetect
  python3 run_and_dump.py --real_path "$SHARED_R" --fake_path "$FLUX_F" \
    --result_folder "clip_vitl14_flux_${SUF}${N}" \
    --out "predictions_flux_${SUF}${N}.csv" --batch_size 16 --max_sample $N $JFLAG \
    > "run_logs_full/jpeg_${SUF}.log" 2>&1
  echo "  ufd $SUF done"
  cd /home/student/data/genimagepp_detectors
done
echo "=== STAGE 3 COMPLETE (JPEG robustness) ==="
echo "ALL_REVIEW_ADDITIONS_DONE"
