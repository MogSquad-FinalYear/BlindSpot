#!/usr/bin/env bash
# Disentangles "no smooth interpolation" from "large zero-padding border" as
# the cause of the BigGAN preprocessing effect, by adding a NEAREST-upsample
# condition (no interpolation, no border) alongside the existing
# upsample (interpolation, no border) and pad (no interpolation, 67% border).
set -e
cd /home/student/data/genimagepp_detectors
source /home/student/qwen_env/bin/activate
BG_R="/home/student/Downloads/imagenet_ai_0419_biggan/val/nature"
BG_F="/home/student/Downloads/imagenet_ai_0419_biggan/val/ai"
N=2000

until grep -q "ALL_REVIEW_ADDITIONS_DONE" /home/student/data/paper/scripts/review_additions.out 2>/dev/null; do sleep 30; done
echo "prior stages complete, starting disentangling runs"

python3 run_clip_lora.py --real_path "$BG_R" --fake_path "$BG_F" \
  --ckpt attack_code/weights/best_model_low_rank.pt \
  --out predictions_cliplora_BigGan_nearest${N}.csv --batch_size 16 \
  --preprocess nearest --max_sample $N > run_logs_full/cliplora_BigGan_nearest.log 2>&1
echo "  cliplora nearest done"

python3 run_clip_baseline.py --real_path "$BG_R" --fake_path "$BG_F" \
  --ckpt attack_code/weights/clip_epoch_20.pth \
  --out predictions_clipbaseline_BigGan_nearest${N}.csv --batch_size 16 \
  --preprocess nearest --max_sample $N > run_logs_full/clipbaseline_BigGan_nearest.log 2>&1
echo "  clipbaseline nearest done"

# matched-N controls so the comparison is like-for-like at N per class
python3 run_clip_lora.py --real_path "$BG_R" --fake_path "$BG_F" \
  --ckpt attack_code/weights/best_model_low_rank.pt \
  --out predictions_cliplora_BigGan_upsample${N}.csv --batch_size 16 \
  --preprocess upsample --max_sample $N > run_logs_full/cliplora_BigGan_up${N}.log 2>&1
python3 run_clip_lora.py --real_path "$BG_R" --fake_path "$BG_F" \
  --ckpt attack_code/weights/best_model_low_rank.pt \
  --out predictions_cliplora_BigGan_pad${N}.csv --batch_size 16 \
  --preprocess pad --max_sample $N > run_logs_full/cliplora_BigGan_pad${N}.log 2>&1
echo "  cliplora matched controls done"
echo "DISENTANGLE_DONE"
