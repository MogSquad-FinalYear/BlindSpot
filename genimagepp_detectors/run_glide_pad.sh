#!/usr/bin/env bash
set -e
cd /home/student/data/genimagepp_detectors
source /home/student/qwen_env/bin/activate
GR="/home/student/Downloads/imagenet_glide/val/nature"
GF="/home/student/Downloads/imagenet_glide/val/ai"
python3 run_clip_baseline.py --real_path "$GR" --fake_path "$GF" \
  --ckpt attack_code/weights/clip_epoch_20.pth \
  --out predictions_clipbaseline_Glide_pad.csv --batch_size 16 --preprocess pad \
  > run_logs_full/clipbaseline_Glide_pad.log 2>&1
echo "CLIPBASE_GLIDE_PAD_DONE"
python3 run_clip_lora.py --real_path "$GR" --fake_path "$GF" \
  --ckpt attack_code/weights/best_model_low_rank.pt \
  --out predictions_cliplora_Glide_pad.csv --batch_size 16 --preprocess pad \
  > run_logs_full/cliplora_Glide_pad.log 2>&1
echo "ALL_GLIDE_PAD_DONE"
