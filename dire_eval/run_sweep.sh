#!/usr/bin/env bash
# DIRE across one generator per architecture family, matched to the real
# pools the rest of the study uses.  N is set from the measured throughput
# so the sweep finishes in the revision window; DIRE costs roughly 40
# diffusion forward passes per image, which is why it is a subset and not
# the full 6,000 per class the other five detectors run on.
set -e
cd /home/student/data/dire_eval
source /home/student/qwen_env/bin/activate

N="${1:-500}"
BS="${2:-4}"

BIGGAN_REAL="/home/student/Downloads/imagenet_ai_0419_biggan/val/nature"
BIGGAN_FAKE="/home/student/Downloads/imagenet_ai_0419_biggan/val/ai"
SHARED_REAL="/home/student/Downloads/imagenet_ai_0419_biggan/train/nature"
FLUX_FAKE="/home/student/data/genimagepp/flux/flux/val/1_fake"
SDXL_FAKE="/home/student/data/genimagepp/sdxl/sdxl_style/val/1_fake"

mkdir -p results logs

run () {
  name="$1"; real="$2"; fake="$3"
  echo "=== DIRE: $name (N=$N per class) ==="
  python3 run_dire.py \
    --real_path "$real" --fake_path "$fake" \
    --diffusion_ckpt weights/lsun_bedroom.pt \
    --classifier_ckpt weights/lsun_adm.pth \
    --out "results/dire_${name}.csv" \
    --max_sample "$N" --batch_size "$BS" \
    2>&1 | tee "logs/${name}.log"
  echo "  $name done"
}

# one generator per architecture family: GAN, Diffusion Transformer, U-Net
run flux   "$SHARED_REAL" "$FLUX_FAKE"
run BigGan "$BIGGAN_REAL" "$BIGGAN_FAKE"
run sdxl   "$SHARED_REAL" "$SDXL_FAKE"

echo "ALL_DIRE_RUNS_DONE"
