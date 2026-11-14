import os
import sys
import time

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from realitydefender import RealityDefender, RealityDefenderError
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
from config import (
    DEVICE,
    REALITY_DEFENDER_API_KEY,
    REALITY_DEFENDER_FLAG_THRESHOLD,
    UPLOAD_DIR,
)
from detectors import genimagepp_cliplora, npr, universalfakedetect
from rule_engine import compute_features, select_detector

app = Flask(__name__)
CORS(app)

DETECTOR_MODULES = {
    "npr": npr,
    "universalfakedetect": universalfakedetect,
    "genimagepp_cliplora": genimagepp_cliplora,
}

DETECTOR_DISPLAY_NAMES = {
    "npr": "NPR (Neighboring Pixel Relationships)",
    "universalfakedetect": "UniversalFakeDetect (CLIP ViT-L/14)",
    "genimagepp_cliplora": "GenImage++ CLIP-LoRA",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# None when no key is configured -- _process_image() then just skips the
# Reality Defender call and keeps the local detector's own score/flagged.
reality_defender = (
    RealityDefender(api_key=REALITY_DEFENDER_API_KEY)
    if REALITY_DEFENDER_API_KEY
    else None
)


def _is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _unique_path(directory, filename):
    base, ext = os.path.splitext(filename)
    candidate = filename
    n = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}_{n}{ext}"
        n += 1
    return candidate


def _process_image(file_storage, folder: str | None) -> dict:
    safe_name = secure_filename(file_storage.filename) or "upload"
    filename = _unique_path(
        os.path.join(UPLOAD_DIR, folder) if folder else UPLOAD_DIR,
        safe_name,
    )

    target_dir = UPLOAD_DIR / folder if folder else UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    file_storage.save(target_path)

    image = Image.open(target_path).convert("RGB")
    features = compute_features(image)
    choice = select_detector(features)
    detector_key = choice["detector"]

    t0 = time.time()
    score = DETECTOR_MODULES[detector_key].predict(image, device)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    flagged = score > 0.2

    record = {
        "filename": filename,
        "folder": folder,
        "flagged": flagged,
        "score": round(score, 4),
        "detector": detector_key,
        "detector_display_name": DETECTOR_DISPLAY_NAMES[detector_key],
        "reason": choice["reason"],
        "features": features,
        "inference_ms": elapsed_ms,
    }

    # Reality Defender second opinion: on success, its score/status replace
    # this record's score/flagged outright (see config.py for why 0.5).
    # local_score/local_flagged are kept alongside so the original
    # detector's call is never actually lost, just no longer the one
    # `flagged`/`score` report. Runs before db.save_detection so what's
    # persisted (and what the gallery shows later) matches what this
    # request returns -- not just this one response.
    if reality_defender is not None:
        try:
            rd_result = reality_defender.detect_file(str(target_path))
            rd_score = rd_result.get("score")
            if rd_score is not None:
                record["local_score"] = record["score"]
                record["local_flagged"] = record["flagged"]
                record["score"] = round(float(rd_score), 4)
                record["flagged"] = rd_score > REALITY_DEFENDER_FLAG_THRESHOLD
                record["reality_defender"] = {
                    "status": rd_result.get("status"),
                    "models": rd_result.get("models", []),
                }
        except RealityDefenderError as exc:
            record["reality_defender_error"] = f"{exc.message} ({exc.code})"
        except Exception as exc:  # noqa: BLE001 -- a third-party API outage
            # must not take down local-detector uploads that were working fine
            record["reality_defender_error"] = str(exc)

    return db.save_detection(record)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "device": str(device)})


@app.route("/api/detect", methods=["POST"])
def detect():
    files = request.files.getlist("images")
    raw_folder = request.form.get("folder") or None
    folder = secure_filename(raw_folder) if raw_folder else None

    if not files:
        return jsonify({"error": "no images provided"}), 400

    results = []
    for file_storage in files:
        if not file_storage or not file_storage.filename:
            continue
        if not _is_image(file_storage.filename):
            continue
        try:
            record = _process_image(file_storage, folder)
            results.append(record)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "filename": file_storage.filename,
                "folder": folder,
                "error": str(exc),
            })

    return jsonify({"results": results})


@app.route("/api/detections", methods=["GET"])
def detections():
    folder = request.args.get("folder") or None
    return jsonify({"detections": db.list_detections(folder)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8010))
    app.run(host="0.0.0.0", port=port, debug=False)
