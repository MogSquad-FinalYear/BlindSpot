"""
Serving-side orchestrator -- the drop-in replacement for rule_engine's router.

`rule_engine.select_detector` answers "which ONE detector should run?" from a
hand-tuned rule. `orchestrate()` answers "what should we actually do with this
image?" from a model trained on measured per-detector outcomes, and returns
the final score directly instead of delegating it to whichever detector the
rule happened to pick.

Three strategies, all served from one artifact (see train_orchestrator.py):

  cascade    run the primary detector; escalate to the rest only when its
             logit lands inside the trained uncertainty band. Default: keeps
             most of fusion's accuracy at close to one detector's cost.
  fusion     run every detector, meta-classify their logits + features.
             Most accurate, N detector calls.
  selection  features alone choose one detector; that detector's score is the
             answer. Same cost as today's router, and the honest
             apples-to-apples comparison against it.

The app calls this instead of compute_features/select_detector:

    from orchestrator.serve import orchestrate
    result = orchestrate(image, device)
    # -> {"score", "flagged", "strategy", "detectors_run", "reason", "features"}
"""
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import features as featmod                                  # noqa: E402
import detectors_bulk                                       # noqa: E402

MODEL_PATH = Path(os.environ.get(
    "ORCHESTRATOR_MODEL", _HERE / "data" / "orchestrator_model.pkl"))
STRATEGY = os.environ.get("ORCHESTRATOR_STRATEGY", "cascade")
FLAG_THRESHOLD = float(os.environ.get("ORCHESTRATOR_FLAG_THRESHOLD", "0.5"))

# How many detector models may stay resident on the GPU. The five together are
# ~3.8 GB (three CLIP ViT-L/14 towers at ~1.2 GB each, plus two ResNet-50s),
# which fits an idle 8 GB card but not one shared with anything else. 0 means
# unlimited; a small number trades reload latency for headroom. Eviction is
# least-recently-used, and an OOM triggers an evict-and-retry regardless.
MAX_RESIDENT = int(os.environ.get("ORCHESTRATOR_MAX_RESIDENT", "0"))

EPS = 1e-6
_artifact = None
_models = OrderedDict()


def _logit(p):
    p = float(np.clip(p, EPS, 1 - EPS))
    return float(np.log(p / (1 - p)))


def load_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"no orchestrator model at {MODEL_PATH}. Train one first:\n"
                f"  cd {_HERE} && python3 train_orchestrator.py")
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


_MEM_MARKERS = ("out of memory", "nvml", "cuda error", "no kernel image")


def _is_memory_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return any(m in str(exc).lower() for m in _MEM_MARKERS)


def _evict_lru():
    """Drop the least recently used detector and return its memory."""
    import gc
    if not _models:
        return False
    name, _ = _models.popitem(last=False)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return name


def _detector(name, device):
    """Load one detector, keeping it resident for subsequent requests.

    A web worker handles one image at a time, so keeping towers warm is the
    right trade against reloading 1.2 GB per request -- but unbounded caching
    is how an 8 GB card dies under `fusion`. MAX_RESIDENT bounds it, and an
    OOM during load evicts and retries rather than failing the request.
    """
    if name in _models:
        _models.move_to_end(name)
        return _models[name]
    while MAX_RESIDENT and len(_models) >= MAX_RESIDENT:
        _evict_lru()
    while True:
        try:
            _models[name] = detectors_bulk.load(name, device)
            return _models[name]
        except Exception as exc:                            # noqa: BLE001
            # Memory pressure does not always surface as OutOfMemoryError on
            # this host: nvidia-smi reports a driver/library version mismatch,
            # so the caching allocator's NVML query fails first and raises
            # "NVML_SUCCESS == ... nvmlInit_v2_ INTERNAL ASSERT FAILED"
            # instead. Both mean the same thing here -- free a model and retry.
            if not _is_memory_error(exc) or not _evict_lru():
                raise


def _score_one(name, image, device):
    model, tf = _detector(name, device)
    with torch.no_grad():
        x = tf(image.convert("RGB")).unsqueeze(0).to(device)
        return float(torch.sigmoid(model(x)).flatten().item())


def orchestrate(image, device, strategy=None):
    art = load_artifact()
    strategy = strategy or STRATEGY
    dets = art["detectors"]

    feats = featmod.compute_from_image(image)
    fvec = np.array([[float(feats[c]) for c in art["feat_cols"]]])

    scores, ran = {}, []

    def run(name):
        if name not in scores:
            scores[name] = _score_one(name, image, device)
            ran.append(name)
        return scores[name]

    if strategy == "selection":
        if art["selection"] is None:
            raise RuntimeError("artifact has no selection head")
        pick = dets[int(art["selection"].predict(fvec)[0])]
        s = run(pick)
        reason = (f"Orchestrator (selection) chose {pick} from image features "
                  f"alone; its score stands as the verdict. One detector call.")
        return _result(s, strategy, ran, reason, feats)

    if strategy == "cascade":
        primary = art["cascade_primary"]
        band = art["cascade_band"]
        s = run(primary)
        lg = _logit(s)
        if band is None or abs(lg) >= band:
            reason = (f"Orchestrator (cascade): {primary} returned {s:.4f} "
                      f"(logit {lg:+.2f}, outside the +-{band} uncertainty band), "
                      f"so no escalation was needed. One detector call.")
            return _result(s, strategy, ran, reason, feats)
        for d in dets:
            run(d)
        p = float(art["fusion"].predict_proba(_stack(scores, dets, fvec))[0, 1])
        reason = (f"Orchestrator (cascade): {primary} was uncertain "
                  f"({s:.4f}, logit {lg:+.2f}, inside +-{band}), so all "
                  f"{len(dets)} detectors ran and the fusion head decided: {p:.4f}.")
        return _result(p, strategy, ran, reason, feats)

    if strategy == "fusion":
        for d in dets:
            run(d)
        p = float(art["fusion"].predict_proba(_stack(scores, dets, fvec))[0, 1])
        detail = ", ".join(f"{d}={scores[d]:.3f}" for d in dets)
        reason = (f"Orchestrator (fusion) combined all {len(dets)} detectors "
                  f"({detail}) with image features into {p:.4f}.")
        return _result(p, strategy, ran, reason, feats)

    raise ValueError(f"unknown strategy {strategy!r}")


def _stack(scores, dets, fvec):
    s = np.array([[_logit(scores[d]) for d in dets]])
    return np.hstack([s, fvec])


def _result(score, strategy, ran, reason, feats):
    return {
        "score": round(float(score), 4),
        "flagged": bool(score > FLAG_THRESHOLD),
        "strategy": strategy,
        "detectors_run": list(ran),
        "n_detector_calls": len(ran),
        "reason": reason,
        "features": feats,
    }
