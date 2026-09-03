"""
BdSL Sign Language Recognition API
====================================
Supports 8 model variants:
  Transformer  : transformer_frontview | transformer_interpolated_frontview
                 transformer_multiview | transformer_interpolated_multiview
  CNN-BiLSTM   : cnn_frontview         | cnn_interpolated_frontview
                 cnn_multiview         | cnn_interpolated_multiview

Usage:
  POST /predict-video?model_type=transformer_frontview
       (default model_type = "transformer_frontview")

  GET  /models   — list all available model keys
"""

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import json
import cv2
import os
import uuid
import mediapipe as mp

from transformer import TransformerSignModel
from cnn_bilstm import CNNBiLSTMAttention
from rqe import apply_rqe

# ============================================================
# App setup
# ============================================================
app = FastAPI(
    title="BdSL Sign Language API",
    description="Upload a sign-language video and get the predicted word (Bangla/English).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Dynamic Path & File Resolver (Hugging Face & Local Compatible)
# ============================================================
def _find_path(filename: str, extra_subdirs: list[str] = None) -> str:
    """Find a file across multiple search locations (root, current dir, subdirectories)."""
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    ]
    if extra_subdirs:
        for sub in extra_subdirs:
            search_dirs.append(os.path.join(os.getcwd(), sub))
            search_dirs.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), sub))
            search_dirs.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", sub))

    for d in search_dirs:
        p = os.path.normpath(os.path.join(d, filename))
        if os.path.exists(p):
            return p
    return filename

LABEL_PATH = _find_path("labels.json")
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"labels.json not found in search paths.")

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(LABEL_PATH, "r", encoding="utf-8") as _f:
    labels = json.load(_f)

NUM_CLASSES = len(labels)
INPUT_DIM = 387
MAX_FRAMES = 60

# ============================================================
# Model registry
# ============================================================
# Each entry: (architecture_class, checkpoint_filename)
MODEL_REGISTRY: dict[str, tuple] = {
    # ---- Transformer ----
    "transformer_frontview":              (TransformerSignModel, "transformer_frontveiw.pth"),
    "transformer_interpolated_frontview": (TransformerSignModel, "transformer_interpolated_frontveiw.pth"),
    "transformer_multiview":              (TransformerSignModel, "transformer_multiview.pth"),
    "transformer_interpolated_multiview": (TransformerSignModel, "transformer_interpolated_multiview.pth"),
    # ---- CNN-BiLSTM ----
    "cnn_frontview":                      (CNNBiLSTMAttention, "cnn_bilstm_attention_frontveiw.pth"),
    "cnn_interpolated_frontview":         (CNNBiLSTMAttention, "cnn_bilstm_attention_interpolated_frontveiw.pth"),
    "cnn_multiview":                      (CNNBiLSTMAttention, "cnn_bilstm_attention_multiview.pth"),
    "cnn_interpolated_multiview":         (CNNBiLSTMAttention, "cnn_bilstm_attention_interpolated_multiview.pth"),
}

DEFAULT_MODEL = "transformer_interpolated_frontview"

# Lazy-loaded model cache: {model_type: nn.Module}
_model_cache: dict[str, torch.nn.Module] = {}


def _load_model(model_type: str) -> torch.nn.Module:
    """Load and cache a model by its registry key. Searches root and subdirectories."""
    if model_type in _model_cache:
        return _model_cache[model_type]

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")

    arch_cls, ckpt_name = MODEL_REGISTRY[model_type]
    ckpt_path = _find_path(ckpt_name, extra_subdirs=["Models", "../Models"])

    m = arch_cls(INPUT_DIM, NUM_CLASSES).to(device)
    if os.path.exists(ckpt_path):
        try:
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[API] Loaded '{model_type}' from {ckpt_path}")
        except Exception as exc:
            print(f"[API] WARNING: Could not load weights for '{model_type}': {exc}")
    else:
        print(f"[API] WARNING: Checkpoint '{ckpt_name}' not found.")

    m.eval()
    _model_cache[model_type] = m
    return m


# Pre-load the default model at startup so the first request is fast
try:
    _load_model(DEFAULT_MODEL)
except Exception as _e:
    print(f"[API] Could not pre-load default model: {_e}")


# ============================================================
# MediaPipe config
# ============================================================
mp_holistic = mp.solutions.holistic

IMPORTANT_FACE_IDX = [
    # Eyes, eyebrows, nose, mouth
    33, 133, 159, 145, 468, 469, 263, 362, 386, 374, 471, 472,
    105, 107, 55, 65, 52, 285, 295, 282, 283, 336,
    1, 2, 98, 327, 94, 97, 168, 197,
    13, 14, 78, 308, 82, 312, 87, 317, 88, 95, 178, 191,
    80, 81, 82, 311, 310, 415, 291, 308, 324, 318, 402, 317,
]


# ============================================================
# Utility functions
# ============================================================
def interpolate_sequence(seq: list, target_len: int = MAX_FRAMES) -> np.ndarray:
    T = len(seq)
    if T == 0:
        return np.zeros((target_len, INPUT_DIM))
    if T == target_len:
        return np.array(seq)

    seq = np.array(seq)
    D = seq.shape[1]
    orig_grid = np.linspace(0, 1, T)
    target_grid = np.linspace(0, 1, target_len)

    new_seq = np.zeros((target_len, D))
    for d in range(D):
        new_seq[:, d] = np.interp(target_grid, orig_grid, seq[:, d])
    return new_seq


def _fill_missing_landmarks(seq: list[list[float]]) -> list[list[float]]:
    """Smooth missing (0,0,0) landmark gaps across time using linear interpolation."""
    if not seq:
        return seq
    
    arr = np.array(seq, dtype=np.float32)  # shape (T, 387)
    T, D = arr.shape
    N_LM = D // 3  # 129
    
    lm_3d = arr.reshape(T, N_LM, 3)
    
    for p in range(N_LM):
        active_mask = np.any(lm_3d[:, p] != 0, axis=1)
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) > 0 and len(active_indices) < T:
            first_act = active_indices[0]
            last_act = active_indices[-1]
            if last_act > first_act:
                for c in range(3):
                    interp_vals = np.interp(
                        np.arange(first_act, last_act + 1),
                        active_indices,
                        lm_3d[active_indices, p, c]
                    )
                    lm_3d[first_act:last_act + 1, p, c] = interp_vals
                    
    return lm_3d.reshape(T, D).tolist()


def extract_landmarks_from_video(video_path: str, max_frames: int = MAX_FRAMES):
    """Robust landmark extraction from video matching the training pipeline exactly."""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

    seq = []

    with mp_holistic.Holistic(model_complexity=1, refine_face_landmarks=True) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            lm = []

            # Face (54 points)
            if results.face_landmarks:
                face = results.face_landmarks.landmark
                for idx in IMPORTANT_FACE_IDX:
                    p = face[idx]
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * len(IMPORTANT_FACE_IDX) * 3

            # Left hand (21 points)
            if results.left_hand_landmarks:
                for p in results.left_hand_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * 21 * 3

            # Right hand (21 points)
            if results.right_hand_landmarks:
                for p in results.right_hand_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * 21 * 3

            # Pose (33 points)
            if results.pose_landmarks:
                for p in results.pose_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * 33 * 3

            seq.append(lm)

    cap.release()

    if not seq:
        empty = np.zeros((max_frames, INPUT_DIM), dtype=np.float32)
        return empty, empty.tolist()

    # 1. Fill missing landmark gaps across frames (motion blur recovery)
    filled_seq = _fill_missing_landmarks(seq)

    # 2. Interpolate uniformly to 60 frames matching the trained model input
    interpolated = interpolate_sequence(filled_seq, max_frames)
    landmark_frames = interpolated.tolist()
    return interpolated, landmark_frames


def _compute_focus_and_cam(logits: torch.Tensor, x: torch.Tensor):
    """Derive per-landmark spatial focus and per-frame temporal CAM from softmax.

    Strategy (lightweight, no hooks needed):
    - spatial focus: std-dev of each landmark's contribution across the feature dim
      (proxy for salience — higher variance = more discriminative)
    - cam: temporal saliency estimated from input magnitude per frame
    """
    T = x.shape[1]        # frames
    D = x.shape[2]        # features per frame
    N_LM = D // 3         # number of landmark points

    # Temporal CAM: L2 norm of each frame, normalised to [0,1]
    frame_norms = x[0].norm(dim=1).cpu().numpy()   # shape (T,)
    cam_max = frame_norms.max()
    cam = (frame_norms / cam_max).tolist() if cam_max > 0 else [0.0] * T

    # Spatial focus: std-dev of (x,y,z) triplet for each landmark, normalised
    x_np = x[0].cpu().numpy()  # (T, D)
    reshaped = x_np.reshape(T, N_LM, 3)            # (T, N_LM, 3)
    lm_std = reshaped.std(axis=2).mean(axis=0)      # (N_LM,) — avg std across frames
    std_max = lm_std.max()
    focus_per_lm = (lm_std / std_max).tolist() if std_max > 0 else [0.0] * N_LM

    # Expand focus to per-frame: same focus map repeated for each frame
    focus_frames = [focus_per_lm for _ in range(T)]

    return focus_frames, cam


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def root():
    """Root endpoint for status check."""
    return {
        "status": "online",
        "message": "BdSL Sign Language Recognition API is running",
        "models": list(MODEL_REGISTRY.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": device,
        "classes": NUM_CLASSES,
        "loaded_models": list(_model_cache.keys()),
    }


@app.get("/models")
def list_models():
    """Return all supported model keys."""
    return {
        "available_models": list(MODEL_REGISTRY.keys()),
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_model_cache.keys()),
    }


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    model_type: str = Query(
        default=DEFAULT_MODEL,
        description=(
            "Which model to use for inference. "
            f"Options: {', '.join(MODEL_REGISTRY.keys())}. "
            f"Default: {DEFAULT_MODEL}"
        ),
    ),
):
    """
    Upload an MP4 sign-language video and get the predicted word.

    - **file**: MP4 video file
    - **model_type**: one of the 8 model keys (see GET /models)
    """
    if model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}",
        )

    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        with open(temp_path, "wb") as fout:
            fout.write(await file.read())

        # Landmark extraction + RQE normalisation
        raw_seq, landmark_frames = extract_landmarks_from_video(temp_path)
        norm_seq = apply_rqe(raw_seq, quantization_step=0.05, shoulder_fixing=True)

        # Inference
        x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
        m = _load_model(model_type)

        with torch.no_grad():
            logits = m(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = float(probs[0, pred_idx].item()) * 100  # percentage

        label = labels[str(pred_idx)]

        # Compute visualisation data (focus map + temporal CAM)
        focus_frames, cam = _compute_focus_and_cam(logits, x)

        return {
            "model_used": model_type,
            "class_index": pred_idx,
            "word_id": label["id"],
            "bangla": label["bangla"],
            "english": label["english"],
            "confidence": f"{confidence:.1f}%",
            "landmarks": landmark_frames,
            "focus_points": focus_frames,
            "cam": cam,
        }

    except HTTPException:
        raise
    except Exception as exc:
        return {"error": str(exc)}

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"[API] Warning: Failed to delete temp file {temp_path}: {cleanup_err}")
