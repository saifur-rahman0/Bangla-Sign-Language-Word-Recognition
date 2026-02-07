import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
import uvicorn
import mediapipe as mp

from model import CNN_BiLSTM_Attention

# ---------------- CONFIG ----------------
app = FastAPI()
mp_holistic = mp.solutions.holistic

MODEL_PATH = "cnn_bilstm_attention.pth"
LABEL_PATH = "labels.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

NUM_CLASSES = len(labels)
INPUT_DIM = 387
MAX_FRAMES = 60

model = CNN_BiLSTM_Attention(INPUT_DIM, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

IMPORTANT_FACE_IDX = [
    33, 133, 159, 145, 468, 469, 263, 362, 386, 374, 471, 472,
    105, 107, 55, 65, 52, 285, 295, 282, 283, 336,
    1, 2, 98, 327, 94, 97, 168, 197,
    13, 14, 78, 308, 82, 312, 87, 317, 88, 95, 178, 191,
    80, 81, 82, 311, 310, 415, 291, 308, 324, 318, 402, 317
]

# ---------------- Utils ----------------
def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            buffer.write(upload_file.file.read())
    finally:
        upload_file.file.close()

def extract_landmarks_from_video(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    seq = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            lm = []

            # Face
            if results.face_landmarks:
                for idx in IMPORTANT_FACE_IDX:
                    p = results.face_landmarks.landmark[idx]
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * len(IMPORTANT_FACE_IDX) * 3

            # Left Hand
            if results.left_hand_landmarks:
                for p in results.left_hand_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * 21 * 3

            # Right Hand
            if results.right_hand_landmarks:
                for p in results.right_hand_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * 21 * 3

            # Pose
            if results.pose_landmarks:
                for p in results.pose_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * 33 * 3

            seq.append(lm)
            if len(seq) >= max_frames:
                break

    cap.release()
    actual_length = len(seq)

    if len(seq) < max_frames:
        if len(seq) == 0:
            seq = [[0.0] * INPUT_DIM] * max_frames
        else:
            padding = [[0.0] * len(seq[0])] * (max_frames - len(seq))
            seq += padding

    return np.array(seq, dtype=np.float32), actual_length

def normalize_sequence(seq):
    FACE_LM, HAND_LM, POSE_LM, DIM = len(IMPORTANT_FACE_IDX), 21, 33, 3
    LEFT_HAND_START = FACE_LM * DIM
    RIGHT_HAND_START = LEFT_HAND_START + HAND_LM * DIM
    POSE_START = RIGHT_HAND_START + HAND_LM * DIM

    def get_landmark(arr, start_idx, lm_index):
        return arr[start_idx + lm_index * 3: start_idx + lm_index * 3 + 3]

    norm_seq = []
    for frame in seq:
        pose0 = get_landmark(frame, POSE_START, 0)
        center = pose0 if np.any(pose0 != 0) else np.mean(frame.reshape(-1, 3), axis=0)

        ls = get_landmark(frame, POSE_START, 11)
        rs = get_landmark(frame, POSE_START, 12)
        scale = np.linalg.norm(ls - rs) if np.any(ls != 0) and np.any(rs != 0) else 1.0
        if scale == 0:
            scale = 1.0

        normalized = (frame.reshape(-1, 3) - center) / (scale + 1e-6)
        norm_seq.append(normalized.flatten())

    return np.array(norm_seq, dtype=np.float32)

# ---------------- Explainability ----------------
class TemporalGradCAM:
    """
    Temporal Grad-CAM for Conv1d layers.
    Produces importance over reduced time T', then we upsample to 60 frames.
    """
    def __init__(self, model, target_conv):
        self.model = model
        self.target_conv = target_conv
        self.activations = None
        self.gradients = None
        self.h1 = target_conv.register_forward_hook(self._save_activation)
        self.h2 = target_conv.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out  # (B,C,T')

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # (B,C,T')

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred].item())

        if class_idx is None:
            class_idx = pred

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.activations[0].detach()  # (C,T')
        dA = self.gradients[0].detach()   # (C,T')

        w = dA.mean(dim=1)  # (C,)
        cam = torch.relu((w[:, None] * A).sum(dim=0))  # (T',)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), pred, conf

def upsample_cam(cam_tprime, target_len=MAX_FRAMES):
    cam = torch.tensor(cam_tprime)[None, None, :]  # (1,1,T')
    cam_up = F.interpolate(cam, size=target_len, mode="linear", align_corners=False)
    return np.clip(cam_up[0, 0].cpu().numpy(), 0, 1)

def landmark_saliency(model, x, class_idx=None):
    """
    Gradient-based saliency on input landmarks.
    Returns saliency (T,F) normalized to [0,1].
    """
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    probs = F.softmax(logits, dim=1)

    pred = int(torch.argmax(probs, dim=1).item())
    conf = float(probs[0, pred].item())

    if class_idx is None:
        class_idx = pred

    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    sal = x.grad.abs()[0]  # (T,F)
    sal = sal / (sal.max() + 1e-8)
    return sal.detach().cpu().numpy(), pred, conf

def feature_saliency_to_point_scores(frame_sal):
    """
    frame_sal: (F,)
    Landmarks are packed as [x,y,z,x,y,z,...]
    -> reshape to (num_points, 3) and mean over xyz -> (num_points,)
    """
    pts = frame_sal.reshape(-1, 3).mean(axis=1)
    return pts.tolist()

# ---------------- Routes ----------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "BdSL API is running",
        "device": device,
        "supported_classes": NUM_CLASSES
    }

@app.post("/prediction-and-landmarks")
async def prediction_and_landmarks(file: UploadFile = File(...)):
    temp_path = f"temp_full_{file.filename}"
    try:
        save_upload_file(file, temp_path)

        raw_seq, actual_frames = extract_landmarks_from_video(temp_path)
        norm_seq = normalize_sequence(raw_seq)

        x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx_tensor = torch.max(probs, dim=1)
            pred_idx = int(pred_idx_tensor.item())
            confidence_percentage = float(conf.item() * 100)

        label = labels[str(pred_idx)]
        return {
            "prediction": {
                "word_id": label["id"],
                "bangla": label["bangla"],
                "english": label["english"],
                "confidence": f"{confidence_percentage:.2f}%"
            },
            "metadata": {
                "filename": file.filename,
                "total_frames": actual_frames
            },
            "landmarks": raw_seq.tolist()
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict-landmarks-focus")
async def predict_landmarks_focus(file: UploadFile = File(...)):
    """
    Returns:
    - prediction
    - landmarks (raw)
    - cam: temporal grad-cam (60,)
    - focus_points: per-point saliency (60, num_points)
    """
    temp_path = f"temp_focus_{file.filename}"
    try:
        save_upload_file(file, temp_path)

        raw_seq, actual_frames = extract_landmarks_from_video(temp_path)
        norm_seq = normalize_sequence(raw_seq)

        x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx_tensor = torch.max(probs, dim=1)
            pred_idx = int(pred_idx_tensor.item())
            confidence_percentage = float(conf.item() * 100)

        # Temporal Grad-CAM: choose last Conv1d layer
        # If your model.cnn differs, adjust index to the last nn.Conv1d
        target_conv = model.cnn[3]
        cam_engine = TemporalGradCAM(model, target_conv)
        try:
            cam_tprime, _, _ = cam_engine(x, class_idx=pred_idx)
        finally:
            cam_engine.close()
        cam_60 = upsample_cam(cam_tprime, target_len=MAX_FRAMES).tolist()

        # Landmark focus
        sal_TF, _, _ = landmark_saliency(model, x, class_idx=pred_idx)
        focus_points = [feature_saliency_to_point_scores(sal_TF[t]) for t in range(MAX_FRAMES)]

        label = labels[str(pred_idx)]
        return {
            "prediction": {
                "word_id": label["id"],
                "bangla": label["bangla"],
                "english": label["english"],
                "confidence": f"{confidence_percentage:.2f}%"
            },
            "metadata": {
                "filename": file.filename,
                "total_frames": actual_frames
            },
            "landmarks": raw_seq.tolist(),
            "cam": cam_60,
            "focus_points": focus_points
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)