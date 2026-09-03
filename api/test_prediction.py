import os
import sys
import io
import json
import cv2
import torch
import numpy as np
import mediapipe as mp
import onnxruntime as ort

from transformer import TransformerSignModel
from cnn_bilstm import CNNBiLSTMAttention
from rqe import apply_rqe

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

LABEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "labels.json"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Models"))
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "onnx_models"))

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

IMPORTANT_FACE_IDX = [
    33, 133, 159, 145, 468, 469,
    263, 362, 386, 374, 471, 472,
    105, 107, 55, 65, 52,
    285, 295, 282, 283, 336,
    1, 2, 98, 327, 94, 97, 168, 197,
    13, 14, 78, 308, 82, 312, 87, 317, 88, 95, 178, 191,
    80, 81, 82, 311, 310, 415, 291, 308, 324, 318, 402, 317,
]

def interpolate_sequence(seq, target_len=60):
    T = len(seq)
    if T == 0:
        return np.zeros((target_len, 387), dtype=np.float32)
    if T == target_len:
        return np.array(seq, dtype=np.float32)
    
    seq = np.array(seq, dtype=np.float32)
    D = seq.shape[1]
    orig_grid = np.linspace(0, 1, T)
    target_grid = np.linspace(0, 1, target_len)
    
    new_seq = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        new_seq[:, d] = np.interp(target_grid, orig_grid, seq[:, d])
    return new_seq

def _fill_missing_landmarks(seq):
    if not seq:
        return seq
    arr = np.array(seq, dtype=np.float32)
    T, D = arr.shape
    N_LM = D // 3
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

def _trim_active_motion_window(seq, min_frames=15):
    T = len(seq)
    if T <= min_frames:
        return seq
    arr = np.array(seq, dtype=np.float32)
    lh_start = 54 * 3
    lh_end = 75 * 3
    rh_start = 75 * 3
    rh_end = 96 * 3
    lh_active = np.any(arr[:, lh_start:lh_end] != 0, axis=1)
    rh_active = np.any(arr[:, rh_start:rh_end] != 0, axis=1)
    hands_active = lh_active | rh_active
    active_indices = np.where(hands_active)[0]
    if len(active_indices) >= min_frames:
        start_f = max(0, active_indices[0] - 4)
        end_f = min(T, active_indices[-1] + 5)
        if (end_f - start_f) >= min_frames:
            return seq[start_f:end_f]
    return seq

def extract_video_features(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    
    seq = []
    mp_holistic = mp.solutions.holistic
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
    print(f"Extracted {len(seq)} raw frames.")
    
    filled = _fill_missing_landmarks(seq)
    trimmed = _trim_active_motion_window(filled)
    print(f"After active motion trimming: {len(trimmed)} frames.")
    
    interpolated = interpolate_sequence(trimmed, target_len=max_frames)
    norm_seq = apply_rqe(interpolated, quantization_step=0.05, shoulder_fixing=True)
    return norm_seq

def test_video(video_path, model_type="transformer_interpolated_frontview"):
    print(f"\n=======================================================")
    print(f"Testing video: {video_path}")
    print(f"Model: {model_type}")
    print(f"=======================================================")
    
    norm_seq = extract_video_features(video_path)
    
    # 1. Test PyTorch
    ckpt_map = {
        "transformer_frontview": ("transformer_frontveiw.pth", TransformerSignModel),
        "transformer_interpolated_frontview": ("transformer_interpolated_frontveiw.pth", TransformerSignModel),
        "transformer_multiview": ("transformer_multiview.pth", TransformerSignModel),
        "transformer_interpolated_multiview": ("transformer_interpolated_multiview.pth", TransformerSignModel),
        "cnn_frontview": ("cnn_bilstm_attention_frontveiw.pth", CNNBiLSTMAttention),
        "cnn_interpolated_frontview": ("cnn_bilstm_attention_interpolated_frontveiw.pth", CNNBiLSTMAttention),
    }
    
    ckpt_name, arch_cls = ckpt_map.get(model_type, ("transformer_interpolated_frontveiw.pth", TransformerSignModel))
    ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
    
    model = arch_cls(387, 401)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    
    x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
    top_indices = np.argsort(probs)[::-1][:5]
    print("\n--- PyTorch Prediction Top 5 ---")
    for rank, idx in enumerate(top_indices, 1):
        lbl = labels[str(idx)]
        print(f"  #{rank}: {lbl['id']} | {lbl['bangla']} ({lbl['english']}) -> {probs[idx]*100:.2f}%")
        
    # 2. Test ONNX Runtime
    onnx_path = os.path.join(ONNX_DIR, f"{model_type}.onnx")
    if os.path.exists(onnx_path):
        ort_sess = ort.InferenceSession(onnx_path)
        ort_out = ort_sess.run(None, {"input": norm_seq[np.newaxis, ...].astype(np.float32)})[0][0]
        exp_s = np.exp(ort_out - np.max(ort_out))
        ort_probs = exp_s / np.sum(exp_s)
        ort_top = np.argsort(ort_probs)[::-1][:5]
        print("\n--- ONNX Runtime Prediction Top 5 ---")
        for rank, idx in enumerate(ort_top, 1):
            lbl = labels[str(idx)]
            print(f"  #{rank}: {lbl['id']} | {lbl['bangla']} ({lbl['english']}) -> {ort_probs[idx]*100:.2f}%")

if __name__ == "__main__":
    test_videos = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../Datasets/Front/Front/test/W001S04F_02.mp4")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../Datasets/Front/Front/test/W020S04F_01.mp4")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../Datasets/Front/Front/test/W045S04F_01.mp4")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_output.mp4")),
    ]
    for v in test_videos:
        if os.path.exists(v):
            test_video(v, "transformer_interpolated_frontview")
