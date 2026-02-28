import streamlit as st
import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import tempfile
import matplotlib.pyplot as plt

# Import your model architecture
from model import CNN_BiLSTM_Attention 

# ---------------- CONFIG & LOAD ----------------
MODEL_PATH = "cnn_bilstm_attention.pth"
LABEL_PATH = "labels.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_resources():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    num_classes = len(labels)
    model = CNN_BiLSTM_Attention(387, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, labels

model, labels = load_resources()

# Constants
INPUT_DIM = 387
MAX_FRAMES = 60
IMPORTANT_FACE_IDX = [
    33, 133, 159, 145, 468, 469, 263, 362, 386, 374, 471, 472,
    105, 107, 55, 65, 52, 285, 295, 282, 283, 336,
    1, 2, 98, 327, 94, 97, 168, 197,
    13, 14, 78, 308, 82, 312, 87, 317, 88, 95, 178, 191,
    80, 81, 82, 311, 310, 415, 291, 308, 324, 318, 402, 317
]

# ---------------- UTILS (The Missing Functions) ----------------

def extract_landmarks_from_video(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    seq = []
    
    with mp_holistic.Holistic(
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

            # --- FACE LANDMARKS (With Safety Check) ---
            if results.face_landmarks:
                num_detected = len(results.face_landmarks.landmark)
                for idx in IMPORTANT_FACE_IDX:
                    # Check if the index is valid for this specific detection
                    if idx < num_detected:
                        p = results.face_landmarks.landmark[idx]
                        lm += [p.x, p.y, p.z]
                    else:
                        # Fallback if specific landmark index is missing
                        lm += [0.0, 0.0, 0.0]
            else:
                # No face detected at all in this frame
                lm += [0.0] * (len(IMPORTANT_FACE_IDX) * 3)

            # --- HANDS ---
            for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand:
                    for p in hand.landmark:
                        lm += [p.x, p.y, p.z]
                else:
                    lm += [0.0] * (21 * 3)

            # --- POSE ---
            if results.pose_landmarks:
                for p in results.pose_landmarks.landmark:
                    lm += [p.x, p.y, p.z]
            else:
                lm += [0.0] * (33 * 3)

            # Safety: Ensure the final list matches INPUT_DIM (387)
            if len(lm) == INPUT_DIM:
                seq.append(lm)
            
            if len(seq) >= max_frames:
                break

    cap.release()
    actual_length = len(seq)

    # Padding if video is shorter than 60 frames
    if len(seq) < max_frames:
        if len(seq) == 0:
            seq = [[0.0] * INPUT_DIM] * max_frames
        else:
            padding = [[0.0] * INPUT_DIM] * (max_frames - len(seq))
            seq.extend(padding)

    return np.array(seq, dtype=np.float32), actual_length

def normalize_sequence(seq):
    # Simplified normalization for display
    norm_seq = []
    for frame in seq:
        frame_reshaped = frame.reshape(-1, 3)
        center = np.mean(frame_reshaped, axis=0)
        normalized = (frame_reshaped - center)
        norm_seq.append(normalized.flatten())
    return np.array(norm_seq, dtype=np.float32)

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="BdSL Recognition", layout="wide")
st.title("🇧🇩 BdSL Sign Recognition")

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    col1, col2 = st.columns(2)
    with col1:
        st.video(video_path)

    if st.button("Predict Sign"):
        with st.spinner("Analyzing Video..."):
            # 1. Extract & Normalize
            raw_seq, actual_frames = extract_landmarks_from_video(video_path)
            norm_seq = normalize_sequence(raw_seq)
            
            # 2. Model Inference
            x = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx_tensor = torch.max(probs, dim=1)
                pred_idx = str(int(pred_idx_tensor.item()))

            # 3. Show Results
            res = labels[pred_idx]
            st.success(f"### Prediction: {res['bangla']} ({res['english']})")
            st.metric("Confidence", f"{conf.item()*100:.2f}%")
            
            # 4. Temporal Plot
            st.subheader("Temporal Importance")
            # For simplicity, we just plot a dummy importance or Grad-CAM here
            st.line_chart(np.random.random(60)) # Placeholder for GradCAM logic

    # Cleanup
    if os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass
