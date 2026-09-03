# 🤟 BdSLW401: A Large-Scale Word-Level Bengali Sign Language Recognition System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED.svg?style=flat&logo=onnx)](https://onnxruntime.ai/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E.svg)](https://huggingface.co/spaces/saifur2025/BdSLW)
[![Flutter Mobile App](https://img.shields.io/badge/Flutter-Android%20%7C%20iOS-02569B.svg?style=flat&logo=flutter)](https://github.com/saifur-rahman0/Sign-Language-App)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**BdSLW401** is an end-to-end framework for **isolated word-level Bengali Sign Language (BdSL) recognition across 401 distinct vocabulary classes**. The project features advanced spatial-temporal feature extraction (MediaPipe Holistic), novel Relative Quantization Encoding with Shoulder Fixing (**RQE-SF**), state-of-the-art deep architectures (**Transformer** and **CNN-BiLSTM Attention**), Explainable AI with **Grad-CAM**, and deployment pipelines for cloud APIs (FastAPI / Hugging Face Spaces) and edge/mobile devices (ONNX Runtime + Flutter).

---

## 📌 Key Highlights

- 📚 **401 Bengali Sign Vocabulary Classes** (`W001` - `W401`) covering family, everyday actions, medical terms, food, emotions, and numerals.
- 📐 **129 Keypoint Spatial Representation (387 Dimensions)** per frame:
  - **54 Face Landmarks** (Eyes, Eyebrows, Nose bridge, Inner/Outer Lips)
  - **21 Left Hand Finger Joints** (Full 3D articulation)
  - **21 Right Hand Finger Joints** (Full 3D articulation)
  - **33 Pose/Body Joints** (Torso, Shoulders, Elbows, Wrists)
- ⚙️ **RQE-SF Normalization (Relative Quantization Encoding with Shoulder Fixing)**:
  - Invariant to distance, subject height, and camera perspective.
  - Fixes body center at Pose 0 and scales based on inter-shoulder bi-acromial distance.
- 🧠 **Dual State-of-the-Art Deep Learning Models**:
  - **Temporal Self-Attention Transformer** (Multi-head attention over 60-frame sequences).
  - **CNN-BiLSTM with Attention** (Local spatial feature extraction + bidirectional temporal context).
- 🔍 **Explainable AI (XAI)**:
  - Spatial landmark focus heatmaps.
  - Temporal Grad-CAM saliency curves.
- ⚡ **Cross-Platform Deployments**:
  - **Cloud API**: Production-ready asynchronous FastAPI backend hosted on Hugging Face Spaces.
  - **Edge / On-Device**: ONNX Runtime INT8/FP32 engine for offline native mobile inference in Flutter.

---

## 📊 Benchmark Results

Evaluated on the standardized **BdSLW401 Test Dataset** across Frontview and Multiview setups:

### 🏆 Frontview Architectures Benchmark

| Architecture | Sequence Type | Top-1 Accuracy (%) | Word Error Rate (WER %) | Parameters | Model Size (MB) | CPU Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Transformer** | **Interpolated (60f)** | **99.64%** | **0.36%** | **2.60M** | **9.96 MB** | **~12.4 ms** |
| Transformer | Standard | 99.12% | 0.88% | 2.60M | 9.96 MB | ~12.2 ms |
| **CNN-BiLSTM Attention** | **Interpolated (60f)** | **98.85%** | **1.15%** | **0.98M** | **3.73 MB** | **~8.1 ms** |
| CNN-BiLSTM Attention | Standard | 98.42% | 1.58% | 0.98M | 3.73 MB | ~8.0 ms |

### 📈 Training Curves & Comparisons

| Frontview Benchmark | Multiview Benchmark |
| :---: | :---: |
| ![Frontview Benchmark](Figures/Frontview%20Model's%20Benchmark.png) | ![Multiview Benchmark](Figures/Multiview%20Model's%20Benchmark.png) |

| Transformer Accuracy Curve | CNN-BiLSTM Accuracy Curve |
| :---: | :---: |
| ![Transformer Accuracy](Figures/Accuracy%20Curve%20of%20Transformer%20frontview.png) | ![CNN-BiLSTM Accuracy](Figures/Accuracy%20Curve%20of%20CNN%20BiLSTM%20Attention%20multiview.png) |

---

## 🗂️ Project Directory Structure

```text
BdSLW401/
├── Datasets/                               # Metadata and split manifests
│   ├── train.csv, val.csv, test.csv        # Standard dataset splits (51,000+ samples)
│   ├── train_interpolated.csv              # 60-frame temporally interpolated splits
│   ├── train_multiview.csv                 # Multiview camera angle splits
│   ├── label.json                          # Full 401-class Bangla/English dictionary
│   └── bdsl words-complete.pdf             # Vocabulary index guide
├── Models/                                 # Trained PyTorch model checkpoints
│   ├── transformer_frontveiw.pth           # Frontview Transformer
│   ├── transformer_interpolated_frontveiw.pth # Best Frontview Transformer (99.64%)
│   ├── transformer_multiview.pth           # Multiview Transformer
│   ├── transformer_interpolated_multiview.pth
│   ├── cnn_bilstm_attention_frontveiw.pth  # Frontview CNN-BiLSTM
│   ├── cnn_bilstm_attention_interpolated_frontveiw.pth
│   ├── cnn_bilstm_attention_multiview.pth  # Multiview CNN-BiLSTM
│   └── cnn_bilstm_attention_interpolated_multiview.pth
├── Notebooks/                              # Step-by-step modular pipeline
│   ├── 0_process_folders.ipynb             # Dataset structuring & indexing
│   ├── 1_extract_landmarks.ipynb           # MediaPipe Holistic landmark extraction
│   ├── 1.1_extract_landmarks_interpolated.ipynb
│   ├── 1.2_extract_landmarks_interpolated_parallel.py # High-speed parallel extractor
│   ├── 2_normalize_landmarks.ipynb         # Spatial normalization
│   ├── 2.2_normalize_landmarks_interpolated.ipynb     # RQE-SF implementation
│   ├── 3_prepare_dataset.ipynb             # Train/Val/Test CSV split generation
│   ├── 3.1_prepare_dataset_interpolated.ipynb
│   ├── 3.2_prepare_dataset_multiview.ipynb
│   ├── 4_train_model.ipynb                 # PyTorch Transformer model training
│   ├── 4.1_cnn_bilstm_model_train.ipynb    # CNN-BiLSTM Attention model training
│   ├── 4.2_cnn_bilstm_multiview_train.ipynb
│   ├── 4.3_transformer_multiview_train.ipynb
│   ├── 5_evaluate_model.ipynb              # Evaluation & confusion matrix metrics
│   ├── 6.1_benchmark_frontview_models.ipynb# Frontview comparative evaluation
│   └── 6.2_benchmark_multiview_models.ipynb# Multiview comparative evaluation
├── Figures/                                # Plots, charts, and visualizations
├── api/                                    # Production API & Deployment
│   ├── app.py                              # FastAPI backend (Hugging Face ready)
│   ├── transformer.py                      # PyTorch Transformer architecture
│   ├── cnn_bilstm.py                       # PyTorch CNN-BiLSTM architecture
│   ├── rqe.py                              # Relative Quantization Encoding module
│   ├── labels.json                         # API label dictionary mapping
│   ├── export_models.py                    # PyTorch to ONNX converter
│   ├── test_prediction.py                  # End-to-end Python prediction validator
│   ├── generate_app_icons.py               # Mobile icon asset generator
│   └── requirements.txt                    # Python runtime dependencies
├── sample_output.mp4                       # Sample demonstration sign video
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/saifur-rahman0/BdSLW401.git
cd BdSLW401
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate

pip install -r api/requirements.txt
```

### 3. Test Model Predictions on Video
```bash
python api/test_prediction.py
```

### 4. Run the FastAPI Local Server
```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```
Navigate to `http://localhost:7860/docs` to test interactive Swagger documentation.

---

## 🚀 API Usage

### Predict Sign Language Video (`POST /predict-video`)
```bash
curl -X POST "http://localhost:7860/predict-video?model_type=transformer_interpolated_frontview" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_output.mp4"
```

#### Sample JSON Response:
```json
{
  "model_used": "transformer_interpolated_frontview",
  "class_index": 0,
  "word_id": "W001",
  "bangla": "বাবা",
  "english": "Father",
  "confidence": "99.6%",
  "landmarks": [...],
  "focus_points": [...],
  "cam": [0.12, 0.45, 0.98, ...]
}
```



## 📑 Citation & License

This project is licensed under the **MIT License**.

If you use **BdSLW401** in your research, please consider citing:

```bibtex
@misc{bdslw401_2026,
  author = {Saifur Rahman},
  title = {BdSLW401: A Large-Scale Isolated Word-Level Bengali Sign Language Recognition Benchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/saifur-rahman0/BdSLW401}}
}
```
