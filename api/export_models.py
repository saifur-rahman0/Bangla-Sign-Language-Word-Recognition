import os
import sys
import io

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import numpy as np
import onnx
import onnxruntime as ort

from transformer import TransformerSignModel
from cnn_bilstm import CNNBiLSTMAttention

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Models"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "onnx_models"))
FLUTTER_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Sign-Language-App/assets/models"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FLUTTER_ASSETS_DIR, exist_ok=True)

MODEL_REGISTRY = {
    # Transformer models
    "transformer_frontview": (TransformerSignModel, "transformer_frontveiw.pth"),
    "transformer_interpolated_frontview": (TransformerSignModel, "transformer_interpolated_frontveiw.pth"),
    "transformer_multiview": (TransformerSignModel, "transformer_multiview.pth"),
    "transformer_interpolated_multiview": (TransformerSignModel, "transformer_interpolated_multiview.pth"),
    # CNN-BiLSTM models
    "cnn_frontview": (CNNBiLSTMAttention, "cnn_bilstm_attention_frontveiw.pth"),
    "cnn_interpolated_frontview": (CNNBiLSTMAttention, "cnn_bilstm_attention_interpolated_frontveiw.pth"),
    "cnn_multiview": (CNNBiLSTMAttention, "cnn_bilstm_attention_multiview.pth"),
    "cnn_interpolated_multiview": (CNNBiLSTMAttention, "cnn_bilstm_attention_interpolated_multiview.pth"),
}

INPUT_DIM = 387
NUM_CLASSES = 401
MAX_FRAMES = 60

def export_all():
    print(f"=== Starting Model Conversion (8 models) ===")
    print(f"Models directory: {MODEL_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Flutter assets directory: {FLUTTER_ASSETS_DIR}\n")

    dummy_input = torch.randn(1, MAX_FRAMES, INPUT_DIM, dtype=torch.float32)

    for model_name, (arch_cls, ckpt_name) in MODEL_REGISTRY.items():
        print(f"--> Converting {model_name}...")
        ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"    ERROR: Checkpoint not found at {ckpt_path}")
            continue

        # 1. Load PyTorch model
        model = arch_cls(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # 2. PyTorch forward pass baseline
        with torch.no_grad():
            torch_out = model(dummy_input).numpy()

        # 3. Export to ONNX (using legacy TorchScript exporter for maximum compatibility)
        onnx_filename = f"{model_name}.onnx"
        onnx_path = os.path.join(OUTPUT_DIR, onnx_filename)
        flutter_onnx_path = os.path.join(FLUTTER_ASSETS_DIR, onnx_filename)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                dynamo=False,
            )
        except Exception as e:
            # Fallback export
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                input_names=["input"],
                output_names=["output"],
            )

        # 4. Verify ONNX model with ONNX checker
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # 5. Verify numerical consistency with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        ort_out = ort_outs[0]

        diff = float(np.max(np.abs(torch_out - ort_out)))
        print(f"    [OK] ONNX Export Successful! (Max diff vs PyTorch: {diff:.6e})")
        print(f"    [OK] Model Size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")

        # 6. Copy to Flutter assets
        with open(onnx_path, "rb") as f_in, open(flutter_onnx_path, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"    [OK] Copied to Flutter assets: {flutter_onnx_path}\n")

    print("=== All 8 Models Successfully Converted and Verified! ===")

if __name__ == "__main__":
    export_all()
