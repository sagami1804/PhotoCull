"""YOLO11s-seg を ONNX 形式に変換して photocull/models/ に置く（ビルドの前に 1 回だけ実行）

必要なもの（アプリ本体とは別の環境で構いません。PyTorch は数 GB あるため）:
    python -m venv export_venv
    export_venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    export_venv\\Scripts\\python.exe -m pip install ultralytics onnx onnxslim

実行:
    export_venv\\Scripts\\python.exe tools\\export_yolo11.py

yolo11s-seg.pt は Ultralytics から自動でダウンロードされます（AGPL-3.0）。
"""
import os
import shutil

from ultralytics import YOLO

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "..", "photocull", "models")
os.makedirs(out_dir, exist_ok=True)

model = YOLO("yolo11s-seg.pt")
onnx_path = model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
shutil.copy(onnx_path, os.path.join(out_dir, "yolo11s-seg.onnx"))
print("保存しました:", os.path.join(out_dir, "yolo11s-seg.onnx"))
