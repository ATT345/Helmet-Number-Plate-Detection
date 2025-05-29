import streamlit as st
from PIL import Image
import os
import tempfile
from ultralytics import YOLO
import shutil
import time
import sys

# 🔒 Fix UnpicklingError in PyTorch 2.6 by adding trusted classes
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.container import ModuleList
from torch.nn.modules.pooling import MaxPool2d
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck, BottleneckCSP, SPPF, SPP

# ✅ Register all custom layers used in the model
torch.serialization.add_safe_globals([
    DetectionModel, Sequential, Conv, Conv2d,
    BatchNorm2d, SiLU, C2f, Bottleneck, BottleneckCSP,
    SPPF, SPP, ModuleList, MaxPool2d
])

# ✅ Check if model file exists
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error("Model file 'best.pt' not found. Please upload it.")
    sys.exit("Missing model file")

# ✅ Load YOLOv8 model once
model = YOLO(MODEL_PATH)

def main():
    st.title("🪖 Helmet and Number Plate Detection")
    st.write("Upload an image to detect helmets and number plates using YOLOv8.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)
        st.write("🔍 Detecting objects...")

        # Run YOLOv8 prediction
        results = model.predict(source=image_path, save=True, project=temp_dir, name="result", conf=0.25)

        result_folder = os.path.join(temp_dir, "result")
        if os.path.exists(result_folder):
            image_files = [f for f in os.listdir(result_folder) if f.endswith((".jpg", ".png"))]
            if image_files:
                result_image_path = os.path.join(result_folder, image_files[0])
                st.image(Image.open(result_image_path), caption="Detected Image", use_column_width=True)
                st.success("✅ Detection complete!")
            else:
                st.error("⚠️ No detected image found.")
        else:
            st.error("❌ Detection failed. Result folder not found.")

        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
