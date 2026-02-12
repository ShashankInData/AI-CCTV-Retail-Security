import torch
print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")

from ultralytics import YOLO
print("YOLO: OK")

import supervision
print(f"Supervision: {supervision.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import fastapi
print(f"FastAPI: {fastapi.__version__}")

import streamlit
print(f"Streamlit: {streamlit.__version__}")

import sqlalchemy
print(f"SQLAlchemy: {sqlalchemy.__version__}")

import pandas
print(f"Pandas: {pandas.__version__}")

from PIL import Image, ImageColor
print(f"Pillow: {Image.__version__}")

print("\nAll imports OK!")
