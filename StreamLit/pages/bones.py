import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

st.title('Segmentation of the broken bones')

CFG_MODEL_PATH = '/Users/irina/workspace/ds_bootcamp/cv_project/weights/best_bones.pt'

image_source = st.radio("Choose the option of uploading the image of bones:", ("File", "URL"))

if image_source == "File":
    uploaded_file = st.file_uploader("Upload the image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    url = st.text_input("Enter the URL of image...")
    if url:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

if 'image' in locals():
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    model = YOLO('/Users/irina/workspace/ds_bootcamp/cv_project/weights/best_bones.pt')  # Укажите путь к предобученной модели или имя (например, 'yolov5s')
    results = model.predict(image, imgsz=320, conf=0.5)
    if st.button("Detect bone fractures", type="primary"):
      for r in results:
          im_array = r.plot()
          im = Image.fromarray(im_array[..., ::-1])
          st.image(im, caption="Predicted Image", use_column_width=True)