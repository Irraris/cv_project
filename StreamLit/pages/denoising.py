import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.utils import save_image
from torchvision.io import read_image
import torchutils as tu
from torchvision import io
import requests
import io
from PIL import Image

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.SELU()
            )

        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True)

        self.unpool = nn.MaxUnpool2d(2, 2)

        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)
        return out

model = ConvAutoencoder()
model.load_state_dict(torch.load('/Users/teery/ds_bootcamp/cv_project/weights/model.pth', map_location=torch.device('cpu')))
model.eval()

choose = st.radio("Выбери способ загрузки", ["Ссыл'очка", "Локал'очка"])
if choose == "Ссыл'очка":
    link = st.text_input('ссыль сюда')
    if link:
        response = requests.get(link)
        image_pil = Image.open(io.BytesIO(response.content))
        image_tensor = T.ToTensor()(image_pil)
        image_pil2 = T.ToPILImage()(model(image_tensor.unsqueeze(0)).squeeze(1))
        st.image([image_pil, image_pil2], ['Бефор', 'Афтер'])
else:
    uploaded_file = st.file_uploader(label='jpg, png, jpeg', label_visibility='collapsed')
    if uploaded_file:
        img = Image.open(uploaded_file)
        image_tensor = T.ToTensor()(img)
        img2 = T.ToPILImage()(model(image_tensor.unsqueeze(0)).squeeze(1))
        st.image([img, img2], ['Бефор', 'Афтер'])