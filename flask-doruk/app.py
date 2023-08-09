from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn  

from werkzeug.utils import secure_filename
from PIL import Image

from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)

# PyTorch modelini yükleme fonksiyonu
def pytorch_modeli_yukle(model_yolu):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 5)
    )
    model.load_state_dict(torch.load(model_yolu))
    model.eval()
    return model

# PyTorch modeliyle görüntüyü ön işleme yapma fonksiyonu
def goruntu_on_isleme(img_yolu):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_yolu)
    img = transform(img).unsqueeze(0)
    return img

# Model tahmini için fonksiyon
def model_tahmin_et(img_yolu, model):
    img = goruntu_on_isleme(img_yolu)
    
    with torch.no_grad():
        cikis = model(img)
        tahmin_sonucu = torch.argmax(cikis, dim=1).item()
    
    sınıflar = ["DR YOK", "HAFİF NPDR", "ORTA NPDR", "ŞİDDETLİ NPDR", "PDR"]
    print("Tahmin:", sınıflar[tahmin_sonucu])
    return sınıflar[tahmin_sonucu]

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")
    
@app.route('/hakkinda')
def hakkinda():
    return "Bu site Doruk Ertosun tarafından oluşturulmuştur."
    
@app.route('/predict', methods=["GET", "POST"])
def yukle():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        dosya_yolu = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(dosya_yolu)

        tahmin_sonucu = model_tahmin_et(dosya_yolu, pytorch_model)
        os.remove(dosya_yolu)

        return tahmin_sonucu
    return None

if __name__ == "__main__":
    pytorch_model_yolu = r"C:\Users\PC\Desktop\flask-doruk\model.pth"  # PyTorch modelinizin yolunu buraya girin
    pytorch_model = pytorch_modeli_yukle(pytorch_model_yolu)
    app.run(debug=True)
