import cv2
from ultralytics import YOLO
import pymorphy3
import socket
import select

import torch
from torch.nn.functional import interpolate

from PIL import Image

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ==============================================================================

def messageProcces(morph, dictionary):
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    data = data.decode()
    print(data)

    words = data.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    print(f"После лемматизации: {lemmatized_words}")

    for key in dictionary.keys():
        if key in lemmatized_words:
            v = dictionary[key]
            current_key = coco_classes.index(v)
            return current_key
        
def detection(image, current_key, imgsz=1280, conf=0.3, iou=0.3):
    image2 = image.copy()
    results = model_yolo(image, imgsz=imgsz, conf=conf, iou=iou, classes=current_key)
    # img = cv2.imread(image_path)    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразуем координаты в int
            conf = float(box.conf[0])  # Уверенность
            cls = int(box.cls[0])  # Класс объекта

            # Рисуем рамку
            cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Добавляем текст с классом и уверенностью
            # print(f"Class: {coco_classes[cls]}")
            label = f"Class: {coco_classes[cls]}, {conf:.2f}"
            cv2.putText(image2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image2

# ==============================================================================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

# ==============================================================================

# Загружаем обученную модель
model_yolo = YOLO("models/runs/detect/train15/weights/best.pt")
morph = pymorphy3.MorphAnalyzer(lang='ru')

# ==============================================================================

model_seg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
processor_seg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

# ==============================================================================

coco_classes = ['drill', 'hammer', 'pliers', 'screwdriver', 'wrench']

dictionary = {'дрель':'drill', 'молоток':'hammer', 'плоскогубцы':'pliers', 'отвёртка':'screwdriver', 'гаечный':'wrench'}

# ==============================================================================

image_path = "photo/Instruments_2.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (1280, 960))

current_key = None

while True:

    img2 = img.copy()

    # Ожидание сообщения.
    ready, _, _ = select.select([sock], [], [], 0.01)  

    # Обработка сообщения
    if ready:
        current_key = messageProcces(morph, dictionary)
  
    # print(f"Названы ключи: {current_keys}")

    # ==============================================================================

    # Детекция
    img2 = detection(image=img, current_key=current_key, imgsz=1280, conf=0.3, iou=0.3)

    cv2.imshow("Detection", img2)

    cv2.waitKey(1)

