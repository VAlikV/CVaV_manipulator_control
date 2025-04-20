import os
import queue
import sounddevice as sd
import vosk
import json

import cv2
from ultralytics import YOLO
import pymorphy3
import select
import time

import torch
from torch.nn.functional import interpolate

from PIL import Image

from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from sklearn.decomposition import PCA

import socket
import threading

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

# Укажи путь к скачанной модели
MODEL_PATH = "models/vosk-model-small-ru-0.22"
msg_queue = queue.Queue()

text_t = "плоскогубцы"
old_text = "плоскогубцы"
msg_queue.put(text_t)
# ==============================================================================

def messageProcces(text, morph, dictionary, current_key):

    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    print(f"После лемматизации: {lemmatized_words}")

    for key in dictionary.keys():
        if key in lemmatized_words:
            v = dictionary[key]
            current_key = coco_classes.index(v)
            return current_key
        
def detection(image, current_key, imgsz=1280, conf=0.5, iou=0.5):
    max_conf = 0

    image2 = image.copy()
    results = model_yolo(image, imgsz=imgsz, conf=conf, iou=iou, classes=current_key)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразуем координаты в int
            conf = float(box.conf[0])  # Уверенность
            cls = int(box.cls[0])  # Класс объекта

            # Рисуем рамку
            cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if conf > max_conf:
                fragment = image2[y1:y2, x1:x2]
                max_conf = conf

                bb_center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            # Добавляем текст с классом и уверенностью
            label = f"Class: {coco_classes[cls]}, {conf:.2f}"
            cv2.putText(image2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image2, fragment, bb_center

def segmentation(model, processor, image, label):

    text = [label]

    inputs = processor(text=text, images=[image]*len(text), return_tensors="pt", padding=True)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    mask = outputs.logits
    # print(mask)
    mask = mask[0].sigmoid().numpy()

    mask = mask / np.max(mask)
    mask[mask >= 0.2] = 1
    
    return mask

def calcPCA(mask):
    points = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                points.append([i,j])

    points = np.array(points)

    pca = PCA(n_components=2)
    pca.fit(points)

    center = np.mean(points, axis=0)
    direction = pca.components_[0]  # главная ось

    length = 200
    pt1 = (int(center[0]), int(center[1]))
    pt2 = (int(center[0] + direction[0]*length), int(center[1] + direction[1]*length))
    
    # Точка захвата (перпендикуляр к главной оси)
    perpendicular = np.array([-direction[1], direction[0]])
    grasp_pt1 = (int(center[0] - perpendicular[0]*20), int(center[1] - perpendicular[1]*20))
    grasp_pt2 = (int(center[0] + perpendicular[0]*20), int(center[1] + perpendicular[1]*20))

    print("grasp_pt1: ", grasp_pt1)
    print("grasp_pt2: ", grasp_pt2)

    return center, pt1, pt2, grasp_pt1, grasp_pt2
    
def fullControl(img, text, current_key):

    new_key = messageProcces(text, morph, dictionary, current_key)
    if new_key != None:
        current_key = new_key
  
    # print(f"Названы ключи: {current_keys}")

    # Детекция
    img2, frag, bb_center = detection(image=img, current_key=current_key, imgsz=1280, conf=0.3, iou=0.3)

    # Сегментация
    mask = segmentation(model=model_seg, processor=processor_seg, image=frag, label="a tool grip")
    
    # PCA
    center, pt1, pt2, grasp_pt1, grasp_pt2 = calcPCA(mask=mask)

    # =========== CONTROL ==============
    print("center: ", bb_center)

    xe_ts = 0.5
    ye_ts = 0.5

    offset = 5

    d_vert = 960
    d_hor = 1280

    xe_p, ye_p = d_vert//2, d_hor//2

    x_old, y_old = bb_center
    x_goal_p, y_goal_p = d_vert - y_old, d_hor - x_old
    print("x_goal_p, y_goal_p: ", x_goal_p, y_goal_p)

    Kx = xe_ts / xe_p
    Ky = ye_ts / ye_p

    # delta_x_p = x_goal_p - xe_p
    # delta_y_p = y_goal_p - ye_p

    # delta_x_ts = Kx * delta_x_p
    # delta_y_ts = Ky * delta_y_p

    # x_g_ts = xe_ts + delta_x_ts
    # y_g_ts = ye_ts + delta_y_ts

    x_g_ts = Kx * x_goal_p
    y_g_ts = Ky * y_goal_p

    print("x_g_ts, y_g_ts: ", x_g_ts, y_g_ts)

      # Визуализация

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.arrowedLine(mask, pt1, pt2, (0, 0, 255), 2)
    cv2.line(mask, grasp_pt1, grasp_pt2, (0, 255, 0), 2)

    cv2.imshow("Detection", img2)
    cv2.imshow("Fragment", frag)
    cv2.imshow("Mask", mask)

    cv2.waitKey(1)   

# ==============================================================================
# ==============================================================================
# ==============================================================================

def voiceRecognition():

    # Проверяем, что модель существует
    if not os.path.exists(MODEL_PATH):
        print("Ошибка: Указанная модель не найдена!")
        exit(1)

    # Загружаем модель
    model_vosk = vosk.Model(MODEL_PATH)

    # Настройки аудио
    sample_rate = 16000  # Частота дискретизации
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """Функция обработки звука в реальном времени"""
        if status:
            print(status, flush=True)
        q.put(bytes(indata))

    # Инициализируем распознаватель
    recognizer = vosk.KaldiRecognizer(model_vosk, sample_rate)

    # ==============================================================================

    # Запускаем захват аудио
    with sd.RawInputStream(samplerate=sample_rate, blocksize=8000, dtype="int16",
                        channels=1, callback=callback):
        print("Назовите инструменты...")

        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                temp = result["text"]
                msg_queue.put(temp)
                print(f"Распознанный текст: {temp}")

# ==============================================================================
# ==============================================================================
# ==============================================================================
    
    

# ==============================================================================
# ==============================================================================
# ==============================================================================

threading.Thread(target=voiceRecognition, args=(), daemon=True).start()

image_path = "photo/Instruments_2.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (1280, 960))
current_key = 3

while True: 
    try:
        old_text = text_t
        text_t = msg_queue.get_nowait()
        if text_t == "":
            text_t = old_text
        fullControl(img, text_t, current_key)
    except queue.Empty:
        text_t = old_text
        fullControl(img, text_t, current_key)

    