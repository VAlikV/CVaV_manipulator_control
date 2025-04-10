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

from sklearn.decomposition import PCA

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

            # Добавляем текст с классом и уверенностью
            label = f"Class: {coco_classes[cls]}, {conf:.2f}"
            cv2.putText(image2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image2, fragment

def segmentation(model, processor, image, label):

    text = [label]

    inputs = processor(text=text, images=[image]*len(text), return_tensors="pt", padding=True)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    mask = outputs.logits
    # print(mask)
    return mask[0].sigmoid().numpy()
    

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

current_key = 2

while True:

    img2 = img.copy()

    # Ожидание сообщения.
    ready, _, _ = select.select([sock], [], [], 0.01)  

    # Обработка сообщения
    if ready:
        new_key = messageProcces(morph, dictionary)
        if new_key != None:
            current_key = new_key
  
    # print(f"Названы ключи: {current_keys}")

    # Детекция
    img2, frag = detection(image=img, current_key=current_key, imgsz=1280, conf=0.3, iou=0.3)

    # Сегментация
    mask = segmentation(model=model_seg, processor=processor_seg, image=frag, label="tool")
    mask = mask / np.max(mask)
    mask[mask >= 0.1] = 1

    points = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                points.append([i,j])

    points = np.array(points)

    # === 3. PCA ===
    pca = PCA(n_components=2)
    pca.fit(points)

    center = np.mean(points, axis=0)
    direction = pca.components_[0]  # главная ось

    # === 4. Визуализация результата ===
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Отрисуем PCA-ось
    length = 100
    pt1 = (int(center[0]), int(center[1]))
    pt2 = (int(center[0] + direction[0]*length), int(center[1] + direction[1]*length))
    cv2.arrowedLine(mask, pt1, pt2, (0, 0, 255), 2)

    # Точка захвата (перпендикуляр к главной оси)
    perpendicular = np.array([-direction[1], direction[0]])
    grasp_pt1 = (int(center[0] - perpendicular[0]*20), int(center[1] - perpendicular[1]*20))
    grasp_pt2 = (int(center[0] + perpendicular[0]*20), int(center[1] + perpendicular[1]*20))
    cv2.line(mask, grasp_pt1, grasp_pt2, (0, 255, 0), 2)

    cv2.imshow("Detection", img2)
    cv2.imshow("Fragment", frag)
    cv2.imshow("Mask", mask)

    # plt.scatter(points[:, 0], points[:, 1], alpha=0.2)
    # plt.quiver(mean[0], mean[1], axis1[0], axis1[1], color='r', scale=3)
    # plt.quiver(mean[0], mean[1], axis2[0], axis2[1], color='b', scale=3)
    # # plt.gca().invert_yaxis()  # Инверсия оси Y, так как изображения идут сверху вниз
    # plt.show()

    # plt.quiver(mean[0], mean[1], axis1[0], axis1[1], color='r', scale=3)
    # plt.quiver(mean[0], mean[1], axis2[0], axis2[1], color='b', scale=3)

    cv2.waitKey(1)

