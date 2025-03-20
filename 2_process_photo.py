import cv2
from ultralytics import YOLO
import pymorphy3
import socket
import select

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

coco_classes = ['drill', 'hammer', 'pliers', 'screwdriver', 'wrench']

dictionary = {'дрель':'drill', 'молоток':'hammer', 'плоскогубцы':'pliers', 'отвёртка':'screwdriver', 'гаечный':'wrench'}

# ==============================================================================

image_path = "photo/Instruments_2.jpg"
img = cv2.imread(image_path)
# results = model_yolo(image_path, imgsz=1280, conf=0.3, iou=0.3)

current_keys = []
new_command = True

while True:

    # Ожидание сообщения.
    ready, _, _ = select.select([sock], [], [], 0.01)  

    # Обработка сообщения
    if ready:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        data = data.decode()
        print(data)

        words = data.split()
        lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
        print(f"После лемматизации: {lemmatized_words}")
        new_command = True

        for key in dictionary.keys():
            if key in lemmatized_words:
                if new_command:
                    current_keys = []
                    new_command = False
                v = dictionary[key]
                ind = coco_classes.index(v)
                current_keys.append(ind)
  
    # print(f"Названы ключи: {current_keys}")

    # ==============================================================================

    # Детекция
    results = model_yolo(image_path, imgsz=1280, conf=0.3, iou=0.3, classes=current_keys)
    img = cv2.imread(image_path)    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразуем координаты в int
            conf = float(box.conf[0])  # Уверенность
            cls = int(box.cls[0])  # Класс объекта

            # Рисуем рамку
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Добавляем текст с классом и уверенностью
            # print(f"Class: {coco_classes[cls]}")
            label = f"Class: {coco_classes[cls]}, {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection", img)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

