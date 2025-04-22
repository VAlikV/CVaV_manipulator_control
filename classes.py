import os
import queue
import sounddevice as sd
import vosk
import json
import cv2
from ultralytics import YOLO
import pymorphy3
import torch
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sklearn.decomposition import PCA
import socket
from collections import OrderedDict


class VoiceProcess:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print("Ошибка: Указанная модель не найдена!")
            exit(1)
        self.model_vosk_ = vosk.Model(model_path)
        self.sample_rate_ = 16000
        self.recognizer_ = vosk.KaldiRecognizer(self.model_vosk_, self.sample_rate_)
        self.q_ = queue.Queue()

    # ==============================================================================

    def callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.q_.put(bytes(indata))

    # ==============================================================================

    def loop(self, msg_queue):
        with sd.RawInputStream(samplerate=self.sample_rate_, blocksize=8000, dtype="int16",
                            channels=1, callback=self.callback):
            print("Назовите инструменты...")

            while True:
                data = self.q_.get()
                if self.recognizer_.AcceptWaveform(data):
                    result = json.loads(self.recognizer_.Result())
                    temp = result["text"]
                    msg_queue.put(temp)
                    print(f"Распознанный текст: {temp}")

# ==============================================================================
# ==============================================================================
# ==============================================================================

class VideoProcess:
    def loadModels(self):
        self.model_yolo_ = YOLO("models/runs/detect/train15/weights/best.pt")
        self.morph_ = pymorphy3.MorphAnalyzer(lang='ru')

        self.model_seg_ = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.processor_seg_ = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_seg_.to(self.device_)

# ==============================================================================

    def __init__(self):
        self.loadModels()

        self.current_key_ = 3
        self.coco_classes_ = ['drill', 'hammer', 'pliers', 'screwdriver', 'wrench']
        self.dictionary_ = {'дрель':'drill', 'молоток':'hammer', 'плоскогубцы':'pliers', 'отвёртка':'screwdriver', 'гаечный':'wrench'}
        
# ==============================================================================

    def textMessageProcces(self, text):
        words = text.split()
        lemmatized_words = [self.morph_.parse(word)[0].normal_form for word in words]
        print(f"После лемматизации: {lemmatized_words}")

        for key in self.dictionary_.keys():
            if key in lemmatized_words:
                v = self.dictionary_[key]
                current_key = self.coco_classes_.index(v)
                return current_key

# ==============================================================================

    def detection(self, image, current_key, imgsz=640, conf=0.5, iou=0.5):
        max_conf = 0

        detected = False
        fragment = None
        bb_point = None

        image2 = image.copy()
        results = self.model_yolo_(image, imgsz=imgsz, conf=conf, iou=iou, classes=current_key)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразуем координаты в int
                conf = float(box.conf[0])  # Уверенность
                cls = int(box.cls[0])  # Класс объекта

                # Рисуем рамку
                cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if conf > max_conf:
                    detected = True
                    fragment = image2[y1:y2, x1:x2]
                    max_conf = conf

                    bb_point = (x1 , y1)

                # Добавляем текст с классом и уверенностью
                label = f"Class: {self.coco_classes_[cls]}, {conf:.2f}"
                cv2.putText(image2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image2, detected, fragment, bb_point

# ==============================================================================

    def segmentation(self, image):

        label = "a tool grip"
        text = [label]

        inputs = self.processor_seg_(text=text, images=[image]*len(text), return_tensors="pt", padding=True)

        inputs = {k: v.to(self.device_) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model_seg_(**inputs)

        mask = outputs.logits
        mask = mask[0].sigmoid().cpu().numpy()
        # mask = mask[0].sigmoid().numpy()

        mask = mask / np.max(mask)
        mask[mask >= 0.05] = 1
        mask[mask <= 0.05] = 0
        
        return mask

# ==============================================================================

    def calcPCA(self, mask):
        points = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j] == 1:
                    points.append([i,j])

        points = np.array(points)

        pca = PCA(n_components=2)
        pca.fit(points)

        center = np.mean(points, axis=0)
        direction = pca.components_[0]

        length = 200
        # pt1 = (int(center[0]), int(center[1]))
        # pt2 = (int(center[0] + direction[0]*length), int(center[1] + direction[1]*length))
        
        # perpendicular = np.array([-direction[1], direction[0]])
        # grasp_pt1 = (int(center[0] - perpendicular[0]*20), int(center[1] - perpendicular[1]*20))
        # grasp_pt2 = (int(center[0] + perpendicular[0]*20), int(center[1] + perpendicular[1]*20))

        pt1 = (int(center[1]), int(center[0]))
        pt2 = (int(center[1] + direction[1]*length), int(center[0] + direction[0]*length))
        
        perpendicular = np.array([-direction[0], direction[1]])
        grasp_pt1 = (int(center[1] - perpendicular[1]*20), int(center[0] - perpendicular[0]*20))
        grasp_pt2 = (int(center[1] + perpendicular[1]*20), int(center[0] + perpendicular[0]*20))

        # print("grasp_pt1: ", grasp_pt1)
        # print("grasp_pt2: ", grasp_pt2)

        return center, pt1, pt2, grasp_pt1, grasp_pt2
    
# ==============================================================================

    def control(self, bb_center, xe_ts, ye_ts, d_vert = 960, d_hor = 1280):

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

        return x_g_ts, y_g_ts

# ==============================================================================

    def loop(self, image, text):
        current_key = self.textMessageProcces(text)

        if current_key != None:
            self.current_key_ = current_key

        image2, detected, fragment, bb_point = self.detection(image, self.current_key_, imgsz=640, conf=0.5, iou=0.5)

        if detected:
            mask = self.segmentation(fragment)

            center, pt1, pt2, grasp_pt1, grasp_pt2 = self.calcPCA(mask)

            Px = fragment.shape[0] / mask.shape[0]
            Py = fragment.shape[1] / mask.shape[1]

            # point = (bb_point[0]+center[1], bb_point[1]+center[0])
            point = (bb_point[0] + fragment.shape[1] - Py*center[1], bb_point[1] + Px*center[0])

            x_g_ts, y_g_ts = self.control(point, xe_ts=1, ye_ts=1, d_vert = 360, d_hor = 640)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.arrowedLine(mask, pt1, pt2, (0, 0, 255), 2)
            cv2.line(mask, grasp_pt1, grasp_pt2, (0, 255, 0), 2)

            cv2.circle(image2, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)

            cv2.imshow("Mask", mask)
            cv2.imshow("Fragment", fragment)

        cv2.imshow("Detection", image2)

        cv2.waitKey(1) 

# ==============================================================================
# ==============================================================================
# ==============================================================================

class CameraProcess:
    def __init__(self, ip, port):
        self.sock_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_.bind((ip, port))
        self.sock_.settimeout(1.0)

        self.packets_ = OrderedDict()
        self.not_first_ = False

    # ==============================================================================

    def takeFrame(self):
        while True:
            try:
                data, addr = self.sock_.recvfrom(1500)

                if len(data) <= 2:
                    continue

                packet_id = int.from_bytes(data[:2], byteorder='little')
                packet_data = data[2:]

                if not_first and packet_id == 0:
                    print(f"\nСобираем кадр из {len(self.packets_)} пакетов...")

                    # Собираем все данные
                    frame_data = b''.join(self.packets_[k] for k in sorted(self.packets_))

                    # Попробуем декодировать JPEG
                    np_arr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        frame = cv2.flip(frame, 1)
                        return frame
                    else:
                        print("⚠️ Ошибка при декодировании кадра.")

                    self.packets_.clear()
                self.packets_[packet_id] = packet_data
                not_first = True

            except socket.timeout:
                pass
