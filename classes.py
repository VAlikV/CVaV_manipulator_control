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
import time


class VoiceProcess:
    def __init__(self, model_path):

        self.commands_ = ['дрель', 'молоток', 'плоскогубцы', 'отвёртка', 'гаечный', 'спасибо', 'стоп']

        self.morph_ = pymorphy3.MorphAnalyzer(lang='ru')

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
                    # msg_queue.put(temp)
                    print(f"Распознанный текст: {temp}")
                    words = temp.split()
                    lemmatized_words = [self.morph_.parse(word)[0].normal_form for word in words]
                    print(f"После лемматизации: {lemmatized_words}")

                    for key in self.commands_:
                        if key in lemmatized_words:
                            current_key = self.commands_.index(key)
                            msg_queue.put(current_key)

# ==============================================================================
# ==============================================================================
# ==============================================================================

class ManipulatorControl:
    def loadModels(self):
        self.model_yolo_ = YOLO("models/runs/detect/train15/weights/best.pt")

        self.model_seg_ = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.processor_seg_ = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_seg_.to(self.device_)

    # ==============================================================================

    def __init__(self):
        self.loadModels()

        # self.position_done_ = False
        self.N_ = 0

        self.mode_ = 0
        # 0 - ожидание команды
        # 1 - наведение
        # 2 - движение вниз
        # 3 - откат
        # 4 - движение в конец
        # 5 - ожидание начала

        self.initial_height = 0.63
        self.final_height = 0.29

        self.current_key_ = 3
        self.coco_classes_ = ['drill', 'hammer', 'pliers', 'screwdriver', 'wrench']
        # self.dictionary_ = {'дрель':'drill', 'молоток':'hammer', 'плоскогубцы':'pliers', 'отвёртка':'screwdriver', 'гаечный':'wrench'}

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

    def control(self, point, d_vert = 960, d_hor = 1280):

        x_frame_center, y_frame_center = d_hor//2, d_vert//2

        x_frame_target, y_frame_target = point

        x_frame_delta = x_frame_target - x_frame_center
        y_frame_delta = y_frame_target - (y_frame_center + 60)

        # print(f"x_frame_delta {x_frame_delta}, y_frame_delta: {y_frame_delta}")

        Kx = 5000
        Ky = 5000

        y_task_space_delta = x_frame_delta/Kx
        x_task_space_delta = y_frame_delta/Ky

        return x_task_space_delta, y_task_space_delta

    # ==============================================================================

    def action(self, image, manipulator_position):
        if self.mode_ == 0:       # 0 - ожидание команды  
            pass

        elif self.mode_ == 1:     # 1 - наведение
            image2, detected, fragment, box_point = self.detection(image, self.current_key_, imgsz=640, conf=0.5, iou=0.5)

            # ====================================================================

            if detected:    # Если задетектировано
                mask = self.segmentation(fragment)
                center, pt1, pt2, grasp_pt1, grasp_pt2 = self.calcPCA(mask)

                # ====================================================================

                Px = fragment.shape[0] / mask.shape[0]
                Py = fragment.shape[1] / mask.shape[1]
                point = (box_point[0] + Py*center[1], box_point[1] + Px*center[0])

                # ====================================================================

                dx, dy = self.control(point, d_vert = 360, d_hor = 640)

                if abs(dx) <= 0.003:
                    dx = 0
                if abs(dy) <= 0.003:
                    dy = 0
                if (abs(dx) <= 0.003) and (abs(dy) <= 0.003):
                    self.N_ += 1
                    if self.N_ >= 7:
                        self.mode_ = 2

                manipulator_position[1] -= dx
                manipulator_position[2] -= dy

                # ====================================================================

                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.arrowedLine(mask, pt1, pt2, (0, 0, 255), 2)
                cv2.line(mask, grasp_pt1, grasp_pt2, (0, 255, 0), 2)

                cv2.circle(image2, (int(point[0]), int(point[1])), 3, (0, 255, 0), 2)

                cv2.imshow("Mask", mask)
                cv2.imshow("Fragment", fragment)

                # ====================================================================

            cv2.imshow("Detection", image2)
            cv2.waitKey(1) 

            return manipulator_position

        elif self.mode_ == 2:     # 2 - движение вниз
            d = (self.initial_height - self.final_height)/30
            manipulator_position[3] -= d
            time.sleep(5/1000)
            if manipulator_position[3] <= self.final_height:
                self.mode_ = 3

        elif self.mode_ == 3:     # 3 - откат
            manipulator_position[3] += 0.05
            time.sleep(5/1000)
            self.mode_ = 4

        elif self.mode_ == 4:     # 4 - движение в конец

            manipulator_position = np.array([manipulator_position[0], 0.0, -0.55, 0.63, 0, -1, 0,
                                                                                0, 0, -1,
                                                                                1, 0, 0])
            time.sleep(3)
            self.mode_ = 5

        elif self.mode_ == 5:     # 5 - ожидание начала
            pass
        
        return manipulator_position

    # ==============================================================================

    def loop(self, image, command, manipulator_position):

        print("Mode: ", self.mode_)

        if command >= 0 and command <= 4:
            self.current_key_ = command
            if self.mode_ == 0:
                self.mode_ = 1
            manipulator_position = self.action(image, manipulator_position)

        elif command == 5:
            pass
        elif command == 6:
            pass

        return manipulator_position

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

# ==============================================================================
# ==============================================================================
# ==============================================================================

class UDPSender:
    def __init__(self, own_ip = "192.168.1.3", own_port = 8083):
        self.sock_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_.bind((own_ip, own_port))
        self.sock_.settimeout(1.0)

    def array2str(self, arr):
        msg = "["
        for i in arr:
            msg = msg + str(i) + ","
        msg = msg[:-1] + "]"
        return msg
    
    def sendMessage(self, ip, port, array):
        msg = self.array2str(array)
        print(msg)
        b = msg.encode('utf-8')
        self.sock_.sendto(b, (ip, port))

