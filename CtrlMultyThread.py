import queue
import cv2
import threading
import socket
from collections import OrderedDict
import time
import numpy as np

from classes import ManipulatorControl, VoiceProcess, CameraProcess, UDPSender

# ==============================================================================
# ==============================================================================

position = np.array([0, 0.55, 0.0, 0.63, 1.0, 0.0, 0.0,
                                        0.0, -1.0, 0.0,
                                        0.0, 0.0, -1.0])
old_position = position.copy()
eps = np.array([0.00001, 0.00001, 0.00001])

msg_queue = queue.Queue()

command = 3
msg_queue.put(command)

control = ManipulatorControl()
audio = VoiceProcess("models/vosk-model-small-ru-0.22")
# camera = CameraProcess("0.0.0.0", 12345)
udp = UDPSender("192.168.1.2", 8083)

# ==============================================================================
# ==============================================================================

def voice():
    audio.loop(msg_queue)

# ==============================================================================
# ==============================================================================

threading.Thread(target=voice, args=(), daemon=True).start()

# ==============================================================================
# ==============================================================================

# image_path = "photo/inst2.jpg"
# img = cv2.imread(image_path)
# img = cv2.resize(img, (640, 360))

udp.sendMessage("192.168.1.3", 8082, position)

cap = cv2.VideoCapture(0)  # 0 — основная камера, 1 — вторая камера

while cap.isOpened(): 

    ret, img = cap.read()  # Считываем кадр
    if not ret:
        break
    img = cv2.resize(img, (640, 360))

    if not msg_queue.empty():
        command = msg_queue.get_nowait()

    print("Command id: ", command)

    position = control.loop(img, command, position)
    
    if np.abs(position[1:4] - old_position[1:4]).any() >= eps.any():
        position[0] += 1
        if position[0] == 1e6 + 1:
            position[0] = 0

        udp.sendMessage("192.168.1.3", 8082, position)
        old_position = position.copy()