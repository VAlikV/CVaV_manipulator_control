import queue
import cv2
import threading
import socket
from collections import OrderedDict
import time

from classes import ManipulatorControl, VoiceProcess, CameraProcess, UDPSender

# ==============================================================================
# ==============================================================================

position = [0, 0.55, 0.0, 0.63, 1.0, 0.0, 0.0,
                                0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0]
msg_queue = queue.Queue()

command = 1
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

    position = control.loop(img, command, position)

    position[0] += 1
    if position[0] == 1e6 + 1:
        position[0] = 0

    udp.sendMessage("192.168.1.3", 8082, position)