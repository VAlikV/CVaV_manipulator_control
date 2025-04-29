import queue
import cv2
import threading
import socket
from collections import OrderedDict
import time

from classes import VideoProcess, VoiceProcess, CameraProcess, UDPSender

# ==============================================================================
# ==============================================================================

init_position = [0, 0.55, 0.0, 0.63, 1.0, 0.0, 0.0,
                                0.0, -1.0, 0.0,
                                0.0, 0.0, -1.0]

msg_queue = queue.Queue()

text_t = "молоток"
old_text = "молоток"
msg_queue.put(text_t)

video = VideoProcess()
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

udp.sendMessage("192.168.1.3", 8082, init_position)

cap = cv2.VideoCapture(0)  # 0 — основная камера, 1 — вторая камера

while cap.isOpened(): 

    ret, img = cap.read()  # Считываем кадр
    if not ret:
        break
    img = cv2.resize(img, (640, 360))

    try:
        old_text = text_t
        text_t = msg_queue.get_nowait()
        if text_t == "":
            text_t = old_text

        video.loop(img, text_t)
        delta = video.delta_

        init_position[0] += 1
        if init_position[0] == 1e6 + 1:
            init_position[0] = 0

        init_position[1] -= delta[0]
        init_position[2] -= delta[0]
        udp.sendMessage("192.168.1.3", 8082, init_position)

    except queue.Empty:
        text_t = old_text

        video.loop(img, text_t)
        point = video.delta_

        init_position[0] += 1
        if init_position[0] == 1e6 + 1:
            init_position[0] = 0

        init_position[1] -= delta[0]
        init_position[2] -= delta[0]
        udp.sendMessage("192.168.1.3", 8082, init_position)