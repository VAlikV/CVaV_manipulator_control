import queue
import cv2
import threading
import socket
from collections import OrderedDict
import time

from classes import VideoProcess, VoiceProcess, CameraProcess

# ==============================================================================
# ==============================================================================

msg_queue = queue.Queue()

text_t = "молоток"
old_text = "молоток"
msg_queue.put(text_t)

video = VideoProcess()
audio = VoiceProcess("models/vosk-model-small-ru-0.22")
camera = CameraProcess("0.0.0.0", 12345)

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

cap = cv2.VideoCapture(0)  # 0 — основная камера, 1 — вторая камера

while cap.isOpened(): 

    # f = camera.takeFrame()
    # cv2.imshow("ESP32-CAM Live", f)
    # if cv2.waitKey(1) == 27:  # ESC — выход
    #     break

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

    except queue.Empty:
        text_t = old_text
        video.loop(img, text_t)
