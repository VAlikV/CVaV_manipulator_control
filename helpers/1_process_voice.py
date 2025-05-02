import os
import queue
import sounddevice as sd
import vosk
import json

import socket

# ==============================================================================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
# MESSAGE = b"Hello, World!"

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
# print("message: %s" % MESSAGE)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

# ==============================================================================

# Укажи путь к скачанной модели
MODEL_PATH = "models/vosk-model-small-ru-0.22"

# ==============================================================================

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
            text = result["text"]
            print(f"Распознанный текст: {text}")
            sock.sendto(text.encode("utf-8"), (UDP_IP, UDP_PORT))