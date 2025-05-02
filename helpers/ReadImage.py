import socket
import time
import cv2
import numpy as np
from collections import OrderedDict
import time

# ==== Настройки ====
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
PACKET_TIMEOUT = 0.5  # Время ожидания завершения кадра

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

print(f"Слушаем порт {UDP_PORT} для потока изображений...")

packets = OrderedDict()
last_packet_time = time.time()

not_first = False

t = time.time()

while True:
    try:
        data, addr = sock.recvfrom(1500)
        last_packet_time = time.time()

        if len(data) <= 2:
            continue

        packet_id = int.from_bytes(data[:2], byteorder='little')
        packet_data = data[2:]

        if not_first and packet_id == 0:
            print(f"\nСобираем кадр из {len(packets)} пакетов...")

            # Собираем все данные
            frame_data = b''.join(packets[k] for k in sorted(packets))

            # Попробуем декодировать JPEG
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                frame = cv2.flip(frame, 1)
                cv2.imshow("ESP32-CAM Live", frame)
                if cv2.waitKey(1) == 27:  # ESC — выход
                    break
            else:
                print("⚠️ Ошибка при декодировании кадра.")

            packets.clear()

        packets[packet_id] = packet_data

        not_first = True

        if 1000*(time.time() - t) >= 50:
            t = time.time()
            if cv2.waitKey(1) == 97:
                sock.sendto(b"a", ("192.168.3.2", UDP_PORT))
                # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaA")
            if cv2.waitKey(1) == 100:
                sock.sendto(b"d", ("192.168.3.2", UDP_PORT))
                # print("B")

    except socket.timeout:
        pass
