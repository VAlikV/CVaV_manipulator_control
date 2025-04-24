import socket
import keyboard
import time

def array2str(arr):
    msg = "["
    for i in arr:
        msg = msg + str(i) + ","
    msg = msg[:-1] + "]"
    return msg

UDP_IP = "192.168.1.3"
UDP_PORT = 8083
PACKET_TIMEOUT = 0.5  # Время ожидания завершения кадра

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

a = [0.5, 0.2, 0.63, 1.0, 0.0, 0.0,
                     0.0, -1.0, 0.0,
                     0.0, 0.0, -1.0]
t = time.time()*1000

speed = 0.00005

while True:
    if time.time()*1000 - t >= 25:
        if keyboard.is_pressed("a"):
            a[1] += speed
            msg = array2str(a)
            print(msg)
            b = msg.encode('utf-8')
            sock.sendto(b, ("192.168.1.2", 8082))
        if keyboard.is_pressed("d"):
            a[1] -= speed
            msg = array2str(a)
            print(msg)
            b = msg.encode('utf-8')
            sock.sendto(b, ("192.168.1.2", 8082))
        if keyboard.is_pressed("w"):
            a[0] += speed
            msg = array2str(a)
            print(msg)
            b = msg.encode('utf-8')
            sock.sendto(b, ("192.168.1.2", 8082))
        if keyboard.is_pressed("s"):
            a[0] -= speed
            msg = array2str(a)
            print(msg)
            b = msg.encode('utf-8')
            sock.sendto(b, ("192.168.1.2", 8082))