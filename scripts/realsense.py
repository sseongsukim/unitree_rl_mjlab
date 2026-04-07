import socket
import struct
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
PORT = 5005

HEADER_FMT = "<B3xII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

color_chunks = {}
depth_chunks = {}

color_size = None
depth_size = None

latest_depth = None  # ← depth raw 여기 저장됨


def reconstruct(chunks, expected_size):
    if expected_size is None:
        return None

    total = sum(len(v) for v in chunks.values())
    if total < expected_size:
        return None

    data = bytearray(expected_size)
    for offset in sorted(chunks.keys()):
        chunk = chunks[offset]
        data[offset : offset + len(chunk)] = chunk

    return bytes(data)


while True:
    packet, _ = sock.recvfrom(65535)

    if len(packet) < HEADER_SIZE:
        continue

    msg_type, total_size, offset = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
    payload = packet[HEADER_SIZE:]

    # # ===== COLOR =====
    # if msg_type == 0:
    #     if color_size != total_size:
    #         color_chunks.clear()
    #         color_size = total_size

    #     color_chunks[offset] = payload

    #     frame = reconstruct(color_chunks, color_size)
    #     if frame is not None:
    #         color_chunks.clear()
    #         img = cv2.imdecode(np.frombuffer(frame, np.uint8), 1)

    #         if img is not None:
    #             cv2.imshow("RGB", img)

    # ===== DEPTH =====
    if msg_type == 1:
        if depth_size != total_size:
            depth_chunks.clear()
            depth_size = total_size

        depth_chunks[offset] = payload

        frame = reconstruct(depth_chunks, depth_size)
        if frame is not None:
            depth_chunks.clear()
            latest_depth = np.frombuffer(frame, np.uint16).reshape(HEIGHT, WIDTH)

            print(latest_depth[240, 320])

    if cv2.waitKey(1) & 0xFF == 27:
        break

sock.close()
cv2.destroyAllWindows()
