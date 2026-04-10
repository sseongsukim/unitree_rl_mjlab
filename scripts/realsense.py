import argparse
import csv
from pathlib import Path
import socket
import struct
import time

WIDTH = 640
HEIGHT = 480
PORT = 5005

HEADER_FMT = "<B3xII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-period-csv",
        default=None,
        help="Path to save depth frame receive period logs as CSV.",
    )
    parser.add_argument(
        "--depth-log-flush",
        action="store_true",
        help="Flush the CSV log after every depth frame.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Keep --help available even when runtime vision dependencies are not installed.
    import cv2
    import numpy as np

    color_chunks = {}
    depth_chunks = {}
    color_size = None
    depth_size = None
    latest_depth = None  # ← depth raw 여기 저장됨
    depth_frame_count = 0
    last_depth_time = None
    depth_period_file = None
    depth_period_writer = None
    sock = None

    try:
        if args.depth_period_csv is not None:
            depth_period_path = Path(args.depth_period_csv)
            depth_period_path.parent.mkdir(parents=True, exist_ok=True)
            depth_period_file = open(depth_period_path, "w", newline="")
            depth_period_writer = csv.writer(depth_period_file)
            depth_period_writer.writerow(
                [
                    "frame",
                    "wall_time_sec",
                    "monotonic_time_sec",
                    "period_sec",
                    "frequency_hz",
                ]
            )

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", PORT))

        while True:
            packet, _ = sock.recvfrom(65535)

            if len(packet) < HEADER_SIZE:
                continue

            msg_type, total_size, offset = struct.unpack(
                HEADER_FMT, packet[:HEADER_SIZE]
            )
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
                    latest_depth = np.frombuffer(frame, np.uint16).reshape(
                        HEIGHT, WIDTH
                    )

                    wall_time = time.time()
                    now = time.perf_counter()
                    period_sec = None
                    frequency_hz = None
                    if last_depth_time is not None:
                        period_sec = now - last_depth_time
                        frequency_hz = 1.0 / period_sec if period_sec > 0.0 else None
                    last_depth_time = now

                    depth_frame_count += 1
                    if depth_period_writer is not None:
                        depth_period_writer.writerow(
                            [
                                depth_frame_count,
                                wall_time,
                                now,
                                period_sec,
                                frequency_hz,
                            ]
                        )
                        if args.depth_log_flush:
                            depth_period_file.flush()

                    print(latest_depth[240, 320])

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if sock is not None:
            sock.close()
        if depth_period_file is not None:
            depth_period_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
