#!/usr/bin/env python3
"""Relay the ESP32 camera to the network as MJPEG over HTTP. Runs on the Pi.

Usage: python3 pi_server.py [/dev/ttyACM0] [port]
  http://<pi>:8000/          page showing the live stream
  http://<pi>:8000/stream    MJPEG stream (browser <img>, cv2.VideoCapture)
  http://<pi>:8000/frame.jpg latest single frame

The ESP32's JPEGs are forwarded unchanged; nothing is decoded here.
Frame format is documented at the top of esp32_firmware/w11_cam_usb/w11_cam_usb.ino.
Standard library only.
"""
import os
import struct
import sys
import threading
import time
import tty
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAGIC = b"\xA5\x5A\xCA\xFE"
MAX_FRAME_BYTES = 512 * 1024
BOUNDARY = "frame"

PAGE = b"""<!doctype html>
<title>Weedbot camera</title>
<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style>
<img src="/stream">
"""


# ---- Reading frames from the ESP32 ------------------------------------------
def read_exact(fd, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError("serial port closed")
        buf += chunk
    return bytes(buf)


def read_frame(fd):
    window = b""
    while True:
        # Slide byte by byte until the last 4 bytes are the magic.
        window = (window + read_exact(fd, 1))[-4:]
        if window != MAGIC:
            continue
        header = read_exact(fd, 8)
        length, timestamp = struct.unpack("<II", header)
        if not 0 < length <= MAX_FRAME_BYTES:
            continue
        jpeg = read_exact(fd, length)
        (crc,) = struct.unpack("<I", read_exact(fd, 4))
        if zlib.crc32(header + jpeg) != crc:
            continue
        return jpeg, timestamp


class LatestFrame:
    """Holds only the newest frame; readers wait for the next one."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0

    def put(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def get(self, after_seq=0, timeout=5.0):
        """Return (seq, jpeg) newer than after_seq, or (after_seq, None) on timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq, timeout)
            if self._seq > after_seq:
                return self._seq, self._jpeg
            return after_seq, None


def reader(port, latest):
    """Read frames forever, reopening the port if the ESP32 is unplugged or resets."""
    while True:
        try:
            fd = os.open(port, os.O_RDONLY | os.O_NOCTTY)
        except OSError as e:
            print(f"waiting for {port}: {e.strerror}", flush=True)
            time.sleep(2)
            continue
        print(f"reading {port}", flush=True)
        try:
            tty.setraw(fd)  # no line buffering or byte translation
            count, since = 0, time.monotonic()
            while True:
                jpeg, _ = read_frame(fd)
                latest.put(jpeg)
                count += 1
                if time.monotonic() - since >= 10:
                    print(f"{count / (time.monotonic() - since):.1f} fps, {len(jpeg)} bytes", flush=True)
                    count, since = 0, time.monotonic()
        except (OSError, EOFError) as e:
            print(f"lost {port}: {e}", flush=True)
            time.sleep(1)
        finally:
            os.close(fd)


# ---- HTTP -------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    latest = None  # set in main()

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif self.path == "/frame.jpg":
            _, jpeg = self.latest.get()
            if jpeg is None:
                self._send(503, "text/plain", b"no frame from camera\n")
            else:
                self._send(200, "image/jpeg", jpeg)
        elif self.path == "/stream":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found\n")

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = 0
        try:
            while True:
                # Always the newest frame: a slow client skips frames instead of lagging.
                seq, jpeg = self.latest.get(seq)
                if jpeg is None:
                    continue
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away

    def log_message(self, fmt, *args):
        if not self.path.startswith("/stream"):  # one line per client, not per frame
            super().log_message(fmt, *args)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    http_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    Handler.latest = LatestFrame()
    threading.Thread(target=reader, args=(port, Handler.latest), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", http_port), Handler)
    server.daemon_threads = True
    print(f"serving on http://0.0.0.0:{http_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
