#!/usr/bin/env python3
"""Read one frame from the w11_cam_usb firmware and save it as a JPEG.

Usage: python3 save_frame.py [/dev/ttyACM0] [frame.jpg]
Frame format is documented at the top of w11_cam_usb/w11_cam_usb.ino.
"""
import os
import struct
import sys
import tty
import zlib

MAGIC = b"\xA5\x5A\xCA\xFE"
MAX_FRAME_BYTES = 512 * 1024


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
            print("CRC mismatch, skipping frame", file=sys.stderr)
            continue
        return jpeg, timestamp


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    out = sys.argv[2] if len(sys.argv) > 2 else "frame.jpg"

    fd = os.open(port, os.O_RDONLY | os.O_NOCTTY)
    try:
        tty.setraw(fd)  # no line buffering or byte translation
        jpeg, timestamp = read_frame(fd)
    finally:
        os.close(fd)

    with open(out, "wb") as f:
        f.write(jpeg)
    print(f"saved {out}: {len(jpeg)} bytes, t={timestamp} ms")


if __name__ == "__main__":
    main()
