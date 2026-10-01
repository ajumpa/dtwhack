#!/usr/bin/env python3
"""Relay the ESP32 camera as MJPEG over HTTP and receive drive commands. Runs on the Pi.

Usage: python3 pi_server.py [/dev/ttyACM0] [http_port] [drive_port]    (defaults 8000, 9000)
  http://<pi>:8000/          page showing the live stream
  http://<pi>:8000/stream    MJPEG stream (browser <img>, cv2.VideoCapture)
  http://<pi>:8000/frame.jpg latest single frame
  UDP port 9000              drive commands from dashboard.py

The ESP32's JPEGs are forwarded unchanged; nothing is decoded here.
Frame format is documented at the top of esp32_firmware/w11_cam_usb/w11_cam_usb.ino.

Each drive packet is JSON: {"seq": 12, "l": 0.6, "r": -0.6}. l and r are the left and right
side speeds, from -1 (full reverse) to 1 (full forward); each motor follows its side in MOTORS.
The dashboard's motor test sends {"seq": 12, "m": [m1, m2, m3, m4]} instead: one speed per motor. The dashboard repeats the command
every 100 ms while a key is held, so if nothing arrives for DRIVE_TIMEOUT seconds (Wi-Fi lost,
tab closed, laptop crashed) the motors stop.

Motors are 28BYJ-48 steppers on ULN2003 boards, driven through the Linux GPIO character device
(/dev/gpiochip*). That needs root (or access to the device); without it the server still runs
the camera and only prints drive commands.

Standard library only.
"""
import ctypes
import fcntl
import glob
import json
import math
import os
import signal
import socket
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
DRIVE_TIMEOUT = 0.5  # seconds without a drive command before stopping

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


# ---- Drive commands ----------------------------------------------------------
# Stepper wiring: BCM GPIO numbers for IN1..IN4 of each ULN2003 board, the side it follows
# ("l" or "r"), and invert to flip a motor that turns the wrong way. The back motors are
# inverted: on each side, front and back spin in opposite directions to move the robot.
# Same controls as the SPV sketch: W forward (both sides forward), S back, A pivot left
# (left side back, right side forward), D pivot right; releasing the keys stops with all coils off.
MOTORS = {
    "M1": {"pins": (5, 6, 13, 19), "side": "l", "invert": False},    # front left
    "M2": {"pins": (12, 16, 20, 21), "side": "r", "invert": False},  # front right
    "M3": {"pins": (22, 23, 24, 25), "side": "l", "invert": True},   # back left
    "M4": {"pins": (4, 18, 26, 27), "side": "r", "invert": True},    # back right
}
MAX_STEP_RATE = 500.0  # full steps/s at speed 1.0 (28BYJ-48: ~2048 per turn, so ~15 RPM); lower if it buzzes
START_RATE = 200.0     # full steps/s to start from; faster starts can stall
ACCEL = 1200.0         # full steps/s per second when speeding up
DEADBAND = 0.05        # speeds below this count as stop
# Full-step, two coils on (more torque and ~2x the speed of half-stepping), IN1..IN4.
# Two coils draw ~200 mA per motor while moving; on the Pi's 5 V pin, watch for brownouts.
STEPS = [int(bits[::-1], 2) for bits in ("1100", "0110", "0011", "1001")]
GPIO_CHIP_LABEL = "pinctrl-rp1"  # Pi 5 header GPIOs; gpiochip number varies by kernel


# Linux GPIO character device, uAPI v2 (include/uapi/linux/gpio.h).
class _ChipInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 32), ("label", ctypes.c_char * 32), ("lines", ctypes.c_uint32)]


class _LineAttribute(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("padding", ctypes.c_uint32), ("value", ctypes.c_uint64)]


class _LineConfigAttribute(ctypes.Structure):
    _fields_ = [("attr", _LineAttribute), ("mask", ctypes.c_uint64)]


class _LineConfig(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("num_attrs", ctypes.c_uint32), ("padding", ctypes.c_uint32 * 5),
                ("attrs", _LineConfigAttribute * 10)]


class _LineRequest(ctypes.Structure):
    _fields_ = [("offsets", ctypes.c_uint32 * 64), ("consumer", ctypes.c_char * 32), ("config", _LineConfig),
                ("num_lines", ctypes.c_uint32), ("event_buffer_size", ctypes.c_uint32),
                ("padding", ctypes.c_uint32 * 5), ("fd", ctypes.c_int32)]


class _LineValues(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint64), ("mask", ctypes.c_uint64)]


def _iowr(nr, struct_type, read_only=False):
    return ((2 if read_only else 3) << 30) | (ctypes.sizeof(struct_type) << 16) | (0xB4 << 8) | nr


GPIO_GET_CHIPINFO_IOCTL = _iowr(0x01, _ChipInfo, read_only=True)
GPIO_V2_GET_LINE_IOCTL = _iowr(0x07, _LineRequest)
GPIO_V2_LINE_SET_VALUES_IOCTL = _iowr(0x0F, _LineValues)
GPIO_V2_LINE_FLAG_OUTPUT = 1 << 3


def find_gpio_chip(label=GPIO_CHIP_LABEL):
    """Return the /dev/gpiochipN path whose label matches."""
    labels = []
    for path in sorted(glob.glob("/dev/gpiochip*")):
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        try:
            info = _ChipInfo()
            fcntl.ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, info)
        finally:
            os.close(fd)
        if info.label.decode() == label:
            return path
        labels.append(f"{path}={info.label.decode()}")
    raise OSError(f"no GPIO chip labelled {label} (found: {', '.join(labels) or 'none'})")


class GpioOutputs:
    """A group of GPIO output lines, all set together with one bitmask (bit i = offsets[i])."""

    def __init__(self, chip_path, offsets, consumer):
        request = _LineRequest()
        for i, offset in enumerate(offsets):
            request.offsets[i] = offset
        request.num_lines = len(offsets)
        request.consumer = consumer.encode()[:31]
        request.config.flags = GPIO_V2_LINE_FLAG_OUTPUT  # lines start low
        chip = os.open(chip_path, os.O_RDWR | os.O_CLOEXEC)
        try:
            fcntl.ioctl(chip, GPIO_V2_GET_LINE_IOCTL, request)
        finally:
            os.close(chip)  # the line request keeps its own file descriptor
        self.fd = request.fd
        self.mask = (1 << len(offsets)) - 1

    def write(self, bits):
        fcntl.ioctl(self.fd, GPIO_V2_LINE_SET_VALUES_IOCTL, _LineValues(bits, self.mask))


class Stepper:
    """One 28BYJ-48 on a ULN2003 board, stepped by its own thread; set_speed() only sets the target."""

    def __init__(self, outputs, invert):
        self.outputs, self.invert = outputs, invert
        self.cond = threading.Condition()
        self.target = 0.0  # signed full steps/s
        self.phase = 0
        outputs.write(0)
        threading.Thread(target=self._run, daemon=True).start()

    def set_speed(self, speed):
        """speed from -1 to 1. Stopping de-energizes the coils immediately, not on the next step."""
        rate = 0.0 if abs(speed) < DEADBAND else speed * MAX_STEP_RATE * (-1 if self.invert else 1)
        with self.cond:
            self.target = rate
            if rate == 0:
                self.outputs.write(0)  # holding the coils on draws ~250 mA and heats the motor
            self.cond.notify()

    def _run(self):
        rate, next_step = 0.0, time.monotonic()
        while True:
            with self.cond:
                if self.target == 0:
                    rate = 0.0
                    self.cond.wait_for(lambda: self.target != 0)
                    next_step = time.monotonic()
                target = self.target
                if rate == 0 or (rate > 0) != (target > 0):
                    rate = math.copysign(min(START_RATE, abs(target)), target)  # start or reverse
                elif abs(rate) < abs(target):
                    rate = math.copysign(min(abs(target), abs(rate) + ACCEL / abs(rate)), target)
                else:
                    rate = target  # slowing down needs no ramp
                # Stepping happens under the lock, so a stop can't be overwritten by a late step.
                self.phase = (self.phase + (1 if rate > 0 else -1)) % len(STEPS)
                self.outputs.write(STEPS[self.phase])
            next_step += 1 / abs(rate)
            delay = next_step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.05:
                next_step = time.monotonic()  # fell well behind; don't try to catch up in a burst


class Motors:
    """All steppers in MOTORS. Prints each change; drives the motors when GPIO is reachable."""

    def __init__(self):
        self.speeds = None
        self.lock = threading.Lock()
        self.steppers = {}
        try:
            chip = find_gpio_chip()
            for name, motor in MOTORS.items():
                self.steppers[name] = Stepper(GpioOutputs(chip, motor["pins"], f"weedbot-{name}"), motor["invert"])
            print(f"motors on {chip}: {', '.join(self.steppers)}", flush=True)
        except OSError as e:
            print(f"no motor output, printing drive commands only: {e}", flush=True)

    def set(self, left, right, reason):
        """Tank driving: each motor runs at its side's speed."""
        speeds = {name: left if motor["side"] == "l" else right for name, motor in MOTORS.items()}
        self._apply(speeds, f"L={left:+.2f} R={right:+.2f}", reason)

    def set_each(self, speeds, reason):
        """Motor test: speeds is one speed per motor, in MOTORS order."""
        speeds = dict(zip(MOTORS, speeds))
        self._apply(speeds, " ".join(f"{name}={v:+.2f}" for name, v in speeds.items()), reason)

    def _apply(self, speeds, text, reason):
        with self.lock:  # called from the drive thread and from main() at exit
            if speeds == self.speeds:
                return  # held keys repeat the same command 10 times a second
            self.speeds = speeds
            for name, speed in speeds.items():
                if name in self.steppers:
                    self.steppers[name].set_speed(speed)
            label = "drive" if any(speeds.values()) else "STOP "
            print(f"{label}  {text}   ({reason})", flush=True)


def parse(packet):
    """Return (seq, speeds) from a packet: speeds is (left, right), or a list with one per motor."""
    try:
        msg = json.loads(packet)
        seq = int(msg["seq"])
        if "m" in msg:
            speeds = [float(v) for v in msg["m"]]
            if len(speeds) != len(MOTORS):
                raise ValueError(f"expected {len(MOTORS)} motor speeds")
        else:
            speeds = (float(msg["l"]), float(msg["r"]))
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"bad packet {packet[:80]!r}") from e
    if not all(math.isfinite(v) and -1 <= v <= 1 for v in speeds):
        raise ValueError(f"speeds out of range: {packet[:80]!r}")
    return seq, speeds


def drive_listener(port, motors):
    """Receive drive commands forever; stop the motors when they stop arriving."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            sock.bind(("0.0.0.0", port))
            break
        except OSError as e:  # e.g. the old drive_server.py still running
            print(f"waiting for UDP port {port}: {e.strerror}", flush=True)
            time.sleep(2)
    sock.settimeout(0.1)  # wake regularly to check the timeout
    print(f"listening for drive commands on UDP port {port}", flush=True)

    last_command = None  # time.monotonic() of the newest command; None once stopped by timeout
    while True:
        try:
            packet, sender = sock.recvfrom(256)
            seq, speeds = parse(packet)
        except socket.timeout:
            pass
        except ValueError as e:
            print(f"ignored packet from {sender[0]}: {e}", flush=True)
        else:
            reason = f"seq {seq} from {sender[0]}"
            if isinstance(speeds, list):
                motors.set_each(speeds, reason)
            else:
                motors.set(*speeds, reason)
            last_command = time.monotonic()
        if last_command is not None and time.monotonic() - last_command > DRIVE_TIMEOUT:
            motors.set(0, 0, f"no command for {DRIVE_TIMEOUT} s")
            last_command = None


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
    drive_port = int(sys.argv[3]) if len(sys.argv) > 3 else 9000

    motors = Motors()
    motors.set(0, 0, "starting")
    Handler.latest = LatestFrame()
    threading.Thread(target=reader, args=(port, Handler.latest), daemon=True).start()
    threading.Thread(target=drive_listener, args=(drive_port, motors), daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", http_port), Handler)
    server.daemon_threads = True
    print(f"serving on http://0.0.0.0:{http_port}/", flush=True)
    # systemd stops the service with SIGTERM; turn it into an exit so the motors stop below.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        motors.set(0, 0, "exiting")


if __name__ == "__main__":
    main()
