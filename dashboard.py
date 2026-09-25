#!/usr/bin/env python3
"""Browser dashboard for the robot's live camera stream, with optional crop/weed detection.

Run: .venv/bin/python dashboard.py   (laptop joined to the weedbot hotspot)
Then open http://127.0.0.1:8080. Ctrl+C stops the server.
The stream comes from pi/pi_server.py (http://10.42.0.1:8000/stream); the Pi does no processing.
WASD drive commands go to pi/pi_server.py as UDP packets (port 9000 on the same host).
Viewing uses only the standard library. Detection requires ultralytics and runs on this machine.
"""
import argparse
import base64
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
from threading import Condition, Lock, Thread
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "mixed": PROJECT_ROOT / "training/runs/yolo26s-caw-acre-mixed-20260924-105511-105391/weights/best.pt",
    "baseline": PROJECT_ROOT / "training/runs/yolo26s-caw-full-baseline/weights/best.pt",
}
BOUNDARY = "frame"


class Dashboard:
    """Shared state: newest camera frame, stream health, detection settings and newest result."""

    def __init__(self, stream_url, device, drive_address):
        self.stream_url = stream_url
        self.drive_address = drive_address
        self.drive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.drive_seq = 0
        self.device = device
        self.lock = Lock()
        self.new_frame = Condition(self.lock)
        self.jpeg, self.seq, self.frame_time = None, 0, 0.0
        self.fps, self.interval, self.stream_error = 0.0, 0.0, "connecting"
        self.detect, self.model_name, self.confidence = False, "mixed", 0.25
        self.detect_status, self.result = "off", None
        self.models = {}

    def put_frame(self, jpeg):
        with self.lock:
            now = time.monotonic()
            if self.frame_time:
                # Average the interval, not the rate: bunched frames would inflate an averaged rate.
                interval = now - self.frame_time
                self.interval = interval if not self.interval else 0.9 * self.interval + 0.1 * interval
                self.fps = 1 / max(self.interval, 1e-3)
            self.jpeg, self.frame_time, self.stream_error = jpeg, now, None
            self.seq += 1
            self.new_frame.notify_all()

    def next_frame(self, after_seq, timeout=1.0):
        """Return (seq, jpeg) newer than after_seq, or (after_seq, None) on timeout."""
        with self.lock:
            self.new_frame.wait_for(lambda: self.seq > after_seq, timeout)
            return (self.seq, self.jpeg) if self.seq > after_seq else (after_seq, None)

    def send_drive(self, left, right):
        """Send one drive command to the Pi; returns its sequence number."""
        with self.lock:
            self.drive_seq += 1
            seq = self.drive_seq
        packet = json.dumps({'seq': seq, 'l': round(left, 3), 'r': round(right, 3)}).encode()
        self.drive_socket.sendto(packet, self.drive_address)
        return seq

    def status(self):
        with self.lock:
            age = time.monotonic() - self.frame_time if self.frame_time else None
            live = age is not None and age < 2
            return {
                "stream": self.stream_url, "live": live, "fps": round(self.fps, 1) if live else 0,
                "frame_bytes": len(self.jpeg) if self.jpeg and live else 0,
                "error": None if live else (self.stream_error or "no frames for %.0f s" % age),
                "detect": self.detect, "model": self.model_name, "confidence": self.confidence,
                "detect_status": self.detect_status,
                "models": [name for name, path in CHECKPOINTS.items() if path.is_file()],
                "result_seq": self.result["seq"] if self.result else 0,
            }


def read_stream(state):
    """Read the Pi's MJPEG stream forever, reconnecting when it drops."""
    while True:
        try:
            with urlopen(state.stream_url, timeout=5) as response:
                while True:
                    line = response.readline()
                    if not line:
                        raise ConnectionError("stream ended")
                    if not line.startswith(b"--" + BOUNDARY.encode()):
                        continue
                    length = None
                    while (header := response.readline().strip()):
                        name, _, value = header.partition(b":")
                        if name.strip().lower() == b"content-length":
                            length = int(value)
                    if length:
                        state.put_frame(response.read(length))
        except (OSError, ValueError, ConnectionError) as exc:
            with state.lock:
                state.stream_error = f"cannot read {state.stream_url}: {getattr(exc, 'reason', exc)}"
            time.sleep(1)


def run_detection(state):
    """Run the selected model on the newest frame whenever detection is on; skips frames it can't keep up with."""
    done = 0
    while True:
        with state.lock:
            enabled, model_name, confidence = state.detect, state.model_name, state.confidence
        if not enabled:
            time.sleep(0.1)
            continue
        seq, jpeg = state.next_frame(done)
        if jpeg is None:
            continue
        try:
            # Imported here so viewing works without ultralytics installed.
            import cv2
            import numpy as np
            from ultralytics import YOLO
            if model_name not in state.models:
                with state.lock:
                    state.detect_status = f"loading {model_name}…"
                state.models[model_name] = YOLO(str(CHECKPOINTS[model_name]))
            if state.device == 'auto':
                import torch
                state.device = 0 if torch.cuda.is_available() else 'cpu'
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                done = seq
                continue
            start = time.monotonic()
            result = state.models[model_name].predict(
                source=image, conf=confidence, imgsz=640, device=state.device, save=False, verbose=False
            )[0]
            elapsed = time.monotonic() - start
            coordinates = result.boxes.xywhn.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            boxes = [[int(cls), *box, score] for cls, box, score in zip(classes, coordinates, scores)]
            with state.lock:
                state.result = {"seq": seq, "jpeg": jpeg, "boxes": boxes, "model": model_name,
                                "confidence": confidence, "ms": round(elapsed * 1000)}
                state.detect_status = f"{model_name} · {elapsed * 1000:.0f} ms per frame on {'GPU ' + str(state.device) if isinstance(state.device, int) else state.device}"
            done = seq
        except ImportError:
            with state.lock:
                state.detect, state.detect_status = False, "error: run with a Python environment containing ultralytics"
        except Exception as exc:  # keep the thread alive; show the problem in the page
            with state.lock:
                state.detect, state.detect_status = False, f"error: {exc}"


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weedbot dashboard</title>
<style>
body{margin:0;background:#15191e;color:#eef2f6;font:15px system-ui}header{padding:16px 22px;background:#222933}h1{font-size:20px;margin:0 0 12px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}select,input,button{font:inherit;padding:6px;border-radius:5px;border:1px solid #637080;background:#303a47;color:white}input[type=number]{width:85px}input[type=range]{vertical-align:middle;padding:0}button{cursor:pointer}#drive{font-variant-numeric:tabular-nums;color:#bec9d5}#drive.moving{color:#39ff88}label{white-space:nowrap}
#status{margin-top:12px;min-height:24px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff6b6b;margin-right:6px}.dot.live{background:#39ff88}#error{color:#ff9696;white-space:pre-wrap}small{color:#bec9d5}.crop{color:#39ff88}.weed{color:#ffb340}
main{padding:16px;max-width:1100px;margin:0 auto}[hidden]{display:none!important}section{background:#1d232b;border-radius:8px;padding:12px}h2{display:inline-block;font-size:16px;margin:0 0 10px;font-weight:600}h2 span{font-weight:400;color:#bec9d5}img,canvas{display:block;max-width:100%;max-height:calc(100vh - 230px);width:auto;height:auto;margin:0 auto;background:#000;border-radius:4px}</style></head>
<body><header><h1>Weedbot dashboard</h1><div class="controls">
<label><input id="detect" type="checkbox">Detection</label><label>Model <select id="model"></select></label><label>Confidence <input id="confidence" type="number" min="0.01" max="1" step="0.05" value="0.25"></label>
<label><input id="crop" type="checkbox" checked><span class="crop">Crop</span></label><label><input id="weed" type="checkbox" checked><span class="weed">Weed</span></label><label><input id="names" type="checkbox" checked>Labels</label>
<label>Speed <input id="speed" type="range" min="0.1" max="1" step="0.1" value="0.5"></label><button id="stop" type="button">Stop</button><span id="drive">Stopped</span></div>
<div id="status"><span class="dot" id="dot"></span><span id="streamStatus">Connecting…</span></div><div id="error" role="alert"></div>
<small>Detection off: live camera. Detection on: each frame the model finished, with its boxes; it updates as fast as detection runs. Drive: W/S forward/back, A/D turn, Space stop. T: toggle detection.</small></header>
<main><section><h2 id="title">Live camera</h2> <h2><span id="viewStatus"></span></h2><img id="live" src="/stream" alt="Live camera stream"><canvas id="canvas" width="640" height="480" hidden></canvas></section></main>
<script>
const $=id=>document.getElementById(id);let resultSeq=0,picture=null,boxes=[],detectStatus='off',size='';
async function api(path){const r=await fetch(path);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
// Drive: held keys -> left/right speeds. Commands repeat every 100 ms while a key is held; the Pi stops the motors after 0.5 s without one.
const DRIVE_KEYS=new Set(['w','a','s','d']),held=new Set();let sending=false,pending=false;
function driveValues(){const throttle=held.has('w')-held.has('s'),turn=held.has('d')-held.has('a'),speed=Number($('speed').value),clamp=x=>Math.max(-1,Math.min(1,x));return [clamp(throttle+turn)*speed,clamp(throttle-turn)*speed];}
async function drive(){pending=true;if(sending)return;sending=true;while(pending){pending=false;const [l,r]=driveValues();const moving=l!==0||r!==0;$('drive').textContent=moving?`L ${l.toFixed(2)}  R ${r.toFixed(2)}`:'Stopped';$('drive').classList.toggle('moving',moving);
try{await api('/api/drive?'+new URLSearchParams({l:l.toFixed(3),r:r.toFixed(3)}));}catch(e){$('error').textContent=e.message;}}sending=false;}
function stop(){held.clear();drive();}
function draw(){const c=$('canvas'),ctx=c.getContext('2d');if(!picture){ctx.fillStyle='#000';ctx.fillRect(0,0,c.width,c.height);return;}c.width=picture.naturalWidth;c.height=picture.naturalHeight;ctx.drawImage(picture,0,0);const scale=Math.max(1,c.width/900);ctx.lineWidth=2*scale;ctx.font=`bold ${14*scale}px system-ui`;
for(const [cls,x,y,w,h,score] of boxes){if(!$(cls===0?'crop':'weed').checked)continue;const left=(x-w/2)*c.width,top=(y-h/2)*c.height,color=cls===0?'#39ff88':'#ffb340';ctx.strokeStyle=color;ctx.strokeRect(left,top,w*c.width,h*c.height);if($('names').checked){const label=(cls===0?'crop ':'weed ')+score.toFixed(2),tw=ctx.measureText(label).width+8*scale,th=19*scale,lx=Math.min(left,c.width-tw),ly=Math.max(th,top);ctx.fillStyle=color;ctx.fillRect(lx,ly-th,tw,th);ctx.fillStyle='#111';ctx.fillText(label,lx+4*scale,ly-4*scale);}}}
function view(){const detecting=$('detect').checked&&picture!==null;$('live').hidden=detecting;$('canvas').hidden=!detecting;$('title').textContent=detecting?'Detections':'Live camera';if(detecting){const crops=boxes.filter(b=>b[0]===0).length;$('viewStatus').textContent=`${detectStatus} · ${crops} crop / ${boxes.length-crops} weed`;}else $('viewStatus').textContent=[size,$('detect').checked?detectStatus:''].filter(Boolean).join(' · ');}
async function loadResult(){const data=await api('/api/result');if(!data.seq||data.seq===resultSeq)return;const img=new Image();await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(Error('Could not decode detection frame'));img.src='data:image/jpeg;base64,'+data.jpeg;});resultSeq=data.seq;picture=img;boxes=data.boxes;detectStatus=data.detect_status;draw();view();}
async function poll(){try{const s=await api('/api/status');$('error').textContent='';$('dot').classList.toggle('live',s.live);$('streamStatus').textContent=s.live?`Live · ${s.fps} fps · ${(s.frame_bytes/1024).toFixed(0)} KB/frame · ${s.stream}`:`No video · ${s.stream}`;if(s.error)$('error').textContent=s.error;
if(!$('model').options.length){for(const name of s.models){const o=document.createElement('option');o.value=name;o.textContent=name;$('model').appendChild(o);}$('model').value=s.model;$('confidence').value=s.confidence;}
if(document.activeElement!==$('detect')&&$('detect').checked!==s.detect){$('detect').checked=s.detect;picture=null;}if(!s.detect||s.result_seq===resultSeq)detectStatus=s.detect_status;view();if(s.detect&&s.result_seq!==resultSeq)await loadResult();}catch(e){$('dot').classList.remove('live');$('streamStatus').textContent='Dashboard server not responding';$('error').textContent=e.message;}}
async function settings(){const confidence=Number($('confidence').value);if(!(confidence>0&&confidence<=1)){$('error').textContent='Confidence must be between 0 and 1.';return;}try{await api('/api/settings?'+new URLSearchParams({detect:$('detect').checked?1:0,model:$('model').value,conf:confidence}));}catch(e){$('error').textContent=e.message;}}
$('detect').onchange=()=>{picture=null;view();settings();};document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT'||e.target.type==='number')return;const k=e.key.toLowerCase();if(DRIVE_KEYS.has(k)){e.preventDefault();if(!held.has(k)){held.add(k);drive();}}else if(k===' '){e.preventDefault();stop();}else if(k==='t'){$('detect').checked=!$('detect').checked;$('detect').onchange();}});
document.addEventListener('keyup',e=>{if(held.delete(e.key.toLowerCase()))drive();});window.addEventListener('blur',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});$('stop').onclick=stop;$('speed').onchange=()=>{if(held.size)drive();};
setInterval(()=>{if(held.size)drive();},100);$('model').onchange=settings;$('confidence').onchange=settings;for(const id of ['crop','weed','names'])$(id).onchange=draw;
$('live').onload=()=>{size=`${$('live').naturalWidth} × ${$('live').naturalHeight}`;view();};$('live').onerror=()=>setTimeout(()=>{$('live').src='/stream?'+Date.now();},1000);
draw();poll();setInterval(poll,250);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *_args):
        pass

    def respond(self, data, content_type='application/json', status=200):
        body = json.dumps(data).encode() if content_type == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def stream(self):
        """Re-serve the newest frames as MJPEG; a slow browser skips frames instead of lagging."""
        self.send_response(200)
        self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={BOUNDARY}')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        seq = 0
        while True:
            seq, jpeg = self.state.next_frame(seq)
            if jpeg is None:
                continue
            self.wfile.write(f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n'.encode())
            self.wfile.write(jpeg)
            self.wfile.write(b'\r\n')

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            if url.path == '/':
                self.respond(PAGE.encode(), 'text/html; charset=utf-8')
            elif url.path == '/stream':
                self.stream()
            elif url.path == '/api/status':
                self.respond(self.state.status())
            elif url.path == '/api/result':
                with self.state.lock:
                    result, detect_status = self.state.result, self.state.detect_status
                if result is None:
                    self.respond({'seq': 0})
                else:
                    self.respond({**{k: v for k, v in result.items() if k != 'jpeg'},
                                  'jpeg': base64.b64encode(result['jpeg']).decode(), 'detect_status': detect_status})
            elif url.path == '/api/settings':
                model_name = params.get('model', [self.state.model_name])[0]
                confidence = float(params.get('conf', [self.state.confidence])[0])
                if model_name not in CHECKPOINTS or not CHECKPOINTS[model_name].is_file():
                    raise ValueError('Unknown model')
                if not 0 < confidence <= 1 or not math.isfinite(confidence):
                    raise ValueError('Confidence must be between 0 and 1')
                with self.state.lock:
                    self.state.detect = params.get('detect', ['0'])[0] == '1'
                    self.state.model_name, self.state.confidence = model_name, confidence
                    self.state.detect_status = 'starting…' if self.state.detect else 'off'
                self.respond({'ok': True})
            elif url.path == '/api/drive':
                left, right = (float(params.get(key, ['0'])[0]) for key in ('l', 'r'))
                if not all(math.isfinite(v) and -1 <= v <= 1 for v in (left, right)):
                    raise ValueError('Drive speeds must be between -1 and 1')
                try:
                    seq = self.state.send_drive(left, right)
                except OSError as exc:  # e.g. laptop not on the hotspot
                    self.respond({'error': f'Cannot send drive command: {exc}'}, status=502)
                else:
                    self.respond({'ok': True, 'seq': seq})
            else:
                self.respond({'error': 'Not found'}, status=404)
        except ValueError as exc:
            self.respond({'error': str(exc)}, status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pi', default='http://10.42.0.1:8000/stream', help='MJPEG stream served by pi/pi_server.py')
    parser.add_argument('--drive-port', type=int, default=9000, help='UDP drive port of pi/pi_server.py, on the --pi host')
    parser.add_argument('--port', type=int, default=8080)
    # The GTX 1050 (sm_61) needs a CUDA 12.6 torch build; CUDA 12.8+ builds dropped Pascal.
    parser.add_argument('--device', default='auto', help='Inference device: auto (first GPU if available, else cpu), cpu, 0, ...')
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device

    state = Dashboard(args.pi, device, (urlsplit(args.pi).hostname, args.drive_port))
    Thread(target=read_stream, args=(state,), daemon=True).start()
    Thread(target=run_detection, args=(state,), daemon=True).start()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, state=state))
    except OSError as exc:
        parser.exit(1, f'Cannot start dashboard: {exc}. Try --port 8081.\n')
    server.daemon_threads = True
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Dashboard: {url}\nCamera: {args.pi}\nDrive: udp://{state.drive_address[0]}:{args.drive_port}\nPress Ctrl+C to stop.', flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
