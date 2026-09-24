#!/usr/bin/env python3
"""Browser viewer for YOLO labels and optional Ultralytics model predictions.

Run: .venv/bin/python scripts/view_yolo.py
Then open http://127.0.0.1:8000. Ctrl+C stops the server.
Browsing labels uses only the standard library. Run model requires ultralytics.
"""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlencode, urlsplit
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "mixed": PROJECT_ROOT / "training/runs/yolo26s-caw-acre-mixed-20260924-105511-105391/weights/best.pt",
    "baseline": PROJECT_ROOT / "training/runs/yolo26s-caw-full-baseline/weights/best.pt",
}
_models = {}
_model_lock = Lock()


def predict_boxes(image_path, checkpoint_name, confidence):
    """Run one checkpoint on one image and return normalized YOLO boxes."""
    if checkpoint_name not in CHECKPOINTS:
        raise ValueError("Unknown model")
    if not 0 < confidence <= 1 or not math.isfinite(confidence):
        raise ValueError("Confidence must be between 0 and 1")
    checkpoint = CHECKPOINTS[checkpoint_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Run the viewer with a Python environment containing ultralytics") from exc
    # Ultralytics inference uses shared model state; serialize load and predict calls.
    with _model_lock:
        if checkpoint_name not in _models:
            _models[checkpoint_name] = YOLO(str(checkpoint))
        result = _models[checkpoint_name].predict(
            source=str(image_path), conf=confidence, imgsz=640, save=False, verbose=False
        )[0]
        coordinates = result.boxes.xywhn.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
    return [[int(cls), *box, score] for cls, box, score in zip(classes, coordinates, scores)]

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crop and weed bounding boxes</title>
<style>
body{margin:0;background:#15191e;color:#eef2f6;font:15px system-ui}header{padding:16px 22px;background:#222933}h1{font-size:20px;margin:0 0 12px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}button,select,input{font:inherit;padding:6px;border-radius:5px;border:1px solid #637080;background:#303a47;color:white}button{cursor:pointer}input[type=number]{width:85px}#status{margin-top:12px;min-height:24px}#stage{overflow:auto;padding:16px;text-align:center}canvas{max-width:100%;height:auto;background:#000}#stage.zoom canvas{max-width:none}small{color:#bec9d5}.crop{color:#39ff88}.weed{color:#ffb340}label{white-space:nowrap}#error{color:#ff9696;white-space:pre-wrap}</style></head>
<body><header><h1>Crop and weed bounding boxes</h1><div class="controls">
<label>Dataset <select id="dataset"></select></label><label>Split <select id="split"><option>all</option><option>train</option><option>val</option><option>test</option><option value="train_subset" disabled>train_subset</option></select></label>
<button id="prev">← Previous</button><button id="next">Next →</button><button id="random">Random</button>
<label>Image <input id="index" type="number" min="1" value="1"></label><span id="total"></span>
<label>Model <select id="model"></select></label><label>Confidence <input id="confidence" type="number" min="0.01" max="1" step="0.05" value="0.25"></label><button id="runModel">Run model</button>
<label><input id="groundTruth" type="checkbox" checked>Ground truth (solid)</label><label><input id="predictions" type="checkbox" checked>Predictions (dashed)</label>
<label><input id="crop" type="checkbox" checked><span class="crop">Crop</span></label>
<label><input id="weed" type="checkbox" checked><span class="weed">Weed</span></label>
<label><input id="names" type="checkbox" checked>Class names</label>
<label><input id="zoom" type="checkbox">Original size</label></div>
<div id="status"></div><div id="predictionStatus"></div><div id="error" role="alert"></div><small>Solid boxes are labels; dashed boxes are model predictions. Arrow keys: previous/next · R: random image · B: toggle both classes. Images and labels are not edited.</small></header>
<div id="stage"><canvas id="canvas"></canvas></div>
<script>
const $=id=>document.getElementById(id);let files=[],position=0,picture=null,boxes=[],predicted=[],generation=0,subsetDatasets=[];
async function api(path){const r=await fetch(path);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
function query(extra={}){return new URLSearchParams({dataset:$('dataset').value,...extra});}
function error(e){$('error').textContent=e.message;}
function draw(){const c=$('canvas'),ctx=c.getContext('2d');if(!picture){ctx.clearRect(0,0,c.width,c.height);return;}c.width=picture.naturalWidth;c.height=picture.naturalHeight;ctx.drawImage(picture,0,0);const scale=Math.max(1,c.width/1100);ctx.lineWidth=2*scale;ctx.font=`bold ${14*scale}px system-ui`;
for(const [items,isPrediction] of [[boxes,false],[predicted,true]]){if(!$(isPrediction?'predictions':'groundTruth').checked)continue;ctx.setLineDash(isPrediction?[8*scale,5*scale]:[]);
for(const [cls,x,y,w,h,confidence] of items){if(!$(cls===0?'crop':'weed').checked)continue;const left=(x-w/2)*c.width,top=(y-h/2)*c.height,bw=w*c.width,bh=h*c.height,color=isPrediction?(cls===0?'#38ddff':'#ff69d1'):(cls===0?'#39ff88':'#ffb340');ctx.strokeStyle=color;ctx.strokeRect(left,top,bw,bh);if($('names').checked){const label=(isPrediction?'pred ':'GT ')+(cls===0?'crop':'weed')+(isPrediction?' '+confidence.toFixed(2):''),tw=ctx.measureText(label).width+8*scale,th=19*scale,lx=Math.min(left,c.width-tw),ly=Math.max(th,top);ctx.setLineDash([]);ctx.fillStyle=color;ctx.fillRect(lx,ly-th,tw,th);ctx.fillStyle='#111';ctx.fillText(label,lx+4*scale,ly-4*scale);ctx.setLineDash(isPrediction?[8*scale,5*scale]:[]);}}}ctx.setLineDash([]);}
async function show(){const token=++generation;picture=null;predicted=[];draw();$('predictionStatus').textContent='';$('error').textContent='';$('index').value=position+1;$('index').max=files.length;$('total').textContent=`/ ${files.length}`;if(!files.length){$('status').textContent='No images in this selection.';return;}
const name=files[position];$('status').textContent=`Loading ${name}…`;try{const data=await api('/api/labels?'+query({name}));const img=new Image();const ready=new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(Error('Could not decode image: '+name));});img.src='/image?'+query({name});await ready;if(token!==generation)return;picture=img;boxes=data.boxes;draw();const crops=boxes.filter(b=>b[0]===0).length;$('status').textContent=`${name} · ${img.naturalWidth} × ${img.naturalHeight} · ${crops} crop / ${boxes.length-crops} weed`;if(data.duplicates)$('error').textContent=`Source labels contain ${data.duplicates} duplicate row(s); overlapping boxes are drawn in the same place.`;}catch(e){if(token===generation){$('status').textContent=name;error(e);}}}
async function load(){const token=++generation;picture=null;draw();files=[];$('error').textContent='';try{const data=await api('/api/list?'+query({split:$('split').value}));if(token!==generation)return;files=data.files;position=0;await show();}catch(e){if(token===generation)error(e);}}
async function runModel(){if(!picture||!files.length)return;const token=generation,name=files[position],model=$('model').value,confidence=Number($('confidence').value);if(!(confidence>0&&confidence<=1)){error(Error('Confidence must be between 0 and 1.'));return;}$('runModel').disabled=true;$('predictionStatus').textContent='Running model…';$('error').textContent='';try{const data=await api('/api/predict?'+query({name,model,conf:confidence}));if(token!==generation)return;predicted=data.boxes;draw();$('predictionStatus').textContent=`${data.model}: ${predicted.length} prediction(s) at confidence ≥ ${confidence}`;}catch(e){if(token===generation){$('predictionStatus').textContent='';error(e);}}finally{$('runModel').disabled=false;}}
function move(delta){if(files.length){position=(position+delta+files.length)%files.length;show();}}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('random').onclick=()=>{if(files.length){position=Math.floor(Math.random()*files.length);show();}};
$('runModel').onclick=runModel;$('index').onchange=()=>{position=Math.max(0,Math.min(files.length-1,(parseInt($('index').value)||1)-1));show();};$('dataset').onchange=()=>{updateSubset();load();};$('split').onchange=load;for(const id of ['crop','weed','names','groundTruth','predictions'])$(id).onchange=draw;$('zoom').onchange=()=>$('stage').classList.toggle('zoom',$('zoom').checked);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}if(e.key==='ArrowRight'){e.preventDefault();move(1);}if(e.key.toLowerCase()==='r')$('random').click();if(e.key.toLowerCase()==='b'){const on=!($('crop').checked||$('weed').checked);$('crop').checked=on;$('weed').checked=on;draw();}});
function updateSubset(){const option=$('split').querySelector('[value="train_subset"]');option.disabled=!subsetDatasets.includes($('dataset').value);if(option.disabled&&$('split').value==='train_subset')$('split').value='all';}
api('/api/datasets').then(data=>{subsetDatasets=data.subset_datasets||[];for(const name of data.datasets){const o=document.createElement('option');o.value=name;o.textContent=name;$('dataset').appendChild(o);}for(const name of data.models){const o=document.createElement('option');o.value=name;o.textContent=name;$('model').appendChild(o);}if(data.datasets.includes('CAW'))$('dataset').value='CAW';$('split').value='val';updateSubset();return load();}).catch(error);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
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

    @staticmethod
    def child(parent, name):
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid file or dataset name')
        path = (parent / name).resolve()
        if not path.is_relative_to(parent.resolve()):
            raise ValueError('Path outside dataset')
        return path

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            if url.path == '/':
                self.respond(PAGE.encode(), 'text/html; charset=utf-8')
                return
            if url.path == '/api/datasets':
                names = sorted(p.name for p in self.root.iterdir()
                               if p.is_dir() and (p/'images').is_dir() and (p/'labels').is_dir())
                self.respond({'datasets': names, 'subset_datasets': [name for name in names
                              if (self.root/name/'train_subset.txt').is_file()],
                              'models': [name for name, path in CHECKPOINTS.items() if path.is_file()]})
                return
            if url.path not in ('/api/list', '/api/labels', '/api/predict', '/image'):
                self.respond({'error': 'Not found'}, status=404)
                return
            dataset = self.child(self.root, params.get('dataset', [''])[0])
            if not dataset.is_dir():
                raise FileNotFoundError('Dataset not found')
            image_dir, label_dir = dataset/'images', dataset/'labels'
            if url.path == '/api/list':
                split = params.get('split', ['all'])[0]
                if split == 'all':
                    files = sorted(p.name for p in image_dir.iterdir()
                                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
                elif split in ('train', 'val', 'test', 'train_subset'):
                    files = [Path(line.strip()).name for line in (dataset/f'{split}.txt').read_text().splitlines() if line.strip()]
                else:
                    raise ValueError('Invalid split')
                self.respond({'files': files})
                return
            name = params.get('name', [''])[0]
            path = self.child(image_dir, name)
            if not path.is_file():
                raise FileNotFoundError('Image not found')
            if url.path == '/api/predict':
                model_name = params.get('model', ['mixed'])[0]
                confidence = float(params.get('conf', ['0.25'])[0])
                self.respond({'model': model_name, 'boxes': predict_boxes(path, model_name, confidence)})
                return
            if url.path == '/image':
                mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(path.suffix.lower())
                if mime is None:
                    raise ValueError('Unsupported image format')
                self.respond(path.read_bytes(), mime)
                return
            label = self.child(label_dir, Path(name).with_suffix('.txt').name)
            boxes = []
            for n, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'{label.name}:{n}: expected class cx cy width height')
                cls = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if cls not in (0, 1) or not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
                    raise ValueError(f'{label.name}:{n}: invalid class or box')
                if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                    raise ValueError(f'{label.name}:{n}: box outside image')
                boxes.append([cls,x,y,w,h])
            self.respond({'boxes': boxes, 'duplicates': len(boxes)-len(set(map(tuple, boxes)))})
        except (ValueError, FileNotFoundError, KeyError) as exc:
            self.respond({'error': str(exc)}, status=400)
        except RuntimeError as exc:
            self.respond({'error': str(exc)}, status=500)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as exc:
            self.respond({'error': str(exc)}, status=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]/'data/YOLO_DATA', help='Parent of ACRE and CAW dataset directories')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f'Dataset root does not exist: {root}')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, root=root))
    except OSError as exc:
        parser.exit(1, f'Cannot start viewer: {exc}. Try --port 8001.\n')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Viewer: {url}\nDatasets: {root}\nPress Ctrl+C to stop.', flush=True)
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
