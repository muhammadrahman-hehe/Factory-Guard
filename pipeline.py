"""Upload processing and browser-playable video export, separate from the UI."""
from pathlib import Path
import json
import math
import shutil
import subprocess
import tempfile
import uuid
import csv
import sqlite3
from contextlib import closing
import cv2
import numpy as np
from PIL import Image, ImageOps
from engine import Analyzer

ROOT = Path(__file__).resolve().parent
DEFAULTS = dict(confidence=.35, size=640, forklift=True, ppe=True,
                full_ppe=True, fire=True, fall=True, camera_id=None, calibration=None, machines=[],
                forklift_confidence=.85, ppe_confidence=.6, margin=.35, gap=1.5,
                zone_enabled=False, zone=(.25, .3, .75, .95), samples_per_second=2, max_seconds=60)


def load_options():
    path = ROOT/'settings.json'
    return {**DEFAULTS, **(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {})}


def resize(frame):
    h, w = frame.shape[:2]
    scale = min(1., 1280/max(h, w))
    return cv2.resize(frame, (max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2)))


def overlay(result, seconds):
    """Put counts, risk indicator and review messages in a separate video footer."""
    frame = result['image']
    width = max(800, frame.shape[1])
    canvas = np.full((frame.shape[0]+110, width, 3), (32, 22, 13), dtype=np.uint8)
    x = (width-frame.shape[1])//2
    canvas[:frame.shape[0], x:x+frame.shape[1]] = frame
    alerts = sorted(set(a['type'].replace('_', ' ') for a in result['alerts']))
    lines = [f"FactoryGuard AI | {seconds:.1f}s | People: {result['people']} | Forklifts: {result['forklifts']} | Review score: {result['score']}/100",
             'Review: '+(', '.join(alerts) if alerts else 'No flags in this sample'),
             'Approximate proximity | Unknown PPE is not a violation | Score is uncalibrated']
    for i, line in enumerate(lines):
        scale = min(.52, (width-30)/max(1, cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
        cv2.putText(canvas, line, (12, frame.shape[0]+27+i*30), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 240, 245), 1, cv2.LINE_AA)
    return canvas


def process_upload(data, filename, progress=None, analyzer_factory=Analyzer, options=None):
    options = options or load_options()
    suffix = Path(filename).suffix.lower()
    video = suffix in ['.mp4', '.avi', '.mov']
    if suffix not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']:
        raise ValueError('Use a JPG, PNG, MP4, AVI or MOV file.')
    if len(data) > 100*1024**2:
        raise ValueError('Choose a file smaller than 100 MB.')
    ffmpeg = shutil.which('ffmpeg') if video else None
    if video and not ffmpeg:
        raise RuntimeError('FFmpeg is required for video export. Install it and restart the app.')
    analyzer = analyzer_factory(options)
    run_id = uuid.uuid4().hex[:12]
    output = ROOT/'outputs'/run_id
    output.mkdir(parents=True, exist_ok=True)
    records = []
    def analyze(frame, seconds):
        result = analyzer.process(resize(frame), seconds)
        rendered = overlay(result, seconds)
        records.append(dict(seconds=round(seconds, 3), **{k:v for k,v in result.items() if k!='image'}))
        return rendered
    try:
        if not video:
            import io
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
            rendered = analyze(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), 0)
            media = output/'result.jpg'
            if not cv2.imwrite(str(media), rendered):
                raise RuntimeError('Could not write the result image.')
        else:
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder)/('upload'+suffix)
                source.write_bytes(data)
                cap, writer = cv2.VideoCapture(str(source)), None
                raw = Path(folder)/'annotated.avi'
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not math.isfinite(fps) or fps <= 0:
                        raise ValueError('Cannot read the video frame rate. Try an H.264 MP4.')
                    step = max(1, round(fps/options['samples_per_second']))
                    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = min(options['max_seconds'], total/fps) if total > 0 else options['max_seconds']
                    index = 0
                    while index/fps < options['max_seconds']:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if index % step == 0:
                            rendered = analyze(frame, index/fps)
                            if writer is None:
                                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'MJPG'), fps/step, (rendered.shape[1], rendered.shape[0]))
                                if not writer.isOpened():
                                    raise RuntimeError('Cannot initialize video writer.')
                            writer.write(rendered)
                            if progress:
                                progress(min(.95, (index/fps)/max(1, duration)*.95))
                        index += 1
                finally:
                    cap.release()
                    if writer is not None:
                        writer.release()
                if not records:
                    raise ValueError('No video frames could be decoded.')
                media = output/'result.mp4'
                command = [ffmpeg, '-nostdin', '-y', '-loglevel', 'error', '-i', str(raw), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-t', str(min(index/fps, options['max_seconds'])), str(media)]
                encoded = subprocess.run(command, capture_output=True, text=True, timeout=180, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if encoded.returncode:
                    raise RuntimeError('Video export failed: '+encoded.stderr[-500:])
        events = analyzer.events.finish()
        counts = {}
        for event in events:
            key = (event['person_id'], event['machine_id'], event['type'])
            counts[key] = counts.get(key, 0)+1
        repeats = [dict(subject=k[0] or k[1], type=k[2], episodes=v) for k,v in counts.items() if v>1]
        report = dict(run_id=run_id, source=Path(filename).name, camera_id=options.get('camera_id'), options=options, samples=records, events=events, repeated_events=repeats, peak_score=analyzer.peak_score,
                      coverage={'people':True,'forklift':bool(options.get('forklift')),'ppe':bool(options.get('ppe')),'fire_smoke':bool(options.get('fire')),'fallen_posture':bool(options.get('fall')),'machine_zones':bool(options.get('machines')),'metric_distance':bool(options.get('calibration'))})
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        with (output/'events.csv').open('w', newline='', encoding='utf-8') as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=['type','person_id','machine_id','severity','detail','start_seconds','last_seen_seconds','samples'])
            writer_csv.writeheader()
            writer_csv.writerows(events)
        with closing(sqlite3.connect(ROOT/'outputs'/'history.sqlite3')) as database:
            database.execute('CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, camera_id TEXT, source TEXT, peak_score INTEGER, event_count INTEGER, report_path TEXT)')
            database.execute('INSERT INTO runs(run_id,camera_id,source,peak_score,event_count,report_path) VALUES (?,?,?,?,?,?)', (run_id, options.get('camera_id'), Path(filename).name, analyzer.peak_score, len(events), str(output/'report.json')))
            database.commit()
        if progress:
            progress(1.)
        return dict(path=str(media), video=video, report=str(output/'report.json'), samples=len(records))
    except Exception:
        # Only this newly-created run's known output files are removed.
        for name in ['result.jpg', 'result.mp4', 'report.json', 'events.csv']:
            (output/name).unlink(missing_ok=True)
        output.rmdir()
        raise
