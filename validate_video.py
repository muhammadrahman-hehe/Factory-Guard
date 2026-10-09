"""Run actual specialist models on a short local video segment, saving evidence."""
import json
import sys
from pathlib import Path
import cv2
from engine import Analyzer

cap = cv2.VideoCapture(sys.argv[1])
options = dict(confidence=.35, size=640, forklift=True, ppe=True, margin=.35, gap=1.5, zone_enabled=True, zone=(.25, .3, .75, .95))
analyzer = Analyzer(options)
folder = Path('validation')
folder.mkdir(exist_ok=True)
records = []
try:
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0
    for seconds in [0, 1, 2, 3, 4]:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds*1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (min(w, 1280), round(h*min(w, 1280)/w)))
        result = analyzer.process(frame, seconds)
        cv2.imwrite(str(folder/f'frame-{seconds}.jpg'), result.pop('image'))
        records.append(dict(seconds=seconds, **result))
finally:
    cap.release()
assert records, 'No decodable video frames'
report = dict(source=Path(sys.argv[1]).name, samples=records, events=analyzer.events.finish())
(folder/'video-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps([{'seconds':r['seconds'], 'people':r['people'], 'forklifts':r['forklifts'], 'alerts':len(r['alerts'])} for r in records]))
