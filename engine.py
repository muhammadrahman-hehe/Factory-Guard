"""Local CPU inference and review-oriented safety rules."""
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from safety import Tracker, EventBook, in_zone, possible_operator, proximity, iou
from intelligence import Persistence, GroundDistance, MachineZones, polygon_contains, risk_score

ROOT = Path(__file__).resolve().parent


def canonical(name):
    return str(name).lower().replace('_', '-').replace(' ', '-')


def detections(model, frame, confidence, size, classes=None):
    result = model.predict(frame, device='cpu', conf=confidence, imgsz=size, classes=classes, iou=.4, verbose=False)[0]
    return [{'box': b.tolist(), 'name': canonical(result.names[int(c)]), 'confidence': float(s)}
            for b, c, s in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu())]


def ppe_status(persons, items):
    """Assign each PPE detection to at most one person. Absence is unknown."""
    states = [{'helmet': set(), 'vest': set()} for _ in persons]
    labels = {'helmet': ('helmet', 'present'), 'hardhat': ('helmet', 'present'),
              'no-helmet': ('helmet', 'missing'), 'no-hardhat': ('helmet', 'missing'),
              'vest': ('vest', 'present'), 'safety-vest': ('vest', 'present'),
              'no-vest': ('vest', 'missing'), 'no-safety-vest': ('vest', 'missing')}
    for item in items:
        if item['name'] not in labels:
            continue
        kind, state = labels[item['name']]
        b = item['box']
        x, y = (b[0]+b[2])/2, (b[1]+b[3])/2
        candidates = []
        for idx, person in enumerate(persons):
            p = person['box']
            h = max(1, p[3]-p[1])
            # Head and upper torso association only; overlapping people remain uncertain.
            if p[0] <= x <= p[2] and p[1]-.1*h <= y <= p[1]+(.45 if kind == 'helmet' else .85)*h:
                candidates.append(idx)
        if len(candidates) == 1:
            states[candidates[0]][kind].add(state)
    return [{k: next(iter(v)) if len(v) == 1 else 'unknown' for k, v in s.items()} for s in states]


class Analyzer:
    def __init__(self, options):
        torch.set_num_threads(2)
        self.options = options
        self.people = YOLO(str(ROOT/'yolo26n.pt'))
        self.forklift = self.load_optional('forklift')
        self.ppe = self.load_optional('ppe')
        self.fire = self.load_optional('fire')
        self.fall = self.load_optional('fall')
        self.hazard_tracker = Tracker()
        self.fall_tracker = Tracker()
        self.persistence = Persistence(gap=options.get('gap', 1.5))
        self.ground = GroundDistance(options.get('calibration'))
        self.machine_zones = MachineZones(options.get('machines', []))
        self.person_tracker = Tracker()
        self.machine_tracker = Tracker()
        self.events = EventBook(gap=options.get('gap', 1.5))
        self.peak_score = 0

    def load_optional(self, name):
        if not self.options.get(name):
            return None
        filename = 'ppe_full' if name == 'ppe' and self.options.get('full_ppe', True) else name
        path = ROOT/'models'/f'{filename}.pt'
        if not path.exists():
            raise FileNotFoundError(f'{name} model is missing. Run setup_models.py or disable this check.')
        return YOLO(str(path))

    def process(self, frame, seconds):
        o = self.options
        people = detections(self.people, frame, o['confidence'], o['size'], [0])
        machines = [d for d in detections(self.forklift, frame, o.get('forklift_confidence', .85), o['size']) if d['name'] == 'forklift'] if self.forklift else []
        gear = detections(self.ppe, frame, o.get('ppe_confidence', .6), o['size']) if self.ppe else []
        hazards = [d for d in detections(self.fire, frame, o.get('fire_confidence', .6), o['size']) if d['name'] in ['fire','smoke']] if self.fire else []
        postures = detections(self.fall, frame, o.get('fall_confidence', .65), o['size']) if self.fall else []
        states = ppe_status(people, gear)
        pids = self.person_tracker.update([p['box'] for p in people], seconds)
        mids = self.machine_tracker.update([m['box'] for m in machines], seconds)
        canvas = frame.copy()
        h, w = frame.shape[:2]
        zone_count = 0
        alerts, reviews = [], []
        machine_states = self.machine_zones.update(frame, seconds)
        for machine in machine_states:
            polygon = machine['danger_zone']
            points = np.array([[int(x*w),int(y*h)] for x,y in polygon], dtype='int32')
            cv2.polylines(canvas, [points], True, (30,150,250), 2)
            label = machine['id'] + (' | ACTIVE' if machine['active'] else ' | inactive' if machine['active'] is False else ' | state unknown')
            if machine['evidence'] == 'visual_motion':
                label = machine['id'] + ' | visual motion only'
            self.draw(canvas, [points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()], label, (30,150,250), 1)
        if o['zone_enabled']:
            z = o['zone']
            a, b = (int(z[0]*w), int(z[1]*h)), (int(z[2]*w), int(z[3]*h))
            overlay = canvas.copy()
            cv2.rectangle(overlay, a, b, (30, 150, 250), -1)
            canvas = cv2.addWeighted(overlay, .12, canvas, .88, 0)
            cv2.rectangle(canvas, a, b, (30, 150, 250), 2)
        for m, mid in zip(machines, mids):
            self.draw(canvas, m['box'], f'Forklift F{mid}', (255, 180, 60))
        for idx, (person, pid) in enumerate(zip(people, pids)):
            box = person['box']
            inside = o['zone_enabled'] and in_zone(box, o['zone'], w, h)
            zone_count += int(inside)
            operator = any(possible_operator(box, m['box']) for m in machines)
            role = 'Possible operator' if operator else 'Person'
            detail = 'Operator status uncertain; review required.' if operator else 'Review detected event.'
            def add(kind, severity, machine=''):
                alerts.append(dict(type=kind, person_id=f'P{pid}', machine_id=machine, severity=severity, detail=detail))
            if inside:
                add('restricted_zone', 'medium')
            for machine, mid in zip(machines, mids):
                distance = self.ground.distance(box, machine['box'], w, h) if not operator else None
                near = distance < o.get('safe_distance_m', 2.) if distance is not None else proximity(box, machine['box'], o['margin'])
                if near:
                    add('distance_review' if distance is not None else 'proximity_review', 'high', f'F{mid}')
                    if distance is not None:
                        alerts[-1]['detail'] = f'Estimated ground-point distance {distance:.2f} m; calibration-dependent.'
            for machine in machine_states:
                if machine['active'] and polygon_contains(box, machine['danger_zone'], w, h):
                    add('active_machine_zone' if machine['evidence'] == 'intervals' else 'machine_motion_zone_review', 'high', machine['id'])
            if self.ppe:
                for kind, state in states[idx].items():
                    if state == 'missing':
                        add(f'missing_{kind}_candidate', 'medium')
            reviews.append(dict(person_id=f'P{pid}', role=role, helmet=states[idx]['helmet'] if self.ppe else 'not evaluated', vest=states[idx]['vest'] if self.ppe else 'not evaluated', in_zone=bool(inside)))
            label = f'{role} P{pid}' + (' | ZONE' if inside else '')
            if self.ppe:
                label += f" | H:{states[idx]['helmet']} V:{states[idx]['vest']}"
            self.draw(canvas, box, label, (80, 80, 255) if any(a['person_id']==f'P{pid}' for a in alerts) else (170, 210, 40))
        for item in gear:
            if item['name'] in ['helmet', 'hardhat', 'vest', 'safety-vest']:
                self.draw(canvas, item['box'], item['name'], (220, 220, 60), 1)
        hids = self.hazard_tracker.update([d['box'] for d in hazards], seconds)
        for detection, hid in zip(hazards, hids):
            name = detection['name']
            persistent = self.persistence.observe((name, hid), seconds, o.get('hazard_persistence_seconds', 1.))
            kind = f'{name}_persistent_review' if persistent else f'{name}_candidate'
            alerts.append(dict(type=kind, person_id='', machine_id=f'H{hid}', severity='critical' if persistent else 'high', detail='Visual detector signal; review footage.'))
            self.draw(canvas, detection['box'], kind.replace('_',' '), (30,100,255))
        fallen = [d for d in postures if d['name'] in ['fallen','fall','fall-detected']]
        fids = self.fall_tracker.update([d['box'] for d in fallen], seconds)
        for detection, fid in zip(fallen, fids):
            best = max(((iou(detection['box'],p['box']), pid) for p,pid in zip(people,pids)), default=(0,None))
            subject = f'P{best[1]}' if best[0] >= .2 else f'FallTrack{fid}'
            persistent = self.persistence.observe(('fall',fid), seconds, o.get('fall_persistence_seconds', 1.))
            kind = 'fallen_posture_persistent_review' if persistent else 'fallen_posture_candidate'
            alerts.append(dict(type=kind, person_id=subject, machine_id='', severity='critical' if persistent else 'high', detail='Fallen/lying posture; does not prove a fall transition.'))
            self.draw(canvas, detection['box'], 'Possible fall | review', (40,60,255))
        self.events.update(seconds, alerts)
        # One severity contribution per person, avoiding overlapping-rule double counting.
        score = risk_score(alerts)
        self.peak_score = max(self.peak_score, score)
        return dict(image=canvas, people=len(people), forklifts=len(machines), zone=zone_count, alerts=alerts, reviews=reviews, score=score, machine_states=machine_states, fire_smoke=len(hazards), possible_falls=len(fallen))

    @staticmethod
    def draw(canvas, box, text, color, thickness=2):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        scale = min(.48, (canvas.shape[1]-12)/max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        tx = max(3, min(x1, canvas.shape[1]-tw-3))
        cv2.putText(canvas, text, (tx, max(18, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
