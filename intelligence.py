"""Temporal review signals and optional camera-specific machine rules."""
import math
import cv2
import numpy as np


class Persistence:
    def __init__(self, gap=1.):
        self.history = {}
        self.gap = gap

    def observe(self, key, seconds, minimum):
        start, last, count = self.history.get(key, (seconds, seconds, 0))
        if seconds-last > self.gap:
            start, count = seconds, 0
        self.history[key] = (start, seconds, count+1)
        return seconds-start >= minimum and count+1 >= 2


def polygon_contains(box, polygon, width, height):
    points = np.array([[x*width, y*height] for x,y in polygon], np.float32)
    return cv2.pointPolygonTest(points, ((box[0]+box[2])/2, box[3]), False) >= 0


class GroundDistance:
    def __init__(self, config):
        self.matrix = None
        if config:
            source = np.asarray(config['image_points'], np.float32)
            target = np.asarray(config['ground_points_m'], np.float32)
            if source.shape != (4,2) or target.shape != (4,2) or not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError('Ground calibration requires four finite image and four ground points.')
            if np.any(source < 0) or np.any(source > 1) or abs(cv2.contourArea(source)) < 1e-5 or abs(cv2.contourArea(target)) < 1e-5:
                raise ValueError('Calibration points must outline a non-degenerate ground quadrilateral; image points must be normalized.')
            self.matrix = cv2.getPerspectiveTransform(source, target)

    def distance(self, a, b, width, height):
        if self.matrix is None:
            return None
        points = np.array([[[(a[0]+a[2])/(2*width), a[3]/height], [(b[0]+b[2])/(2*width), b[3]/height]]], np.float32)
        mapped = cv2.perspectiveTransform(points, self.matrix)[0]
        value = float(np.linalg.norm(mapped[0]-mapped[1]))
        return value if math.isfinite(value) else None


class MachineZones:
    """Explicit activity intervals or visual motion evidence; neither inferred as certified state."""
    def __init__(self, machines):
        self.machines = machines
        self.previous = {}

    def update(self, frame, seconds):
        h,w = frame.shape[:2]
        output = []
        for machine in self.machines:
            name = machine['id']
            roi = machine['roi']
            x1,y1,x2,y2 = int(roi[0]*w),int(roi[1]*h),int(roi[2]*w),int(roi[3]*h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError(f'Invalid machine ROI: {name}')
            gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64,64))
            mode = machine.get('activity_source', 'unknown')
            active = None
            if mode == 'intervals':
                active = any(start <= seconds < end for start,end in machine.get('active_intervals', []))
            elif mode == 'visual_motion' and name in self.previous:
                delta = np.abs(gray.astype(float)-self.previous[name].astype(float))
                active = float((delta > 25).mean()) > machine.get('motion_fraction', .15)
            self.previous[name] = gray
            output.append(dict(id=name, active=active, evidence=mode, danger_zone=machine['danger_zone']))
        return output


def risk_score(alerts):
    weights = {'medium':15, 'high':30, 'critical':50}
    per_subject = {}
    for alert in alerts:
        key = alert['person_id'] or alert.get('machine_id') or alert['type']
        per_subject[key] = max(per_subject.get(key, 0), weights[alert['severity']])
    return min(100, sum(per_subject.values()))
