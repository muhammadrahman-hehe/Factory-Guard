"""Zone rules independent of inference and UI."""
from dataclasses import dataclass
import math


def in_zone(box, zone, width, height):
    x = (box[0] + box[2]) / (2 * width)
    y = box[3] / height
    left, top, right, bottom = zone
    return left <= x <= right and top <= y <= bottom


@dataclass
class EpisodeLog:
    active: dict | None = None

    def update(self, seconds, count):
        completed = None
        if count and self.active is None:
            self.active = {'start_seconds': round(seconds, 2), 'last_seen_seconds': round(seconds, 2), 'peak_people': count}
        elif count:
            self.active['last_seen_seconds'] = round(seconds, 2)
            self.active['peak_people'] = max(count, self.active['peak_people'])
        elif self.active is not None:
            completed, self.active = self.active, None
        return completed

    def finish(self):
        completed, self.active = self.active, None
        return completed


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(1, area(a)+area(b)-overlap)


def possible_operator(person, forklift):
    """Geometric candidate only. Never use this to suppress safety alerts."""
    px = (person[0]+person[2])/2
    py = (person[1]+person[3])/2
    return (intersection(person, forklift)/max(1, area(person)) > .8
            and forklift[0] <= px <= forklift[2]
            and forklift[1] <= py <= forklift[1]+.7*(forklift[3]-forklift[1])
            and person[3]-person[1] < .7*(forklift[3]-forklift[1]))


def proximity(person, forklift, margin):
    """Footpoint to machine box distance, normalized by forklift width (not meters)."""
    x, y = (person[0]+person[2])/2, person[3]
    dx = max(forklift[0]-x, 0, x-forklift[2])
    dy = max(forklift[1]-y, 0, y-forklift[3])
    return math.hypot(dx, dy) / max(1, forklift[2]-forklift[0]) <= margin


class Tracker:
    """Short-lived per-clip IDs; no worker identity or cross-camera recognition."""
    def __init__(self, max_gap=1.5):
        self.tracks = {}
        self.next_id = 1
        self.max_gap = max_gap

    def update(self, boxes, seconds):
        self.tracks = {k: v for k, v in self.tracks.items() if seconds-v[1] <= self.max_gap}
        candidates = sorted(((iou(box, old[0]), idx, tid) for idx, box in enumerate(boxes)
                             for tid, old in self.tracks.items()), reverse=True)
        assigned, used = {}, set()
        for score, idx, tid in candidates:
            if score >= .2 and idx not in assigned and tid not in used:
                assigned[idx] = tid
                used.add(tid)
        for idx, box in enumerate(boxes):
            if idx not in assigned:
                assigned[idx] = self.next_id
                self.next_id += 1
            self.tracks[assigned[idx]] = (box, seconds)
        return [assigned[idx] for idx in range(len(boxes))]


class EventBook:
    """Group same-track alerts, bridging brief missed detections."""
    def __init__(self, gap=1.5):
        self.gap = gap
        self.active = {}
        self.events = []

    def update(self, seconds, alerts):
        for key in list(self.active):
            if seconds-self.active[key]['last_seen_seconds'] > self.gap:
                self.events.append(self.active.pop(key))
        for alert in alerts:
            key = (alert['type'], alert['person_id'], alert.get('machine_id', ''))
            if key not in self.active:
                self.active[key] = dict(alert, start_seconds=round(seconds, 2), last_seen_seconds=round(seconds, 2), samples=0)
            row = self.active[key]
            row['last_seen_seconds'] = round(seconds, 2)
            row['samples'] += 1
            row['detail'] = alert['detail']

    def finish(self):
        self.events.extend(self.active.values())
        self.active.clear()
        return sorted(self.events, key=lambda e: e['start_seconds'])
