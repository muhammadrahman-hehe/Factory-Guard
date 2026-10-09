import unittest
from safety import Tracker, EventBook, possible_operator, proximity
from engine import ppe_status


class InteractionTests(unittest.TestCase):
    def test_operator_is_only_geometric_candidate(self):
        self.assertTrue(possible_operator([30, 10, 60, 45], [0, 0, 100, 100]))
        self.assertFalse(possible_operator([110, 0, 140, 100], [0, 0, 100, 100]))

    def test_proximity_boundary_and_far_person(self):
        self.assertTrue(proximity([105, 20, 115, 100], [0, 0, 100, 100], .1))
        self.assertFalse(proximity([150, 20, 170, 100], [0, 0, 100, 100], .1))

    def test_tracking_expiry_and_one_to_one(self):
        tracker = Tracker()
        self.assertEqual(tracker.update([[0, 0, 20, 50]], 0), [1])
        self.assertEqual(tracker.update([[1, 0, 21, 50], [60, 0, 80, 50]], .5), [1, 2])
        self.assertEqual(tracker.update([[1, 0, 21, 50]], 3), [3])

    def test_brief_gap_bridged_long_gap_restarts(self):
        book = EventBook(1.5)
        alert = dict(type='restricted_zone', person_id='P1', machine_id='', severity='medium', detail='review')
        book.update(0, [alert])
        book.update(.5, [])
        book.update(1, [alert])
        book.update(3, [alert])
        events = book.finish()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['samples'], 2)

    def test_absent_ppe_unknown_and_ambiguous_assignment(self):
        people = [{'box': [0, 0, 100, 200]}]
        self.assertEqual(ppe_status(people, [])[0], {'helmet': 'unknown', 'vest': 'unknown'})
        helmet = {'box': [30, 0, 60, 30], 'name': 'no-helmet'}
        self.assertEqual(ppe_status(people, [helmet])[0]['helmet'], 'missing')
        self.assertEqual(ppe_status(people+people, [helmet])[0]['helmet'], 'unknown')
        present = dict(helmet, name='helmet')
        self.assertEqual(ppe_status(people, [helmet, present])[0]['helmet'], 'unknown')
