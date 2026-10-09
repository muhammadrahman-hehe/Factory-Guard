import unittest
from safety import EpisodeLog, in_zone


class SafetyTests(unittest.TestCase):
    def test_uses_feet_not_box_overlap(self):
        self.assertFalse(in_zone((0, 0, 60, 90), (.2, .2, .8, .8), 100, 100))
        self.assertTrue(in_zone((0, 0, 60, 80), (.2, .2, .8, .8), 100, 100))

    def test_episode_aggregates_and_restarts(self):
        log = EpisodeLog()
        self.assertIsNone(log.update(0, 0))
        self.assertIsNone(log.update(1, 1))
        self.assertIsNone(log.update(2, 3))
        self.assertEqual(log.update(3, 0), {'start_seconds': 1, 'last_seen_seconds': 2, 'peak_people': 3})
        log.update(4, 1)
        self.assertEqual(log.finish()['start_seconds'], 4)
        self.assertIsNone(log.finish())


if __name__ == '__main__':
    unittest.main()
