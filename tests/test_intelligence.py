import unittest
import numpy as np
from intelligence import Persistence, GroundDistance, MachineZones, risk_score


class IntelligenceTests(unittest.TestCase):
    def test_persistence_and_reset(self):
        p=Persistence(gap=1.)
        self.assertFalse(p.observe('fire',0,1))
        self.assertFalse(p.observe('fire',.5,1))
        self.assertTrue(p.observe('fire',1,1))
        self.assertFalse(p.observe('fire',3,1))

    def test_ground_distance(self):
        g=GroundDistance(dict(image_points=[[0,0],[1,0],[1,1],[0,1]],ground_points_m=[[0,0],[10,0],[10,10],[0,10]]))
        self.assertAlmostEqual(g.distance([0,0,20,50],[30,0,50,50],100,100),3.)
        self.assertIsNone(GroundDistance(None).distance([0,0,1,1],[1,1,2,2],100,100))

    def test_bad_calibration_rejected(self):
        with self.assertRaises(ValueError):
            GroundDistance(dict(image_points=[[0,0]]*4,ground_points_m=[[0,0]]*4))

    def test_activity_intervals_and_unknown(self):
        config=dict(id='press',roi=[0,0,1,1],danger_zone=[[0,0],[1,0],[1,1],[0,1]],activity_source='intervals',active_intervals=[[1,2]])
        machine=MachineZones([config])
        frame=np.zeros((20,20,3),np.uint8)
        self.assertFalse(machine.update(frame,0)[0]['active'])
        self.assertTrue(machine.update(frame,1)[0]['active'])
        self.assertFalse(machine.update(frame,2)[0]['active'])
        config['activity_source']='unknown'
        self.assertIsNone(machine.update(frame,1)[0]['active'])

    def test_risk_does_not_double_count_person(self):
        rows=[dict(person_id='P1',type='zone',severity='medium'),dict(person_id='P1',type='fall',severity='critical'),dict(person_id='',machine_id='H1',type='fire',severity='critical')]
        self.assertEqual(risk_score(rows),100)
