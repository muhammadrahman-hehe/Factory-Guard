import unittest
from unittest.mock import patch
import numpy as np
from engine import Analyzer
from pipeline import DEFAULTS


class SignalTests(unittest.TestCase):
    def test_all_detector_signals_reach_events(self):
        def detect(model, *args, **kwargs):
            name=str(model)
            if 'yolo26n' in name:
                return [dict(name='person',box=[20,20,100,200],confidence=.9)]
            if 'forklift' in name:
                return [dict(name='forklift',box=[80,20,200,240],confidence=.9)]
            if 'ppe_full' in name:
                return [dict(name='no-safety-vest',box=[30,60,90,130],confidence=.9)]
            if 'fire' in name:
                return [dict(name='fire',box=[250,250,300,300],confidence=.9)]
            if 'fall' in name:
                return [dict(name='fallen',box=[20,20,100,200],confidence=.9)]
            return []
        options={**DEFAULTS,'machines':[dict(id='press',roi=[0,0,1,1],danger_zone=[[0,0],[1,0],[1,1],[0,1]],activity_source='intervals',active_intervals=[[0,10]])]}
        with patch('engine.YOLO',side_effect=lambda p:p),patch('engine.detections',side_effect=detect):
            analyzer=Analyzer(options)
            image=np.zeros((400,400,3),np.uint8)
            first=analyzer.process(image,0)
            second=analyzer.process(image,1)
        kinds={a['type'] for a in second['alerts']}
        self.assertTrue({'missing_vest_candidate','active_machine_zone','proximity_review','fire_persistent_review','fallen_posture_persistent_review'} <= kinds)
        self.assertEqual(second['score'],100)
        self.assertNotIn('fire_persistent_review',{a['type'] for a in first['alerts']})
