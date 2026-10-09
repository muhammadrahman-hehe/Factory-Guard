import io
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from pipeline import process_upload, DEFAULTS
from safety import EventBook


class EmptyAnalyzer:
    def __init__(self, options):
        self.events=EventBook()
        self.peak_score=0

    def process(self, frame, seconds):
        return dict(image=frame,people=0,forklifts=0,score=0,alerts=[],reviews=[])


class PipelineTests(unittest.TestCase):
    def test_outputs_and_history_commit(self):
        data=io.BytesIO()
        Image.new('RGB',(32,32)).save(data,format='PNG')
        with tempfile.TemporaryDirectory() as folder, patch('pipeline.ROOT',Path(folder)):
            result=process_upload(data.getvalue(),'image.png',analyzer_factory=EmptyAnalyzer,options=DEFAULTS)
            self.assertTrue(Path(result['path']).exists())
            report=json.loads(Path(result['report']).read_text())
            self.assertEqual(report['peak_score'],0)
            self.assertTrue((Path(result['report']).parent/'events.csv').exists())
            with closing(sqlite3.connect(Path(folder)/'outputs/history.sqlite3')) as db:
                self.assertEqual(db.execute('select count(*) from runs').fetchone()[0],1)

    def test_invalid_image_cleans_partial_run(self):
        with tempfile.TemporaryDirectory() as folder, patch('pipeline.ROOT',Path(folder)):
            with self.assertRaises(Exception):
                process_upload(b'bad','bad.png',analyzer_factory=EmptyAnalyzer,options=DEFAULTS)
            self.assertEqual(list((Path(folder)/'outputs').iterdir()),[])
