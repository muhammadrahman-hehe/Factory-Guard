"""Exercise the upload-only interface with real inference."""
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest
from ultralytics.utils import ASSETS

app = AppTest.from_file('app.py', default_timeout=180).run()
assert not app.exception, app.exception
assert not app.slider and not app.checkbox and not app.radio
payload = (ASSETS/'bus.jpg').read_bytes()
upload = SimpleNamespace(name='sample.jpg', getvalue=lambda:payload)
with patch('streamlit.file_uploader', return_value=upload):
    app.run(timeout=180)
    assert not app.exception, app.exception
    assert not app.error, app.error
    result = app.session_state['result']
    assert Path(result['path']).exists()
    assert result['samples'] == 1
    app.run(timeout=180)
    assert app.session_state['result']['path'] == result['path']
print('Upload-only UI, real image inference, and rerun caching passed')
