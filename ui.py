"""Minimal upload -> annotated media interface."""
import hashlib
from pathlib import Path
import streamlit as st
from pipeline import process_upload, ROOT


def main():
    st.set_page_config(page_title='FactoryGuard AI', page_icon='🛡️', layout='wide')
    st.title('FactoryGuard AI')
    st.write('Upload a video or image. Detection results appear directly on it.')
    upload = st.file_uploader('Upload video or image', type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])
    st.caption('Up to 100 MB · First 60 seconds · Video output is sampled and silent')
    if upload is None:
        st.session_state.pop('result', None)
        st.session_state.pop('upload_key', None)
        return
    data = upload.getvalue()
    backend_signature = b''.join((ROOT/name).read_bytes() for name in ['settings.json', 'engine.py', 'intelligence.py', 'pipeline.py'])
    key = hashlib.sha256(upload.name.encode()+data+backend_signature).hexdigest()
    if key != st.session_state.get('upload_key'):
        st.session_state.pop('result', None)
        bar = st.progress(0, text='Processing your upload…')
        try:
            with st.spinner('Analyzing safety conditions…'):
                result = process_upload(data, upload.name, progress=bar.progress)
            st.session_state.result = result
            st.session_state.upload_key = key
        except Exception as exc:
            st.error(str(exc))
            return
        finally:
            bar.empty()
    result = st.session_state.get('result')
    if result:
        path = Path(result['path'])
        if result['video']:
            st.video(str(path))
        else:
            st.image(str(path), width='stretch')
        st.download_button('Download result', path.read_bytes(), file_name='factoryguard-result'+path.suffix, mime='video/mp4' if result['video'] else 'image/jpeg')
