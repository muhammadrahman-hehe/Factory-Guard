"""Download pinned public specialist weights and record provenance."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
REPOS = {'forklift': 'keremberke/yolov8s-forklift-detection', 'ppe': 'melihuzunoglu/ppe-detection',
         'ppe_full': 'Hexmon/vyra-yolo-ppe-detection',
         'fire': 'rabahdev/fire-smoke-yolov8n', 'fall': 'melihuzunoglu/human-fall-detection'}


def main():
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, repo in REPOS.items():
        target = folder / f'{name}.pt'
        if target.exists() and name in manifest and hashlib.sha256(target.read_bytes()).hexdigest() == manifest[name].get('sha256'):
            print(f'{name}: already verified', flush=True)
            continue
        response = requests.get(f'https://huggingface.co/api/models/{repo}', timeout=30)
        response.raise_for_status()
        info = response.json()
        revision = info['sha']
        url = f'https://huggingface.co/{repo}/resolve/{revision}/best.pt'
        target = folder / f'{name}.pt'
        print(f'Downloading {name} from {repo}', flush=True)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            temp = target.with_suffix('.part')
            with temp.open('wb') as output:
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk)
            temp.replace(target)
        manifest[name] = {'repository': repo, 'revision': revision, 'url': url, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'license': info.get('cardData', {}).get('license', 'not specified; review upstream')}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Models ready.', flush=True)


if __name__ == '__main__':
    main()
