"""Download the pinned, MIT-licensed official DynaIP release. No import on download."""
import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

REVISION = 'ddfff5b88b77e124744bc74a792967cb8bad6735'
CHECKPOINT_SHA256 = '4d30d5bd0898dddfa6f8ba4c76c444d1a290dbc1fc4c342bb606e8799c59faca'
ARCHIVE_SHA256 = 'b5bfeb0b6d36f1f4e19da184bee971d3173ad0f93660e3a7c39254832794c37f'
PROJECT = Path(__file__).resolve().parents[2]

def fetch(destination):
    url = f'https://codeload.github.com/dx118/dynaip/zip/{REVISION}'
    with urlopen(Request(url, headers={'User-Agent': 'MetaFit-research/1.0'}), timeout=60) as response:
        archive = response.read(80_000_001)
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('official source archive digest mismatch')
    destination = destination.resolve()
    files = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for info in zipped.infolist():
            relative = Path(*Path(info.filename).parts[1:])
            if info.is_dir() or not relative.parts:
                continue
            target = (destination/relative).resolve()
            if not target.is_relative_to(destination) or info.file_size > 30_000_000:
                raise ValueError('unsafe archive member')
            content = zipped.read(info)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError('refusing to replace different local source')
            else:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(content)
                try:
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        if target.read_bytes() != content:
                            raise ValueError('refusing to replace different local source')
                finally:
                    temporary.unlink()
            files[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    if files['weights/DynaIP.pth'] != CHECKPOINT_SHA256:
        raise ValueError('checkpoint digest mismatch')
    provenance = {'repository': 'https://github.com/dx118/dynaip', 'revision': REVISION,
                  'archive_sha256': ARCHIVE_SHA256, 'files': files}
    provenance_path = destination/'provenance.json'
    try:
        with provenance_path.open('x', encoding='utf-8') as handle:
            json.dump(provenance, handle, indent=2)
    except FileExistsError:
        if json.loads(provenance_path.read_text(encoding='utf-8')) != provenance:
            raise ValueError('refusing to replace different provenance')
    return provenance

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'.tmp/dynaip_official')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        result = fetch(args.output)
        print(json.dumps({'revision': result['revision'], 'files': len(result['files'])}))
    else:
        print(f'Offline preview: DynaIP {REVISION}; use --download to retrieve ~46 MB.')
