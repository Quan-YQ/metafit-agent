"""Fetch preregistered public CMU BVH files at a pinned repository revision."""
from pathlib import Path
from urllib.request import urlopen
import argparse, hashlib, json

ROOT = Path(__file__).resolve().parent / 'vision_balanced'
BASE = 'https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/09a07f54f3bbb58797325f009282d0b2048a2871/data/'
FILES = {'049/49_02.bvh':'cf201c93b7cf1337c8a41d74c96d2062a51b621a9b1e272961e1fe4dd6590924',
         '069/69_70.bvh':'fac827b41e02faeaa68ed6b24b6470e00db22be252805d9b3724c0cc97cfffb5',
         '075/75_01.bvh':'944504bc271fcfa57c918ba755dc98beed21e2a65783f27a72bf5d1bfc4d0f77',
         '086/86_09.bvh':'9c4138c29a481dd3ac87eeaf827c5c8e403351a16a0c2644cca367f8847df4b4'}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='download the four fixed public BVH files')
    if not parser.parse_args().download:
        print('4 public files planned; offline preview. Use --download to fetch.');return
    ROOT.mkdir(exist_ok=True)
    for name in FILES:
        target = ROOT / Path(name).name
        if target.exists():
            raise ValueError(f'target exists: {target.name}')
        with urlopen(BASE + name, timeout=60) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000 or not data.startswith(b'HIERARCHY') or hashlib.sha256(data).hexdigest() != FILES[name]:
            raise ValueError(f'unexpected content: {name}')
        target.write_bytes(data)
        print(json.dumps({'file': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))

if __name__=='__main__':main()
