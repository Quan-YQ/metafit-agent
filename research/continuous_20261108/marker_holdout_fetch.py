"""Fetch fixed holdout source files at the pinned commit and verify hashes."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import argparse
from marker_holdout_eval import FILES, MOVING

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='download the 16 fixed public BVH files')
    if not parser.parse_args().download:
        print('16 public files planned; offline preview. Use --download to fetch.');return
    root=Path(__file__).resolve().parent/'vision_balanced'
    root.mkdir(exist_ok=True)
    base='https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/09a07f54f3bbb58797325f009282d0b2048a2871/data/'
    for name,digest in (FILES|MOVING).items():
        path=root/(name+'.bvh')
        if path.exists(): raise ValueError('refusing to overwrite '+name)
        url=base+f'{int(name.split("_")[0]):03d}/{name}.bvh'
        with urlopen(url,timeout=60) as f: data=f.read(20_000_001)
        if len(data)>20_000_000 or sha256(data).hexdigest()!=digest:raise ValueError('digest mismatch '+name)
        with path.open('xb') as f:f.write(data)
        print(name,'verified')

if __name__=='__main__':main()
