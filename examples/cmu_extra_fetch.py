"""Fetch and verify five fixed CMU-converted BVH files; no overwrite or retry."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE='https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/09a07f54f3bbb58797325f009282d0b2048a2871/data/'
FILES={'007/07_01.bvh':(241649,'0e50b7db19c7ea0bf2cd3df942f82642b9b22ebc'),
       '008/08_01.bvh':(211678,'8cbb15e622e5bf0b3d8e969276b971b3224e684c'),
       '013/13_01.bvh':(1747390,'4cbc18dd53f3d744ef68d3a6243df85354f1a121'),
       '016/16_01.bvh':(244676,'d8d1ddee2783b79322349d4b2d617445c78bde54'),
       '035/35_01.bvh':(270282,'69a4da6da3f96a7ebfd348c931d0bf576f7ac427')}
DEST=Path(__file__).resolve().parents[1]/'.tmp'/'cmu_extra'
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for name,(size,digest) in FILES.items():
        path=DEST/Path(name).name
        if path.exists():raise ValueError('target exists')
        with urlopen(BASE+name,timeout=60) as res:raw=res.read(size+1)
        if len(raw)!=size or hashlib.sha1(f'blob {size}\0'.encode()+raw).hexdigest()!=digest:
            raise ValueError('digest mismatch')
        with path.open('xb') as f:f.write(raw)
        print(path.name,'verified')
if __name__=='__main__':main()
