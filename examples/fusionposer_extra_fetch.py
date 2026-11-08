"""One-shot download of fixed holdout recordings; verify Git blob, never overwrite."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE='https://raw.githubusercontent.com/LuzyCat/FusionPoser/e8ba6f89b2f6cc031b37c55dd72fc674192df86f/Dataset/real/'
FILES={
'motion_0625174933_x.npz':(12002240,'7f8c54e58c5db24ff935576e652b636c25ff4a27'),
'motion_0625174933_y.npz':(4075232,'aa74608f55130e6a91a2e4523d4116c738bef8e7'),
'motion_0625175157_x.npz':(10485200,'85f65cc0fa5954ab8abd34b22f1faba0b925703d'),
'motion_0625175157_y.npz':(3560144,'094c997be56125cc31b3b7af5857fda85f69465d'),
'motion_0625175446_x.npz':(14417120,'7d8f89918b4d247e28d3539d41189e8096b20cd5'),
'motion_0625175446_y.npz':(4895168,'086bc2c5bd3bcf8c0a81fdb5e04d949491af55ba'),
'motion_0625175813_x.npz':(57946880,'ef894e79c5a97435499d558ffaeaa93d2a6b52cb'),
'motion_0625175813_y.npz':(19675040,'484d514436f01a118d95b99a8090ce13f830a170'),
'motion_0625173852_x.npz':(17120960,'a28c5b261224982823936b07f2ca7659775f5a6d'),
'motion_0625173852_y.npz':(5813216,'b25e23bc492d77dbfbf6b21d16c27303c0c3b5a1'),
'motion_0625174451_x.npz':(21348720,'7b934dd701a2340b8dc688b58181207144b41fe7'),
'motion_0625174451_y.npz':(7248688,'121af1dd768c7ed79c5b6b2dee175769a4469a2d'),
'motion_0625181450_x.npz':(16859520,'70eb4df393478d207e4f75a2d27c650af35c873c'),
'motion_0625181450_y.npz':(5724448,'09aa9cc8ad849cfdb18b780f050413c07e832622'),
'motion_0625181811_x.npz':(18032560,'a707280c97663f84dd285cae78f8e27ee29249e2'),
'motion_0625181811_y.npz':(6122736,'3a78a339cd9aa435c10e30ed262fad7d4f8f176c'),
'motion_0625182119_x.npz':(11630720,'712ed892240ef5a000bfddd83bd9e6abb01398dc'),
'motion_0625182119_y.npz':(3949088,'7632ba3246170abd0c6863e84d2c85b1767e5713'),
}
ROOT=Path(__file__).resolve().parents[1]/'data'/'fusionposer'

def verify(path,size,digest):
    if path.stat().st_size != size: raise ValueError(f'{path.name}: size mismatch')
    h=hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    if h.hexdigest()!=digest: raise ValueError(f'{path.name}: blob mismatch')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='download the nine additional fixed recordings')
    if not parser.parse_args().download:
        print(f'{len(FILES)} public files planned; offline preview. Use --download to fetch.');return
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,(size,digest) in FILES.items():
        final=ROOT/name
        if final.exists():
            verify(final,size,digest)
            continue
        temp=ROOT/(name+'.partial')
        with urlopen(BASE+name,timeout=60) as resp, temp.open('xb') as out:
            remaining=size
            while remaining:
                chunk=resp.read(min(1024*1024,remaining))
                if not chunk: raise ValueError(f'{name}: truncated download')
                out.write(chunk); remaining-=len(chunk)
            if resp.read(1): raise ValueError(f'{name}: oversized download')
        verify(temp,size,digest)
        temp.rename(final)
        print(name,'verified',flush=True)
if __name__=='__main__': main()
