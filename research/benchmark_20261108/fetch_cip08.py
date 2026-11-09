"""Fetch CC-BY4.0 CIP subject08 with bounded HTTP ranges and official MD5."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import tempfile
from urllib.request import Request, urlopen

PROJECT=Path(__file__).resolve().parents[2]
SIZE=1393381371
MD5='4e5b9caf2d926006f08f0330024130d3'
URL='https://zenodo.org/records/6480543/files/MTwAwinda_trials_subject08.zip?download=1'

def fetch(output, workers=6):
    provenance_path=output.with_suffix('.provenance.json')
    if output.exists() or provenance_path.exists():raise ValueError('refuse to replace downloaded archive or provenance')
    output.parent.mkdir(parents=True,exist_ok=True)
    prefix=output.with_suffix('.partial')
    prefix_size=prefix.stat().st_size if prefix.exists() else 0
    if not 0<=prefix_size<=SIZE:raise ValueError('invalid partial archive size')
    parts=output.parent/'.cip08_ranges'
    parts.mkdir(exist_ok=True)
    blocks=[(start,min(SIZE-1,start+16*1024*1024-1)) for start in range(prefix_size,SIZE,16*1024*1024)]
    def block(item):
        start,end=item;target=parts/f'{start}-{end}.part'
        if target.exists() and target.stat().st_size==end-start+1:return target
        for attempt in range(3):
            try:
                request=Request(URL+'&range_start='+str(start),headers={'Range':f'bytes={start}-{end}',
                                'User-Agent':'MetaFit-public-research/1.0'})
                with urlopen(request,timeout=60) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('range response differs')
                    data=response.read(end-start+2)
                if len(data)!=end-start+1:raise ValueError('incomplete range')
                target.write_bytes(data)
                return target
            except (OSError,ValueError):
                if attempt==2:raise
                time.sleep(2**attempt)
    completed=prefix_size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task in as_completed([pool.submit(block,item) for item in blocks]):
            path=task.result();completed+=path.stat().st_size
            print(json.dumps({'downloaded_mib':round(completed/1024**2),'total_mib':round(SIZE/1024**2)}),flush=True)
    # A prior interrupted assembly may exist; never overwrite it.
    with tempfile.NamedTemporaryFile(prefix=output.name+'.', suffix='.assembling', dir=output.parent, delete=False) as temp:
        assembled=Path(temp.name)
    digest=hashlib.md5();sha=hashlib.sha256();total=0
    with assembled.open('wb') as handle:
        sources=([prefix] if prefix_size else [])+[parts/f'{start}-{end}.part' for start,end in blocks]
        for source in sources:
            with source.open('rb') as item:
                for data in iter(lambda:item.read(4*1024*1024),b''):
                    total+=len(data);handle.write(data);digest.update(data);sha.update(data)
    if total!=SIZE or digest.hexdigest()!=MD5:raise ValueError('official MD5 mismatch')
    # Hard-link publication fails atomically if another process created the destination.
    import os
    os.link(assembled, output)
    assembled.unlink()
    result={'source':'https://zenodo.org/records/6480543','file':'MTwAwinda_trials_subject08.zip',
            'size':SIZE,'md5':MD5,'sha256':sha.hexdigest(),'license':'CC-BY-4.0'}
    with provenance_path.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--output',type=Path,default=PROJECT/'data/cip/CIP_MTwAwinda_subject08.zip')
    args=parser.parse_args()
    if args.download:fetch(args.output)
    else:print('Offline preview: subject08 is ~1.4GB; CC-BY4.0, official MD5 is pinned. Use --download.')
