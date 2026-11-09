"""Offline evidence audit: recompute vision scores and verify frozen hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from vision_benchmark import summarize, digest
from vision_dataset import get_source, load_bvh, render

def verify_images(manifest_path, cache):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    sources={}
    for name,expected in manifest['source_sha256'].items():
        path=get_source(name,cache)
        if digest(path)!=expected:
            raise ValueError('source changed: '+name)
        sources[name]=load_bvh(path)
    for case in manifest['cases']:
        joints,frames=sources[case['source']]
        clip=frames[case['start']:case['end_exclusive']]
        content,reference=render(joints,list(reversed(clip)) if case['reverse'] else clip)
        if hashlib.sha256(content).hexdigest()!=case['image_sha256']:
            raise ValueError('render differs: '+case['id'])
        if reference!=case['reference']:
            raise ValueError('geometry differs: '+case['id'])
    return len(manifest['cases'])

def replay(manifest_path, responses_path, results_path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in responses_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result=json.loads(results_path.read_text(encoding='utf-8'))
    expected_images={case['id']:case['image_sha256'] for case in manifest['cases']}
    if digest(manifest_path)!=result['manifest_sha256'] or digest(responses_path)!=result['responses_sha256']:
        raise ValueError('recorded evidence hash mismatch')
    if any(expected_images.get(row['case_id'])!=row['image_sha256'] for row in rows):
        raise ValueError('response belongs to different image')
    recomputed=summarize(rows,manifest['cases'],list(result['summary']))
    if recomputed!=result['summary']:
        raise ValueError('summary differs from recorded responses')
    return {'requests':len(rows),'summary_matches':True,'stage':manifest['stage']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--responses',type=Path)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--rerender',action='store_true')
    parser.add_argument('--cache',type=Path,default=Path(__file__).resolve().parents[2]/'.tmp/vision_sources_20261108')
    args=parser.parse_args()
    if args.rerender:
        print(json.dumps({'images_verified':verify_images(args.manifest,args.cache)}))
    if args.responses and args.results:
        print(json.dumps(replay(args.manifest,args.responses,args.results)))
