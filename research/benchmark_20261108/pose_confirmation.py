"""Evaluate the already frozen pose models on fresh CIP subject08, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from cip_shared import official_modules, assert_untrained, select_recordings, prepare
from fetch_dynaip import PROJECT, CHECKPOINT_SHA256
from pose_benchmark import baseline_predict, measure, sha256, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-run',type=Path,required=True)
    parser.add_argument('--archive',type=Path,default=PROJECT/'data/cip/CIP_MTwAwinda_subject08.zip')
    parser.add_argument('--official-dir',type=Path,default=PROJECT/'.tmp/dynaip_official')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    frozen=json.loads((args.frozen_run/'selection_frozen.json').read_text(encoding='utf-8'))
    if '08' in frozen['train_subjects'] or '08' in frozen['validation_subjects']:
        raise ValueError('confirmation subject used during baseline selection')
    if sha256(args.frozen_run/'aligned_baseline.pth')!=frozen['baseline_sha256']:
        raise ValueError('baseline changed after selection')
    if args.output.exists():raise ValueError('new output directory required')
    torch.set_num_threads(4)
    modules=official_modules(args.official_dir)
    net=modules[1]().cpu().eval()
    net.load_state_dict(torch.load(args.official_dir/'weights/DynaIP.pth',map_location='cpu',weights_only=True),strict=True)
    saved=torch.load(args.frozen_run/'aligned_baseline.pth',map_location='cpu',weights_only=True)
    if saved['kind']=='mlp':
        network=torch.nn.Sequential(torch.nn.Linear(156,128),torch.nn.ReLU(),torch.nn.Linear(128,128),
                                     torch.nn.ReLU(),torch.nn.Linear(128,66))
        network.load_state_dict(saved.pop('state_dict'),strict=True);network.eval();saved['network']=network
    train=(args.official_dir/'datasets/split_info/xsens/cip_train.txt').read_text()
    test=(args.official_dir/'datasets/split_info/xsens/cip_test.txt').read_text()
    assert_untrained(('08',),train)
    if args.archive.stat().st_size!=1393381371:raise ValueError('archive size differs')
    md5=hashlib.md5()
    with args.archive.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):md5.update(block)
    if md5.hexdigest()!='4e5b9caf2d926006f08f0330024130d3':raise ValueError('official archive MD5 differs')
    per_recording={};metadata={}
    with zipfile.ZipFile(args.archive) as archive,torch.inference_mode():
        for key,member in select_recordings(archive,'08',set(test.splitlines())):
            content=archive.read(member)
            record=prepare(content,modules)
            per_recording[key]={
                'DynaIP':measure(net.predict(record['imu'].clone(),record['v_init'],record['p_init'])[0],record,modules),
                'aligned_small':measure(baseline_predict(saved,record,net),record,modules)}
            metadata[key]=record['metadata']|{'workbook_sha256':hashlib.sha256(content).hexdigest()}
            print(json.dumps({'recording':key,'results':per_recording[key]}),flush=True)
    macro={model:{metric:float(np.mean([row[model][metric] for row in per_recording.values()]))
                  for metric in ('canonical22_mpjpe_cm','learned11_geodesic_deg')}
           for model in ('DynaIP','aligned_small')}
    result={'schema':'dynaip-fresh-subject-confirmation-v1','subject':'08',
            'frozen_selection_sha256':sha256(args.frozen_run/'selection_frozen.json'),
            'baseline_sha256':frozen['baseline_sha256'],'checkpoint_sha256':CHECKPOINT_SHA256,
            'data':{'source':'https://zenodo.org/records/6480543','file':'MTwAwinda_trials_subject08.zip',
                    'md5':md5.hexdigest(),'sha256':sha256(args.archive),'license':'CC-BY-4.0'},
            'per_recording':per_recording,'recordings':metadata,'macro':macro,
            'notice':'Fresh confirmation after subject04 results; no retraining, tuning or protocol changes; same reference-assisted calibration and initialization as selection study.'}
    args.output.mkdir(parents=True)
    write_json(args.output/'results.json',result)
    print(json.dumps({'macro':macro},indent=2))

if __name__=='__main__':main()
