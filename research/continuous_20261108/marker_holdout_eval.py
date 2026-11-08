"""Evaluate fixed pixel reader on preregistered, previously unused CMU sources."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from marker_geometry import classify
from vision_balanced_eval import make_window

FILES={'18_01':'08a783b18cb63314cefe5ea1d953377d0f02de718de4148067a8febd756bdabf',
       '19_01':'24770906878edfc86d17417360fa9e22edbf2aa1bc926f19dde1848f0c130304',
       '20_01':'bc200ceee9cd238c6b9d507d428dc09e8dce69dda30924e2eb7f9b988032ae70',
       '21_01':'2f3de16fce83cca8b0b95332c899ac0883aba8390029cc8b0ac4e8d28536973c',
       '22_01':'ce19883050475c7760354ba8fe3b09ce72b335e6525abe10eae03acec42939fb',
       '24_01':'d183504be20d87933292a709f506adf1a89fa0a0415a7e53f37d9990dc757be6',
       '25_01':'bc3a2d104500ec5ce699b68708d6cd2158e1784adac04fc735a0ca88976f78b1',
       '26_01':'325a7a82ade7d43765f5d077c8f76a080c9e5e887c2b09e03763196bd2490a97'}
MOVING={'01_06':'8882922f6d51ab25cf7c44ac31e23ad21b5deaebc3332c74b01e2a61cb751b1a',
        '15_10':'03d58e8f3731b4bd1c5b9ac73bc1955dd827f4171a2e16628d0c89f7a935bdde',
        '40_06':'9fb48c767f91189e3c9b293dd2d3a6c1326cc834a357fa27abe197ff0afffd54',
        '88_08':'c09ca52ae671d90f5c3086f7a5841e6a73dc8e07731a98e127171da1fb19c693',
        '114_16':'702952e8521d615e4425e569a95f8913d11d932ec7ea73f809ce96fda747790f',
        '136_09':'f958ad4cfc516b025da94eb1085b7ded2f34e0fad2c0520fb554a29794b28c33',
        '141_17':'50480e797770f6ae881caba3f27746a750bbdae237342bb72b2723df9f0663c7',
        '143_18':'1b55e3b6fe9254c14acc5cc53cd6e8ede16d21682029c4ccf56d374d612f508f'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=('initial','moving'),default='initial')
    parser.add_argument('--output',type=Path,required=True,help='new output file; archived results are never overwritten')
    args=parser.parse_args();files=FILES if args.group=='initial' else MOVING
    root=Path(__file__).resolve().parent
    output=args.output
    if output.exists(): raise ValueError('output exists')
    rows=[]
    for name,digest in files.items():
        path=root/'vision_balanced'/(name+'.bvh')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('file digest differs')
        for window in range(5):
            for reverse in (False,True):
                ref,image,start,end=make_window(name,window,reverse)
                try: result=classify(image)
                except ValueError: result='invalid_image'
                rows.append({'name':name,'window':window,'reverse':reverse,'reference':ref,
                    'result':result,'image_sha256':hashlib.sha256(image).hexdigest()})
    grouped={}
    for category in ('up','down','no_clear_change'):
        group=[r for r in rows if r['reference']['direction']==category]
        grouped[category]={'total':len(group),'matches':sum(r['result']==category for r in group),
            'uncertain':sum(r['result']=='uncertain' for r in group),
            'other':dict(Counter(r['result'] for r in group if r['result'] not in (category,'uncertain')))}
    result={'model_sha256':hashlib.sha256((root/'marker_geometry.py').read_bytes()).hexdigest(),
            'files':files,'rows':rows,'grouped':grouped,
            'notice':'Controlled rendered colors and reference guides only; overlapping windows are not independent.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'cases':len(rows),'grouped':grouped}))

if __name__=='__main__':main()
