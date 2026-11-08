"""Fixed 20-case public CMU hip-direction evaluation; defaults offline."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_motion_geometry import SAMPLES, analyze
from check_vision import build_payload, call_once
from check_vision_observation import PROMPT,compare
from render_motion_sample import load_bvh,render_frames,world_positions

NAMES=('round_a','round_b','round_c','squat_1','walk_1')
MARKED_PROMPT=PROMPT.replace('髋部圆点','髋部根关节点（橙色实心点）')
NOTICE=('仅衡量公开骨架图的髋部首尾方向，非专业动作标准。正逆序和标记前后共用源片段；'
        '对照同时更改标记与提示，不能单独归因；不代表动作安全或跨人准确率。')

def cases(extra=False):
    if extra:
        folder=Path(__file__).resolve().parents[1]/'.tmp'/'cmu_extra'
        for name in ('07_01','08_01','13_01','16_01','35_01'):
            path=folder/(name+'.bvh')
            joints,all_frames=load_bvh(path)
            if len(all_frames)<6:raise ValueError('too few motion frames')
            indices=[round(i*(len(all_frames)-1)/5) for i in range(6)]
            frames=[all_frames[i] for i in indices]
            for reverse in (False,True):
                ordered=list(reversed(frames)) if reverse else frames
                poses=[world_positions(joints,frame) for frame in ordered]
                root=next(i for i,j in enumerate(joints) if j.name=='Hips')
                span=max(p[1] for pose in poses for p in pose)-min(p[1] for pose in poses for p in pose)
                if span<=1e-12:raise ValueError('zero span')
                ratio=(poses[-1][root][1]-poses[0][root][1])/span
                direction='up' if ratio>=.05 else 'down' if ratio<=-.05 else 'no_clear_change'
                ref={'direction':direction,'change_over_skeleton_y_span':round(ratio,4),'threshold':.05}
                yield name,reverse,True,ref,render_frames(joints,frames,reverse=reverse,draw_root=True)
        return
    for name in NAMES:
        path=SAMPLES[name]
        joints,frames=load_bvh(path)
        for reverse in (False,True):
            reference=analyze(path,reverse=reverse)['hip_endpoint_reference']
            for marked in (False,True):
                yield name,reverse,marked,reference,render_frames(joints,frames,reverse=reverse,draw_root=marked)

def summarize(records):
    output={}
    for marked in (False,True):
        subset=[r for r in records if r['marked']==marked and r.get('comparison')]
        c=Counter(r['comparison'] for r in subset)
        output['marked' if marked else 'original']={'cases':len(subset),'correct':c['matches_reference'],
                    'incorrect':c['differs_from_reference'],'uncertain':c['inconclusive']}
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',action='store_true')
    p.add_argument('--extra',action='store_true',help='five newly fixed public CMU sources, marked only')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    key=os.environ.get('DEEPSEEK_API_KEY','').strip() if args.live else None
    if args.live and not key: raise ValueError('missing API key')
    records=[]; total_input=total_output=0; failure=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        for name,reverse,marked,reference,image in cases(extra=args.extra):
            row={'sample':name,'reversed':reverse,'marked':marked,'reference':reference['direction']}
            if args.live:
                payload=json.loads(build_payload(image,MARKED_PROMPT if marked else PROMPT))
                payload['response_format']={'type':'json_object'};payload['max_tokens']=160
                try:
                    answer,usage,elapsed=call_once(json.dumps(payload,ensure_ascii=False).encode('utf-8'),key)
                    total_input+=usage.get('prompt_tokens',0);total_output+=usage.get('completion_tokens',0)
                    if not {'prompt_tokens','completion_tokens'}.issubset(usage):
                        raise ValueError('usage_missing')
                    compared=compare(answer,reference)
                    row.update(observation=compared['model_observation']['hip_change'],
                               comparison=compared['comparison'],usage=usage,elapsed_ms=round(elapsed,1))
                except ValueError as e:
                    failure=str(e);row['error']='call_or_output_failed'
                # Conservative peak USD estimate, convert at 10 CNY/USD; stop well before 10 CNY.
                estimate=(total_input*0.3+total_output*1.2)/1_000_000*10
                if estimate>2: failure='estimated_budget_guard'
            records.append(row)
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            if failure: break
    output={'mode':'live' if args.live else 'offline_preview','dataset':'extra' if args.extra else 'original',
            'requests_started':len(records) if args.live else 0,
            'summary':summarize(records),'input_tokens':total_input if args.live else None,
            'output_tokens':total_output if args.live else None,'stopped':failure is not None,
            'stop_reason':failure,'peak_estimated_yuan_at_10_per_usd':round(estimate,6) if args.live else None,
            'pricing':'https://api-docs.deepseek.com/quick_start/pricing/',
            'notice':NOTICE+' 费用仅为估算，以服务商账单为准。'}
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    return 1 if failure else 0
if __name__=='__main__':raise SystemExit(main())
