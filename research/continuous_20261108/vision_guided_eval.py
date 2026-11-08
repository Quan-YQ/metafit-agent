"""Guide-lines research experiment for public CMU geometry, no user recordings."""
import argparse,hashlib,io,json,os,sys
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'examples')]
from check_vision import build_payload,call_once
from check_vision_observation import compare
from render_motion_sample import load_bvh,world_positions,render_frames

DEV=PROJECT/'.tmp'/'cmu_extra'
HOLD=ROOT/'vision_holdout'
DEV_IDS=('07_01','08_01','13_01','16_01','35_01')
HOLD_IDS=('09_01','10_01','12_01','14_01','17_01')
PROMPT=('这是骨架变化示意图，同一人物的六个时刻按从左到右排列，上下两排是两个视角。'
        '紫色大点标识髋部，绿色两条虚线分别位于第一列髋点上下整幅骨架高度的5%处。'
        '只看第六列紫色点的位置：在上方绿线之外为up，在下方绿线之外为down，'
        '在两线之间为no_clear_change，看不清或无法确定为uncertain。'
        '只输出JSON对象，字段只有hip_change和explanation，解释一句话，不给训练或安全建议。')

def make_case(path,reverse):
    joints,all_frames=load_bvh(path)
    if len(all_frames)<6:raise ValueError('too few frames')
    indices=[round(i*(len(all_frames)-1)/5) for i in range(6)]
    selected=[all_frames[i] for i in indices]
    if reverse:selected=list(reversed(selected))
    poses=[world_positions(joints,f) for f in selected]
    root=next(i for i,j in enumerate(joints) if j.name=='Hips')
    low=min(p[1] for pose in poses for p in pose);high=max(p[1] for pose in poses for p in pose)
    span=high-low
    if span<=1e-12:raise ValueError('zero y-span')
    extent=max(max(abs(p[0]-pose[0][0]),abs(p[2]-pose[0][2])) for pose in poses for p in pose)
    scale=min(155/max(span,1),78/max(extent,1))
    ratio=(poses[-1][root][1]-poses[0][root][1])/span
    direction='up' if ratio>=.05 else 'down' if ratio<=-.05 else 'no_clear_change'
    img=Image.open(io.BytesIO(render_frames(joints,selected))).convert('RGB');d=ImageDraw.Draw(img)
    for row in range(2):
        bottom=row*230+205
        base=bottom-(poses[0][root][1]-low)*scale
        for y in (round(base-span*scale*.05),round(base+span*scale*.05)):
            for x in range(12,1192,12):d.line((x,y,min(x+6,1198),y),fill=(0,145,70),width=1)
        for i,pose in enumerate(poses):
            px=100+i*200;py=round(bottom-(pose[root][1]-low)*scale)
            d.ellipse((px-6,py-6,px+6,py+6),fill=(170,0,190),outline=(255,255,255),width=1)
    b=io.BytesIO();img.save(b,format='PNG')
    return {'direction':direction,'ratio':round(ratio,5),'threshold':.05},b.getvalue()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('dev','holdout'),required=True)
    p.add_argument('--data-dir',type=Path,help='directory containing the selected BVH files')
    p.add_argument('--live',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder,names=(DEV,DEV_IDS) if a.dataset=='dev' else (HOLD,HOLD_IDS)
    if a.data_dir is not None:folder=a.data_dir
    key=os.environ.get('DEEPSEEK_API_KEY','').strip() if a.live else None
    if a.live and not key:raise ValueError('missing key')
    if a.output.exists():raise ValueError('output exists')
    rows=[];usage_in=usage_out=0;failure=None
    with a.output.open('x',encoding='utf-8') as f:
        for name in names:
            for reverse in (False,True):
                reference,image=make_case(folder/(name+'.bvh'),reverse)
                row={'sample':name,'reversed':reverse,'reference':reference}
                if a.live:
                    try:
                        body=json.loads(build_payload(image,PROMPT));body['response_format']={'type':'json_object'};body['max_tokens']=160
                        response,usage,elapsed=call_once(json.dumps(body,ensure_ascii=False).encode('utf-8'),key)
                        if not {'prompt_tokens','completion_tokens'}.issubset(usage):raise ValueError('usage missing')
                        usage_in+=usage['prompt_tokens'];usage_out+=usage['completion_tokens']
                        observation=compare(response,reference)
                        row.update(observation=observation['model_observation']['hip_change'],comparison=observation['comparison'],usage=usage)
                    except ValueError as e:
                        failure=str(e);row['error']='call_or_format_error'
                    if (usage_in*.3+usage_out*1.2)/1_000_000*10>1:failure='budget_guard'
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                if failure:break
            if failure:break
    counts=Counter(r.get('comparison') for r in rows)
    print(json.dumps({'dataset':a.dataset,'live':a.live,'requests':len(rows) if a.live else 0,
        'matches':counts['matches_reference'],'differs':counts['differs_from_reference'],'uncertain':counts['inconclusive'],
        'usage':{'input':usage_in,'output':usage_out},'failure':failure,
        'notice':'Only aided geometric comparison, not general action quality. Forward/reverse share a clip.'},ensure_ascii=False))
    return 1 if failure else 0
if __name__=='__main__':raise SystemExit(main())
