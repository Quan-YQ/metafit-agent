"""Fixed participant-split CIP research prototype with reference-assisted alignment."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import cip_prepare as prep

TRAIN=('00','01','02');DEV=('03',);HOLD=('04',)
ALPHAS=(10.,100.,1000.,10000.)

def prepare_group(subjects,dir):
    records={};failures=[];hashes={}
    for subject in subjects:
        hashes[subject]=prep.verify_archive(subject)
        for action in prep.TRIALS:
            try:
                x,y,meta=prep.prepare(subject,action)
                records[f'{subject}_{action}']=(x,y,meta)
            except (OSError,ValueError,KeyError,StopIteration) as e:
                failures.append({'subject':subject,'action':action,'error':str(e)})
    return records,failures,hashes

def sample_training(records):
    xs,ys=[],[]
    for name,(x,y,_) in records.items():
        pick=np.linspace(0,len(x)-1,min(3000,len(x)),dtype=np.int64)
        xs.append(x[pick]);ys.append(y[pick])
    if not xs:raise ValueError('no usable training clips')
    return np.concatenate(xs),np.concatenate(ys)

def normalize(x):
    mu=x.mean(axis=0,dtype=np.float64);sd=x.std(axis=0,dtype=np.float64);sd[sd<1e-10]=1
    return mu,sd

def fit_ridge(x,y,alpha):
    x=x.astype(np.float64);y=y.astype(np.float64)
    mu,sd=normalize(x);z=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    g=z.T@z;idx=np.arange(1,len(g));g[idx,idx]+=alpha
    return {'kind':'ridge','alpha':alpha,'x_mean':mu,'x_scale':sd,'weights':np.linalg.solve(g,z.T@y)}

def predict(model,x):
    if model['kind']=='ridge':
        z=(x.astype(np.float64)-model['x_mean'])/model['x_scale']
        return np.column_stack((np.ones(len(x)),z))@model['weights']
    import torch
    z=torch.from_numpy(((x-model['x_mean'])/model['x_scale']).astype(np.float32))
    with torch.no_grad():return model['net'](z).numpy()*model['y_scale']+model['y_mean']

def score(pred,y):
    return float(np.linalg.norm((pred-y).reshape(-1,22,3),axis=2).mean())

def dev_score(model,records):return sum(score(predict(model,x),y) for x,y,_ in records.values())/len(records)

def fit_mlp(x,y,dev):
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    xm,xs=normalize(x);ym,ys=normalize(y)
    tx=torch.from_numpy(((x-xm)/xs).astype(np.float32));ty=torch.from_numpy(((y-ym)/ys).astype(np.float32))
    model={'kind':'mlp','x_mean':xm,'x_scale':xs,'y_mean':ym,'y_scale':ys,
           'net':torch.nn.Sequential(torch.nn.Linear(43,128),torch.nn.ReLU(),torch.nn.Linear(128,128),
                                     torch.nn.ReLU(),torch.nn.Linear(128,66))}
    optimizer=torch.optim.Adam(model['net'].parameters(),lr=.0005,weight_decay=.0001)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=512,shuffle=True,
                                       generator=torch.Generator().manual_seed(42),num_workers=0)
    best,weights,epoch_best,stale=float('inf'),None,0,0;hist=[]
    for epoch in range(1,31):
        model['net'].train()
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model['net'](bx),by)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite neural training loss')
            loss.backward();optimizer.step()
        model['net'].eval();validation=dev_score(model,dev);hist.append(round(validation,6))
        if validation<best-1e-7:
            best,stale,epoch_best=validation,0,epoch
            weights={k:v.detach().clone() for k,v in model['net'].state_dict().items()}
        else:
            stale+=1
            if stale>=5:break
    model['net'].load_state_dict(weights);model['net'].eval()
    model['epoch']=epoch_best;model['history']=hist
    return model

def measure(model,x,y,base):
    d=np.linalg.norm((predict(model,x)-y).reshape(-1,22,3),axis=2)
    b=np.linalg.norm((base-y).reshape(-1,22,3),axis=2)
    return {'frames':len(y),'mean_source_units':round(float(d.mean()),6),
            'p95_frame_mean_source_units':round(float(np.percentile(d.mean(axis=1),95)),6),
            'constant_baseline_source_units':round(float(b.mean()),6),
            'per_segment_source_units':[round(float(v),6) for v in d.mean(axis=0)]}

def export_model(path,model,extra):
    arrays={'metadata_json':np.asarray(json.dumps(extra,ensure_ascii=False))}
    for k in ('x_mean','x_scale','y_mean','y_scale','weights'):
        if k in model:arrays[k]=np.asarray(model[k])
    if model['kind']=='mlp':arrays.update({f'layer_{k}':v.detach().numpy() for k,v in model['net'].state_dict().items()})
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):raise ValueError('output directory not empty')
    out.mkdir(parents=True,exist_ok=True)
    train,train_errors,train_hash=prepare_group(TRAIN,out)
    dev,dev_errors,dev_hash=prepare_group(DEV,out)
    # Four actions per person must all have independently valid alignment for this first evaluation.
    if train_errors or dev_errors:
        with (out/'preparation_errors.json').open('x',encoding='utf-8') as f:
            json.dump({'train':train_errors,'validation':dev_errors},f,ensure_ascii=False,indent=2)
        raise ValueError('some development recordings failed preparation; examine errors without using test data')
    X,Y=sample_training(train);baseline=Y.mean(axis=0)
    choices=[];trials=[]
    for alpha in ALPHAS:
        model=fit_ridge(X,Y,alpha);s=dev_score(model,dev)
        choices.append((s,f'ridge_{alpha}',model));trials.append({'model':f'ridge_{alpha}','validation_macro':round(s,6)})
    model=fit_mlp(X,Y,dev);s=dev_score(model,dev)
    choices.append((s,'mlp',model));trials.append({'model':'mlp','validation_macro':round(s,6),'best_epoch':model['epoch'],'history':model['history']})
    selected_score,name,selected=min(choices,key=lambda a:(a[0],a[1]))
    # Decision is frozen before holdout archive is verified/opened.
    frozen={'train':list(TRAIN),'validation':list(DEV),'test':list(HOLD),'actions':list(prep.TRIALS),
            'candidates':trials,'selected':name,'archive_sha256':train_hash|dev_hash}
    with (out/'selection_frozen.json').open('x',encoding='utf-8') as f:json.dump(frozen,f,indent=2,allow_nan=False)
    hold,hold_errors,hold_hash=prepare_group(HOLD,out)
    if hold_errors:
        with (out/'holdout_preparation_errors.json').open('x',encoding='utf-8') as f:json.dump(hold_errors,f,indent=2,ensure_ascii=False)
        raise ValueError('holdout has preprocessing failures; no partial score reported')
    results={n:{'metric':measure(selected,x,y,baseline),'preprocessing':m} for n,(x,y,m) in hold.items()}
    macro=sum(v['metric']['mean_source_units'] for v in results.values())/len(results)
    base_macro=sum(v['metric']['constant_baseline_source_units'] for v in results.values())/len(results)
    result={'schema':'cip-cross-subject-research-1','selection':frozen,'holdout_archive_sha256':hold_hash,
            'train_frames_used':len(X),'train_clip_metadata':{n:m for n,(_,_,m) in train.items()},
            'validation_clip_metadata':{n:m for n,(_,_,m) in dev.items()},'test_results':results,
            'test_macro_source_units':round(macro,6),'baseline_macro_source_units':round(base_macro,6),
            'notice':'Single unseen participant; Xsens inertial reference, reference-derived head input and reference-aided alignment. 22 segment origins in source units; no product or clinical claim.'}
    export_model(out/'cip_selected_model.npz',selected,{'schema':result['schema'],'selected':name,
                 'input':'6 IMUs + reference head z','targets':'22 pelvis-relative Xsens segment origins'})
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
