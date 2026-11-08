"""Fixed-split offline 6-IMU+head research experiment. Not a device model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fusionposer_expanded import OLD_SHA256, NEW_GIT
from fusionposer_extra_fetch import FILES

TRAIN=('0625180826','0625181142','0625181240','0625175446','0625175813','0625173852','0625174451')
DEV=('0625174933','0625175157')
HOLD=('0625181450','0625181811','0625182119')
OLD_TRAIN=TRAIN[:5]
ALPHAS=(20.,200.,2000.,20000.)

def verified_matrix(path, digest, algorithm, columns):
    if not path.is_file(): raise ValueError('missing '+path.name)
    size=path.stat().st_size
    h=hashlib.sha1(f'blob {size}\0'.encode()) if algorithm=='git' else hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''): h.update(c)
    if h.hexdigest()!=digest: raise ValueError('digest mismatch '+path.name)
    arr=np.load(path,mmap_mode='r',allow_pickle=False)
    if not isinstance(arr,np.ndarray) or arr.ndim!=2 or arr.shape[1]!=columns or arr.dtype!=np.dtype('<f8') or not np.isfinite(arr).all():
        raise ValueError('invalid matrix '+path.name)
    return arr

def load_recording(directory,identifier):
    result=[]
    for i,suffix in enumerate(('x','y')):
        name=f'motion_{identifier}_{suffix}.npz'
        if identifier in OLD_SHA256: digest,algorithm=OLD_SHA256[identifier][i],'sha256'
        elif identifier in NEW_GIT: digest,algorithm=NEW_GIT[identifier][i],'git'
        else: digest,algorithm=FILES[name][1],'git'
        arr=verified_matrix(directory/name,digest,algorithm,(430,146)[i]); result.append(arr)
    x,y=result
    if len(x)!=len(y) or len(x)<100: raise ValueError('unpaired recording '+identifier)
    return np.asarray(x,dtype=np.float32).copy(),np.asarray(y[:,:60],dtype=np.float32).copy()

def concat(records,names):
    return tuple(np.concatenate([records[n][j] for n in names]) for j in range(2))

def select_features(x,mode):
    return x[:,258:301] if mode=='single' else x

def metrics(pred,y,baseline):
    d=np.linalg.norm((pred-y).reshape(-1,20,3),axis=2)
    b=np.linalg.norm((baseline-y).reshape(-1,20,3),axis=2)
    return {'frames':len(y),'mean':round(float(d.mean()),3),
            'p95_frame_mean':round(float(np.percentile(d.mean(axis=1),95)),3),
            'constant_baseline':round(float(b.mean()),3),
            'per_point_mean':[round(float(v),3) for v in d.mean(axis=0)]}

def normalize(x):
    mean=x.mean(axis=0,dtype=np.float64); scale=x.std(axis=0,dtype=np.float64)
    scale[scale<1e-10]=1
    return mean,scale

def linear_model(x,y,mode,alpha):
    x=select_features(x,mode).astype(np.float64); y=y.astype(np.float64)
    mu,sd=normalize(x); z=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    g=z.T@z; idx=np.arange(1,len(g));g[idx,idx]+=alpha
    w=np.linalg.solve(g,z.T@y)
    return {'kind':'ridge','mode':mode,'alpha':alpha,'mean':mu,'scale':sd,'weights':w}

def linear_predict(model,x):
    x=select_features(x,model['mode']).astype(np.float64)
    return np.column_stack((np.ones(len(x)),(x-model['mean'])/model['scale']))@model['weights']

def train_mlp(x,y,dev_records):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    mu,sd=normalize(x); ym,ys=normalize(y)
    tx=torch.from_numpy(((x-mu)/sd).astype(np.float32)); ty=torch.from_numpy(((y-ym)/ys).astype(np.float32))
    net=torch.nn.Sequential(torch.nn.Linear(430,256),torch.nn.ReLU(),torch.nn.Linear(256,128),
                            torch.nn.ReLU(),torch.nn.Linear(128,60))
    optimizer=torch.optim.Adam(net.parameters(),lr=0.0005,weight_decay=0.0001)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=512,shuffle=True,
                                   generator=torch.Generator().manual_seed(42),num_workers=0)
    dx=[(torch.from_numpy(((dev_records[n][0]-mu)/sd).astype(np.float32)),dev_records[n][1]) for n in DEV]
    best,best_weights,best_epoch,stale=float('inf'),None,0,0
    history=[]
    for epoch in range(1,21):
        net.train()
        for bx,by in dl:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(net(bx),by)
            if not bool(torch.isfinite(loss)): raise ValueError('nonfinite neural loss')
            loss.backward(); optimizer.step()
        net.eval(); scores=[]
        with torch.no_grad():
            for features,target in dx:
                pred=net(features).numpy()*ys+ym
                scores.append(float(np.linalg.norm((pred-target).reshape(-1,20,3),axis=2).mean()))
        val=sum(scores)/len(scores); history.append(round(val,5))
        if val < best-0.00001:
            best,best_epoch,stale=val,epoch,0
            best_weights={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:
            stale+=1
            if stale>=4: break
    net.load_state_dict(best_weights); net.eval()
    return {'kind':'mlp','mode':'window','mean':mu,'scale':sd,'y_mean':ym,'y_scale':ys,
            'network':net,'best_epoch':best_epoch,'dev_history':history}

def predict(model,x):
    if model['kind']=='ridge': return linear_predict(model,x)
    import torch
    with torch.no_grad():
        z=torch.from_numpy(((x-model['mean'])/model['scale']).astype(np.float32))
        return model['network'](z).numpy()*model['y_scale']+model['y_mean']

def dev_score(model,records):
    return sum(float(np.linalg.norm((predict(model,records[n][0])-records[n][1]).reshape(-1,20,3),axis=2).mean()) for n in DEV)/len(DEV)

def save_model(path,model,metadata):
    arrays={k:np.asarray(model[k]) for k in ('mean','scale')}
    if model['kind']=='ridge': arrays['weights']=model['weights']
    else:
        arrays.update(y_mean=model['y_mean'],y_scale=model['y_scale'])
        arrays.update({f'layer_{k}':v.detach().numpy() for k,v in model['network'].state_dict().items()})
    arrays['metadata_json']=np.asarray(json.dumps(metadata,ensure_ascii=False))
    with path.open('xb') as f: np.savez_compressed(f,**arrays)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data'/'fusionposer')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise ValueError('output directory not empty')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records={n:load_recording(args.data_dir,n) for n in TRAIN+DEV}
    train_x,train_y=concat(records,TRAIN); baseline=train_y.mean(axis=0)
    old_x,old_y=concat(records,OLD_TRAIN)
    old_model=linear_model(old_x,old_y,'single',200.)
    candidates=[]; dev_trials=[]
    for mode in ('single','window'):
        for alpha in ALPHAS:
            model=linear_model(train_x,train_y,mode,alpha)
            score=dev_score(model,records)
            dev_trials.append({'model':'ridge_'+mode,'alpha':alpha,'dev_macro_mean':round(score,4)})
            candidates.append((score,f'ridge_{mode}_{alpha}',model))
    neural=train_mlp(train_x,train_y,records)
    neural_score=dev_score(neural,records)
    candidates.append((neural_score,'mlp_window',neural))
    dev_trials.append({'model':'mlp_window','best_epoch':neural['best_epoch'],'dev_macro_mean':round(neural_score,4),'dev_history':neural['dev_history']})
    score,name,chosen=min(candidates,key=lambda x:(x[0],x[1]))
    # Holdouts are only opened after model selection.
    holdout={n:load_recording(args.data_dir,n) for n in HOLD}
    results={}
    for n,(x,y) in holdout.items():
        results[n]={'five_recording_ridge':metrics(predict(old_model,x),y,old_y.mean(axis=0)),
                    'selected':metrics(predict(chosen,x),y,baseline)}
    out={'schema':'pose-optimization-fixed-split-1','train':list(TRAIN),'development':list(DEV),'holdout':list(HOLD),
         'train_frames':len(train_x),'previous_train_frames':len(old_x),'selection_metric':'unweighted mean of per-recording development errors',
         'development_candidates':dev_trials,'selected_model':name,'selected_dev_macro_mean':round(score,4),
         'holdout_results':results,
         'holdout_macro_previous':round(sum(v['five_recording_ridge']['mean'] for v in results.values())/len(results),3),
         'holdout_macro_selected':round(sum(v['selected']['mean'] for v in results.values())/len(results),3),
         'notice':'Research only. Recording identities unknown; inferred 20-point layout and unit; temporal model may use future frames; heldout recordings now observed. Not actual device or clinical performance.'}
    save_model(args.output_dir/'selected_model.npz',chosen,{'model':name,'schema':out['schema'],'input':'6 IMU plus head height; source window and layout assumed','unit':'unconfirmed'})
    with (args.output_dir/'results.json').open('x',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__': main()
