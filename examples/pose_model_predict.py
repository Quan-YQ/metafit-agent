"""Inference for a verified research artifact; do not use for training guidance."""
import json
from pathlib import Path
import numpy as np

def load_model(path: Path):
    with np.load(path,allow_pickle=False) as f:
        data={k:np.array(f[k]) for k in f.files}
    meta=json.loads(str(data.pop('metadata_json').item()))
    if meta.get('schema')!='pose-optimization-fixed-split-1': raise ValueError('unsupported research model')
    name=meta.get('model','')
    if name=='mlp_window':
        shapes={'mean':(430,),'scale':(430,),'y_mean':(60,),'y_scale':(60,),
                'layer_0.weight':(256,430),'layer_0.bias':(256,),
                'layer_2.weight':(128,256),'layer_2.bias':(128,),
                'layer_4.weight':(60,128),'layer_4.bias':(60,)}
    elif name.startswith('ridge_'):
        width=43 if name.startswith('ridge_single_') else 430
        shapes={'mean':(width,),'scale':(width,),'weights':(width+1,60)}
    else: raise ValueError('unsupported model type')
    if set(data)!=set(shapes) or any(data[k].shape!=shape or not np.isfinite(data[k]).all() for k,shape in shapes.items()):
        raise ValueError('invalid model weights')
    if np.any(data['scale']<=0):raise ValueError('invalid scale')
    return meta,data

def predict(model,windows):
    meta,data=model
    x=np.asarray(windows)
    if x.ndim!=2 or x.shape[1]!=430 or not np.isfinite(x).all():raise ValueError('expected finite N x 430 input')
    name=meta['model']
    if name.startswith('ridge_'):
        if name.startswith('ridge_single_'):x=x[:,258:301]
        x=(x.astype(np.float64)-data['mean'])/data['scale']
        return np.column_stack((np.ones(len(x)),x))@data['weights']
    x=((x-data['mean'])/data['scale']).astype(np.float32)
    for layer in (0,2,4):
        x=x@data[f'layer_{layer}.weight'].T+data[f'layer_{layer}.bias']
        if layer!=4:x=np.maximum(x,0)
    return x*data['y_scale']+data['y_mean']
