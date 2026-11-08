from pathlib import Path
import sys
import json
import pytest
np=pytest.importorskip('numpy')

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from pose_model_predict import load_model,predict

def test_ridge_artifact_load_and_input_validation(tmp_path):
    p=tmp_path/'model.npz'
    meta={'schema':'pose-optimization-fixed-split-1','model':'ridge_single_200.0'}
    weights=np.zeros((44,60));weights[0,:]=3;weights[1,:]=2
    np.savez(p,metadata_json=json.dumps(meta),mean=np.zeros(43),scale=np.ones(43),weights=weights)
    model=load_model(p);x=np.zeros((2,430));x[:,258]=4
    assert np.allclose(predict(model,x),11)
    with pytest.raises(ValueError):predict(model,np.zeros((1,429)))
    x[0,0]=np.nan
    with pytest.raises(ValueError):predict(model,x)

def test_invalid_model_weights_are_rejected(tmp_path):
    p=tmp_path/'bad.npz';meta={'schema':'pose-optimization-fixed-split-1','model':'ridge_single_200.0'}
    np.savez(p,metadata_json=json.dumps(meta),mean=np.zeros(43),scale=np.zeros(43),weights=np.zeros((44,60)))
    with pytest.raises(ValueError):load_model(p)
