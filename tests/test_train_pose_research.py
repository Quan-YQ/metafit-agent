from pathlib import Path
import sys
import pytest
np=pytest.importorskip('numpy')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
import train_pose_research as pose

def test_single_frame_selection_and_ridge_fit_on_known_linear_problem():
    x=np.zeros((100,430),dtype=np.float32)
    x[:,258]=np.linspace(-2,2,100)
    y=np.tile((3*x[:,258]+1)[:,None],(1,60))
    model=pose.linear_model(x,y,'single',0.001)
    result=pose.linear_predict(model,x)
    assert np.max(np.abs(result-y))<0.001
    assert pose.select_features(x,'single').shape==(100,43)

def test_error_metric_is_mean_euclidean_distance_not_coordinate_average():
    y=np.zeros((2,60)); pred=np.tile(np.array([3.,4.,0.]),(2,20))
    result=pose.metrics(pred,y,np.zeros(60))
    assert result['mean']==5 and result['p95_frame_mean']==5
