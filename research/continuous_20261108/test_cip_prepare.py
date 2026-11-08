"""Packet and quaternion alignment tests on synthetic, non-personal data."""
import pytest
np=pytest.importorskip('numpy')
import cip_prepare as p

def test_common_packets_respects_different_sensor_starts_and_internal_drops():
    common=np.arange(100,400)
    streams={name:(common.copy(),None,None) for name in p.PARTS}
    streams['Pelvis']=(common[1:],None,None)
    streams['Right Hand']=(np.delete(common,80),None,None)
    idx=p.common_packets(streams)
    assert idx['Head'][0]==1 and idx['Pelvis'][0]==0
    assert len(idx['Head'])==298
    for name in p.PARTS:
        assert np.array_equal(streams[name][0][idx[name]],streams['Head'][0][idx['Head']])

def test_reference_lag_uses_only_consecutive_packets_and_correct_offset():
    t=np.arange(600);angle=.3*np.sin(t*.017)+.2*np.sin(t*t*.00017)
    q=np.column_stack((np.cos(angle/2),np.zeros(len(t)),np.zeros(len(t)),np.sin(angle/2)))
    indices=np.delete(np.arange(0,500),np.arange(90,100))
    result=p.indexed_lag_correlations(q[indices+2],q,indices,indices)
    best=max((v,k) for k,v in result.items() if v is not None)
    assert best[1]==2 and best[0]>.99999

def test_select_first_unflagged_repetition():
    available={
        'MTwAwinda/trials/subject03/subject03_sequence/__subject03__sequence_rep1_(loose_LShoulder)/subject03_sequence_rep1.xlsx',
        'MTwAwinda/trials/subject03/subject03_sequence/subject03_sequence_rep2/subject03_sequence_rep2.xlsx'}
    assert p.select_trial(available,'subject03','sequence')=='subject03_sequence_rep2'
    with pytest.raises(ValueError):p.select_trial(available,'subject03','task')
