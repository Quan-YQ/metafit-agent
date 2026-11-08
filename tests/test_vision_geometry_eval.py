from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from evaluate_vision_geometry import cases,summarize

def test_fixed_cases_are_paired_and_marked_images_change():
    items=list(cases())
    assert len(items)==20
    for plain,marked in zip(items[::2],items[1::2]):
        assert plain[:2]==marked[:2] and plain[2] is False and marked[2] is True
        assert plain[3]==marked[3]
        assert plain[4]!=marked[4]
        assert plain[4].startswith(b'\x89PNG') and marked[4].startswith(b'\x89PNG')

def test_summary_counts_uncertainty_separately():
    rows=[{'marked':True,'comparison':'matches_reference'},
          {'marked':True,'comparison':'inconclusive'},
          {'marked':False,'comparison':'differs_from_reference'}]
    out=summarize(rows)
    assert out['marked']=={'cases':2,'correct':1,'incorrect':0,'uncertain':1}
    assert out['original']['incorrect']==1
