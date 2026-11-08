import io
import pytest
pytest.importorskip('PIL')
from PIL import Image, ImageDraw
from marker_geometry import classify

def sample(last, include_guides=True):
    image=Image.new('RGB',(1200,460),'white')
    draw=ImageDraw.Draw(image)
    if include_guides:
        for y in (90,110): draw.line((12,y,18,y),fill=(0,145,70))
    for x,y in ((100,100),(1100,last)):
        draw.ellipse((x-6,y-6,x+6,y+6),fill=(170,0,190),outline='white',width=1)
    out=io.BytesIO();image.save(out,format='PNG');return out.getvalue()

@pytest.mark.parametrize(('last','result'),[(80,'up'),(120,'down'),(100,'no_clear_change'),(91,'uncertain')])
def test_controlled_markers(last,result):
    assert classify(sample(last))==result

def test_missing_guides_fail():
    with pytest.raises(ValueError):classify(sample(80,False))
