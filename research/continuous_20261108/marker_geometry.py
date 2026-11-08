"""Read deliberately colored guide marks in a controlled research image."""
import io
from PIL import Image

def classify(png):
    image=Image.open(io.BytesIO(png)).convert('RGB')
    if image.size!=(1200,460): raise ValueError('unsupported image size')
    def purple_center(column):
        cx=100+column*200
        ys=[y for x in range(cx-7,cx+8) for y in range(230) if image.getpixel((x,y))==(170,0,190)]
        if len(ys)<8: raise ValueError('missing purple marker')
        return sum(ys)/len(ys)
    first,last=purple_center(0),purple_center(5)
    guides=[y for y in range(230) if image.getpixel((12,y))==(0,145,70)]
    if len(guides)!=2 or not guides[0]<first<guides[1]: raise ValueError('invalid guide bands')
    if min(abs(last-y) for y in guides)<=2: return 'uncertain'
    return 'up' if last<guides[0] else 'down' if last>guides[1] else 'no_clear_change'
