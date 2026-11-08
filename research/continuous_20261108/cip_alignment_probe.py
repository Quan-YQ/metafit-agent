"""Research diagnostic: compare relative quaternion motion across CIP data streams."""
import argparse,csv,io,json,math,os,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

DATA=Path(os.environ.get('CIP_DATA_DIR',Path(__file__).resolve().parents[2]/'data'/'cip'))
SENSORS={'Pelvis':'00b45c2d','Head':'00b45c29','Left Hand':'00b45c28',
         'Right Hand':'00b45c23','Left Foot':'00b45c31','Right Foot':'00b45c24'}
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def read_reference(book):
    wb=ET.fromstring(book.read('xl/workbook.xml'))
    rel=ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
    mapping={e.attrib['Id']:e.attrib['Target'].lstrip('/') for e in rel}
    sheet=next(e for e in wb.findall('.//s:sheet',NS) if e.attrib['name']=='Sensor Orientation - Quat')
    target=mapping[sheet.attrib['{'+NS['r']+'}id']];target=target if target.startswith('xl/') else 'xl/'+target
    rows=[]; positions=None
    with book.open(target) as f:
        for _,e in ET.iterparse(f,events=('end',)):
            if e.tag!='{'+NS['s']+'}row':continue
            vals={}
            for c in e.findall('s:c',NS):
                ref=c.attrib.get('r','');col=''.join(t for t in ref if t.isalpha())
                v=c.find('s:v',NS);txt=v.text if v is not None else ''
                if c.attrib.get('t')=='inlineStr':
                    node=c.find('s:is',NS);txt=''.join(node.itertext()) if node is not None else ''
                vals[col]=txt
            if positions is None:
                inverse={v:k for k,v in vals.items()}
                positions=[[inverse[f'{name} q{i}'] for i in range(4)] for name in SENSORS]
            else:
                rows.append([[float(vals[c]) for c in cols] for cols in positions])
            e.clear()
    return np.asarray(rows,dtype=float)

def read_sensor(z,path):
    with z.open(path) as raw,io.TextIOWrapper(raw,encoding='utf-8-sig',newline='') as text:
        data=csv.DictReader((line for line in text if not line.startswith('//')),delimiter=';')
        rows=list(data)
        return np.asarray([[float(r[f'Quat_q{i}']) for i in range(4)] for r in rows]),[int(r['PacketCounter']) for r in rows]

def motion(q):
    norms=np.linalg.norm(q,axis=1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norms<1e-8):raise ValueError('invalid quaternion')
    q=q/norms
    dots=np.abs(np.sum(q[1:]*q[:-1],axis=1))
    return 2*np.arccos(np.clip(dots,-1,1))

def lag_correlations(a,b):
    result={}
    for lag in range(-8,9):
        aa=a[max(0,-lag):min(len(a),len(b)-lag)]
        bb=b[max(0,lag):min(len(b),len(a)+lag)]
        result[lag]=float(np.corrcoef(aa,bb)[0,1]) if len(aa)>10 and np.std(aa)>1e-8 and np.std(bb)>1e-8 else None
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--subject',choices=('00','01','02','03'),default='00')
    args=parser.parse_args();subject='subject'+args.subject
    archive=DATA/f'CIP_MTwAwinda_{subject}.zip'
    results={}
    with zipfile.ZipFile(archive) as outer:
        for trial in (f'{subject}_sequence_rep1',f'{subject}_sequence_rep2',f'{subject}_sequence_rep3'):
            prefix=f'MTwAwinda/trials/{subject}/{subject}_sequence/{trial}/{trial}'
            with zipfile.ZipFile(io.BytesIO(outer.read(prefix+'.xlsx'))) as book:ref=read_reference(book)
            sensors={}
            for idx,(part,sid) in enumerate(SENSORS.items()):
                q,counter=read_sensor(outer,prefix+'_'+sid+'.csv')
                corr=lag_correlations(motion(q),motion(ref[:,idx,:]))
                valid=[(v,k) for k,v in corr.items() if v is not None and math.isfinite(v)]
                best=max(valid) if valid else (None,None)
                a,b=motion(q),motion(ref[:,idx,:]); blocks=[]
                for start in range(0,min(len(a),len(b))-300,450):
                    piece=lag_correlations(a[start:start+450],b[start:start+450])
                    options=[(v,k) for k,v in piece.items() if v is not None and math.isfinite(v)]
                    if options:
                        value,lag=max(options);blocks.append({'lag':lag,'correlation':round(value,4)})
                sensors[part]={'sensor_frames':len(q),'reference_frames':len(ref),'best_lag_frames':best[1],
                               'first_packet':counter[0],'last_packet':counter[-1],
                               'best_correlation':round(best[0],4) if best[0] is not None else None,
                               'block_lags':blocks,
                               'lag_zero_correlation':round(corr[0],4) if corr[0] is not None else None,
                               'lag_plus_one_correlation':round(corr[1],4) if corr[1] is not None else None}
            results[trial]=sensors
    print(json.dumps({'cases':results,'notice':'Quaternion motion correlation is exploratory and not proof of independent ground truth or synchronization. Lag convention compares sensor t to reference t+lag.'},indent=2))
if __name__=='__main__':main()
