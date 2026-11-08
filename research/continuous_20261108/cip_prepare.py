"""Research-only CIP preprocessing with per-sensor counters and reference-aided alignment."""
import csv,hashlib,io,json,math,os,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import Counter
import numpy as np
from cip_alignment_probe import SENSORS,NS,motion,lag_correlations

DATA=Path(os.environ.get('CIP_DATA_DIR',Path(__file__).resolve().parents[2]/'data'/'cip'))
SUBJECTS=('00','01','02','03','04')
TRIALS=('sequence','circuit','task','random')
PARTS=list(SENSORS)
ARCHIVES={'00':(1135241724,'6cb72f43775ff5ffdc8ec29af9f2ba9c'),
          '01':(1102763782,'4a787920bc7a8b121ed26f72a4775634'),
          '02':(1074322866,'766d5b1ff8c2a013c379331e335c8ee6'),
          '03':(1283483265,'4e48890a271133fe223b32895a6a57bf'),
          '04':(1093789041,'a9844fea91b0c5ab80aca29734eacf5d')}

def verify_archive(subject):
    size,expected=ARCHIVES[subject]; path=DATA/f'CIP_MTwAwinda_subject{subject}.zip'
    if path.stat().st_size!=size:raise ValueError('archive size mismatch')
    md5,sha=hashlib.md5(),hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            md5.update(block);sha.update(block)
    if md5.hexdigest()!=expected:raise ValueError('archive hash mismatch')
    return sha.hexdigest()

def csv_stream(z,name):
    with z.open(name) as raw,io.TextIOWrapper(raw,encoding='utf-8-sig',newline='') as text:
        reader=csv.DictReader((line for line in text if not line.startswith('//')),delimiter=';')
        rows=list(reader)
    packet=np.array([int(r['PacketCounter']) for r in rows],dtype=np.int64)
    q=np.array([[float(r[f'Quat_q{i}']) for i in range(4)] for r in rows],dtype=np.float32)
    acc=np.array([[float(r['Acc_'+axis]) for axis in 'XYZ'] for r in rows],dtype=np.float32)
    if len(packet)<120 or len(packet)>=65536 or len(np.unique(packet))!=len(packet):raise ValueError('invalid packet counts')
    if not np.isfinite(q).all() or not np.isfinite(acc).all():raise ValueError('nonfinite sensor values')
    return packet,q,acc

def read_sheets(book):
    wb=ET.fromstring(book.read('xl/workbook.xml'));rels=ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
    targets={e.attrib['Id']:e.attrib['Target'].lstrip('/') for e in rels}
    sheets={e.attrib['name']:e.attrib['{'+NS['r']+'}id'] for e in wb.findall('.//s:sheet',NS)}
    result={}
    for name in ('Sensor Orientation - Quat','Segment Position'):
        target=targets[sheets[name]];target=target if target.startswith('xl/') else 'xl/'+target
        if book.getinfo(target).file_size>200_000_000:raise ValueError('oversized worksheet')
        rows=[];frame_numbers=[];columns=None;labels=None;frame_column=None
        with book.open(target) as f:
            for _,e in ET.iterparse(f,events=('end',)):
                if e.tag!='{'+NS['s']+'}row':continue
                cells={}
                for c in e.findall('s:c',NS):
                    col=''.join(ch for ch in c.attrib.get('r','') if ch.isalpha())
                    v=c.find('s:v',NS);value=v.text if v is not None else ''
                    if c.attrib.get('t')=='inlineStr':
                        node=c.find('s:is',NS);value=''.join(node.itertext()) if node is not None else ''
                    cells[col]=value
                if columns is None:
                    inverse={v:k for k,v in cells.items()}
                    frame_column=inverse['Frame']
                    if name=='Sensor Orientation - Quat':
                        labels=PARTS; columns=[[inverse[f'{part} q{i}'] for i in range(4)] for part in labels]
                    else:
                        labels=[v[:-2] for v in cells.values() if v.endswith(' x')]
                        if len(labels)!=23 or labels[0]!='Pelvis' or 'Head' not in labels:raise ValueError('unexpected segment headers')
                        columns=[[inverse[f'{part} {axis}'] for axis in 'xyz'] for part in labels]
                else:
                    frame_numbers.append(int(cells[frame_column]))
                    rows.append([[float(cells[c]) for c in cols] for cols in columns])
                    if len(rows)>100000:raise ValueError('too many rows')
                e.clear()
        array=np.asarray(rows,dtype=np.float32)
        if not len(array) or not np.isfinite(array).all():raise ValueError('bad reference values')
        if any(b!=a+1 for a,b in zip(frame_numbers,frame_numbers[1:])):raise ValueError('reference frame gaps')
        if name=='Sensor Orientation - Quat': prior_frames=frame_numbers
        elif frame_numbers!=prior_frames:raise ValueError('reference sheet frame indices differ')
        result[name]=(labels,array)
    return result

def common_packets(streams,anchor='Head'):
    reference=streams[anchor][0]
    sets=[set(item[0].tolist()) for item in streams.values()]
    packets=[int(p) for p in reference if all(int(p) in s for s in sets)]
    if len(packets)<120:raise ValueError('too little shared data')
    indices={}
    for part,item in streams.items():
        lookup={int(counter):i for i,counter in enumerate(item[0])}
        indices[part]=np.array([lookup[p] for p in packets])
    if any(np.any(np.diff(idx)<=0) for idx in indices.values()):raise ValueError('nonmonotonic sensor index')
    return indices

def indexed_lag_correlations(data_q,ref_q,head_indices,packets):
    """Compare angular increments only at adjacent common sensor packets."""
    a,b=motion(data_q),motion(ref_q)
    adjacent=(np.diff(head_indices)==1) & (np.diff(packets)%65536==1)
    out={}
    for lag in range(-4,5):
        dest=head_indices[:-1]+lag
        valid=adjacent & (dest>=0) & (dest<len(b))
        aa,bb=a[valid],b[dest[valid]]
        out[lag]=float(np.corrcoef(aa,bb)[0,1]) if len(aa)>=100 and np.std(aa)>1e-8 and np.std(bb)>1e-8 else None
    return out

def select_trial(available,sid,action):
    trial=next((f'{sid}_{action}_rep{i}' for i in range(1,6)
                if f'MTwAwinda/trials/{sid}/{sid}_{action}/{sid}_{action}_rep{i}/{sid}_{action}_rep{i}.xlsx' in available),None)
    if trial is None:raise ValueError('no unflagged repetition')
    return trial

def prepare(subject,action):
    if subject not in SUBJECTS or action not in TRIALS:raise ValueError('invalid fixed sample')
    sid='subject'+subject
    archive=DATA/f'CIP_MTwAwinda_{sid}.zip'
    with zipfile.ZipFile(archive) as z:
        available=set(z.namelist())
        trial=select_trial(available,sid,action)
        prefix=f'MTwAwinda/trials/{sid}/{sid}_{action}/{trial}/{trial}'
        streams={part:csv_stream(z,prefix+'_'+identifier+'.csv') for part,identifier in SENSORS.items()}
        info=z.getinfo(prefix+'.xlsx')
        if info.file_size>100_000_000:raise ValueError('oversized workbook')
        with zipfile.ZipFile(io.BytesIO(z.read(info))) as workbook: sheets=read_sheets(workbook)
    names,pos=sheets['Segment Position']; _,ref_q=sheets['Sensor Orientation - Quat']
    if len(pos)!=len(ref_q):raise ValueError('reference sheets have different lengths')
    eligible=[part for part in PARTS if np.all(np.diff(streams[part][0])%65536==1)]
    if not eligible:raise ValueError('no sensor has a continuous packet timeline')
    anchor=max(eligible,key=lambda name:len(streams[name][0]))
    index=common_packets(streams,anchor=anchor); n=len(index[anchor]);anchor_indices=index[anchor]
    common_counter=streams[anchor][0][anchor_indices]
    lag_votes=[]; correlations=[]
    for i,part in enumerate(PARTS):
        data_q=streams[part][1][index[part]]
        ref=ref_q[:,i,:]
        if len(ref)<120:raise ValueError('too few reference orientations')
        corr=indexed_lag_correlations(data_q,ref,anchor_indices,common_counter)
        choices=[(v,k) for k,v in corr.items() if k in range(-4,5) and v is not None and math.isfinite(v)]
        if choices:
            score,lag=max(choices)
            if score>=.97:lag_votes.append(lag);correlations.append(score)
    if len(lag_votes)<4:raise ValueError('alignment lacks four reliable sensors')
    counts=Counter(lag_votes);lag,votes=counts.most_common(1)[0]
    if votes<4:raise ValueError('sensor alignment has no consensus')
    raw_ref=anchor_indices+lag
    # All clips: reserve the first 300 anchor packets (~5 seconds at 60 Hz)
    # for initial calibration instead of scoring the static start.
    keep=(raw_ref>=0)&(raw_ref<len(pos))&(anchor_indices>=300)
    if np.count_nonzero(keep)<120:raise ValueError('insufficient aligned frames')
    ref_index=raw_ref[keep]
    features=[pos[ref_index,names.index('Head'),2:3]]
    for part in PARTS:
        packet,q,acc=streams[part];seq=index[part][keep]
        features.extend([q[seq],acc[seq]])
    X=np.concatenate(features,axis=1).astype(np.float32)
    local=pos[ref_index,1:,:]-pos[ref_index,0:1,:]
    Y=local.reshape(len(local),-1).astype(np.float32)
    if X.shape[1]!=43 or Y.shape[1]!=66:raise ValueError('unexpected output dims')
    return X,Y,{'subject':subject,'action':action,'trial':trial,'rows':len(X),'reference_lag':lag,
               'timing_anchor':anchor,'omitted_anchor_packets':int(len(streams[anchor][0])-len(anchor_indices)),
               'initial_anchor_rows_excluded':300,
               'nonconsecutive_steps':int(np.count_nonzero((np.diff(anchor_indices)!=1)|(np.diff(common_counter)%65536!=1))),
               'votes':dict(counts),'minimum_corr':round(min(correlations),4),
               'head_reference_proxy':True,'alignment_uses_reference_sensor_orientations':True,
               'target_segments':names[1:],'position_units':'source units; independently confirm',
               'notice':'Offline oracle alignment and reference-derived head height. Not a direct deployment model.'}
