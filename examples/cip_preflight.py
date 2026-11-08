"""Inspect one fixed public CIP subject: no model training or extraction."""
import argparse
import csv
import hashlib
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import Path
import sys
import zipfile
from xml.etree import ElementTree as ET

SIZE = 1135241724
MD5 = '6cb72f43775ff5ffdc8ec29af9f2ba9c'  # Zenodo 6480543
SENSORS = {'head': '00b45c29', 'pelvis': '00b45c2d', 'left_hand': '00b45c28',
           'right_hand': '00b45c23', 'left_foot': '00b45c31', 'right_foot': '00b45c24'}
TRIALS = {'calibration_npose': 'subject00_calibration/subject00_calibration_npose',
          'sequence_rep2': 'subject00_sequence/subject00_sequence_rep2'}
BASE = 'MTwAwinda/trials/subject00/'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def verify(path):
    if path.stat().st_size != SIZE:
        raise ValueError('archive size differs from official subject00 release')
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != MD5:
        raise ValueError('archive MD5 differs from official release')
    return sha256.hexdigest()

def sensor_data(z, name):
    info = z.getinfo(name)
    if info.file_size > 30_000_000:
        raise ValueError('unexpected sensor file size')
    with z.open(info) as raw:
        with TextIOWrapper(raw, encoding='utf-8-sig', newline='') as text:
            rate = None
            def data_lines():
                nonlocal rate
                for line in text:
                    if line.startswith('//'):
                        if line.startswith('// Update Rate:'):
                            rate = line.strip().split(':',1)[1].strip()
                        continue
                    yield line
            reader = csv.DictReader(data_lines(), delimiter=';')
            needed = ['PacketCounter','Acc_X','Acc_Y','Acc_Z','Quat_q0','Quat_q1','Quat_q2','Quat_q3']
            if not reader.fieldnames or not set(needed).issubset(reader.fieldnames):
                raise ValueError('missing expected sensor columns')
            indices, min_norm, max_norm = [], math.inf, 0.0
            for row in reader:
                if len(indices) >= 100_000: raise ValueError('unexpected number of frames')
                index = int(row['PacketCounter'])
                if not 0 <= index <= 65535: raise ValueError('unexpected packet counter')
                values = [float(row[k]) for k in needed[1:]]
                if not all(math.isfinite(v) for v in values): raise ValueError('nonfinite sensor reading')
                q = math.sqrt(sum(v*v for v in values[3:]))
                min_norm, max_norm = min(min_norm,q), max(max_norm,q)
                indices.append(index)
            if not indices: raise ValueError('empty sensor file')
            gaps = sum((b-a) % 65536 != 1 for a,b in zip(indices,indices[1:]))
            return {'rows':len(indices),'declared_rate':rate,'counter_discontinuities':gaps,
                    'quaternion_norm_range':[round(min_norm,5),round(max_norm,5)]}, indices

def position_sheet(z, name):
    info=z.getinfo(name)
    if info.file_size > 80_000_000: raise ValueError('unexpected workbook size')
    with zipfile.ZipFile(BytesIO(z.read(info))) as book:
        wb=ET.fromstring(book.read('xl/workbook.xml'))
        rels=ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        relmap={e.attrib['Id']:e.attrib['Target'].lstrip('/') for e in rels}
        sheets=wb.findall('.//s:sheet',NS)
        sheet=next(e for e in sheets if e.attrib.get('name')=='Segment Position')
        target=relmap[sheet.attrib['{'+NS['r']+'}id']]
        target=target if target.startswith('xl/') else 'xl/'+target
        if target not in book.namelist() or book.getinfo(target).file_size > 150_000_000:
            raise ValueError('unexpected worksheet size')
        shared=[]
        if 'xl/sharedStrings.xml' in book.namelist():
            if book.getinfo('xl/sharedStrings.xml').file_size > 20_000_000:
                raise ValueError('unexpected shared string size')
            root=ET.fromstring(book.read('xl/sharedStrings.xml'))
            shared=[''.join(e.itertext()) for e in root.findall('s:si',NS)]
        headings, counts, lo, hi = None, 0, math.inf, -math.inf
        previous, nonsequential = None, 0
        with book.open(target) as f:
            for _, e in ET.iterparse(f, events=('end',)):
                if e.tag != '{'+NS['s']+'}row': continue
                vals={}
                for cell in e.findall('s:c',NS):
                    ref=cell.attrib.get('r','')
                    col=''.join(c for c in ref if c.isalpha())
                    v=cell.find('s:v',NS)
                    value=v.text if v is not None else ''
                    if cell.attrib.get('t')=='s' and value: value=shared[int(value)]
                    if cell.attrib.get('t')=='inlineStr':
                        node=cell.find('s:is',NS)
                        value=''.join(node.itertext()) if node is not None else ''
                    vals[col]=value
                if headings is None:
                    headings={v:k for k,v in vals.items()}
                    if not {'Frame','Head z','Pelvis z'}.issubset(headings):
                        raise ValueError('missing segment headings')
                else:
                    frame=int(vals[headings['Frame']])
                    head_z=float(vals[headings['Head z']])
                    if not math.isfinite(head_z): raise ValueError('nonfinite head position')
                    if previous is not None and frame != previous+1: nonsequential+=1
                    previous=frame
                    counts+=1
                    if counts>100_000: raise ValueError('unexpected workbook row count')
                    lo,hi=min(lo,head_z),max(hi,head_z)
                e.clear()
        if not counts: raise ValueError('empty position worksheet')
        return {'rows':counts, 'nonsequential_frame_steps':nonsequential,
                'head_z_min_max_raw_units':[round(lo,4),round(hi,4)],
                'head_position_available': True}

def run(path):
    sha=verify(path)
    result={}
    with zipfile.ZipFile(path) as z:
        for trial, subdir in TRIALS.items():
            prefix=BASE+subdir+'/'
            basename=subdir.split('/')[-1]
            readings, indices={},{}
            for part, sensor in SENSORS.items():
                readings[part],indices[part]=sensor_data(z,prefix+basename+'_'+sensor+'.csv')
            reference=position_sheet(z,prefix+basename+'.xlsx')
            result[trial]={'sensors':readings,'all_six_packet_counters_identical':
                           all(indices[name]==indices['head'] for name in SENSORS),
                           'position_reference':reference,
                           'sensor_reference_row_counts_equal':
                           all(v['rows']==reference['rows'] for v in readings.values())}
    return {'schema':'cip-subject00-preflight-1','dataset':'CIP MTwAwinda subject00',
            'source':'https://zenodo.org/records/6480543', 'archive_sha256':sha,
            'trials':result,'notice':'Rows and counters do not establish time synchronisation; Xsens positions are inertial reference, not independent optical ground truth. Head position is privileged reference, not IMU-only measurement. No training or model evaluation.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path,help='Path to the official subject00 ZIP, without extracting')
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    try:
        print(json.dumps(run(args.archive),ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,KeyError,StopIteration,zipfile.BadZipFile,ET.ParseError) as e:
        print('Preflight failed: '+str(e),file=sys.stderr)
        return 1
if __name__=='__main__': raise SystemExit(main())
