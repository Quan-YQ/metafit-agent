"""Common DynaIP/CIP contract; all reference assistance is explicit.

Matches pinned official read_xlsx/process_xsens/normalize_imu. Uses Xsens
workbook streams together, never target-derived temporal cross-correlation.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

SENSORS = ('Pelvis', 'Left Lower Leg', 'Right Lower Leg', 'Head', 'Left Forearm', 'Right Forearm')
# Workbook sensor columns use spaced names; the official 23->17->6 masks compose to these.
SENSOR_MASK = [0, 20, 16, 6, 13, 9]
JOINT_MASK = [19, 15, 1, 2, 3, 4, 5, 11, 7, 12, 8]
VELOCITY_MASK = [0, 6, 14, 10, 21, 17]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
FPS = 60
CALIBRATION = 150
WARMUP = 300
G = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)

def read_workbook(content):
    wanted = {'Segment Orientation - Quat': 4, 'Segment Position': 3,
              'Sensor Orientation - Quat': 4, 'Sensor Free Acceleration': 3}
    result = {}
    with zipfile.ZipFile(io.BytesIO(content)) as book:
        workbook = ET.fromstring(book.read('xl/workbook.xml'))
        rels = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        paths = {e.attrib['Id']: e.attrib['Target'].lstrip('/') for e in rels}
        frames = None
        for sheet in workbook.findall('.//s:sheet', NS):
            name = sheet.attrib['name']
            if name not in wanted:
                continue
            target = paths[sheet.attrib['{'+NS['r']+'}id']]
            target = target if target.startswith('xl/') else 'xl/'+target
            if book.getinfo(target).file_size > 250_000_000:
                raise ValueError('oversized reference sheet')
            rows, ids, columns, labels = [], [], None, None
            with book.open(target) as handle:
                for _, element in ET.iterparse(handle, events=('end',)):
                    if element.tag != '{'+NS['s']+'}row':
                        continue
                    cells = {}
                    for cell in element.findall('s:c', NS):
                        col = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                        value = cell.find('s:v', NS)
                        if cell.attrib.get('t') == 'inlineStr':
                            value = cell.find('s:is', NS)
                            cells[col] = ''.join(value.itertext())
                        else:
                            cells[col] = value.text if value is not None else ''
                    if columns is None:
                        frame_col = next(k for k, v in cells.items() if v == 'Frame')
                        columns = [k for k in cells if k != frame_col]
                        labels = [cells[k] for k in columns]
                        if len(columns) != 23*wanted[name]:
                            raise ValueError('unexpected workbook columns')
                    else:
                        ids.append(int(cells[frame_col]))
                        rows.append([float(cells[k]) for k in columns])
                        if len(rows) > 100_000:
                            raise ValueError('too many frames')
                    element.clear()
            ids = np.asarray(ids)
            if len(ids) <= WARMUP+120 or not np.all(np.diff(ids) == 1):
                raise ValueError('insufficient or noncontinuous workbook frames')
            if frames is not None and not np.array_equal(ids, frames):
                raise ValueError('workbook sheets are misaligned')
            frames = ids
            data = np.asarray(rows, dtype=np.float32).reshape(-1, 23, wanted[name])
            if not np.isfinite(data).all():
                raise ValueError('nonfinite workbook')
            result[name] = data
            result[name+' headers'] = labels
    if any(name not in result for name in wanted):
        raise ValueError('missing required sheet')
    return result

def official_modules(directory):
    from fetch_dynaip import CHECKPOINT_SHA256, REVISION, ARCHIVE_SHA256
    directory = directory.resolve()
    provenance = json.loads((Path(__file__).parent/'source_audit/dynaip_provenance.json').read_text(encoding='utf-8'))
    canonical = json.dumps(provenance, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    if hashlib.sha256(canonical).hexdigest() != '7d6dbc3d5bcd83ec7bfbcac7f787176f42db00615e5636f1b6be0a921d31e559':
        raise ValueError('modified DynaIP audit manifest')
    if provenance['revision'] != REVISION or provenance['archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('unreviewed DynaIP revision')
    if json.loads((directory/'provenance.json').read_text(encoding='utf-8')) != provenance:
        raise ValueError('download provenance differs from audit manifest')
    # Validate before executing any downloaded Python, including math and skeleton helpers.
    for path, expected in provenance['files'].items():
        target = (directory/path).resolve()
        if not target.is_relative_to(directory) or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('modified official source')
    if hashlib.sha256((directory/'weights/DynaIP.pth').read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('wrong checkpoint')
    sys.path.insert(0, str(directory))
    import articulate as art
    from model.model import Poser
    from utils.data import normalize_imu
    from utils.read_data import quaternion_product, quaternion_inverse, quaternion_normalize
    from utils.skeleton import XsensSkeleton
    return art, Poser, normalize_imu, quaternion_product, quaternion_inverse, quaternion_normalize, XsensSkeleton

def prepare(content, modules):
    import torch
    art, _, normalize, product, inverse, unit, skeleton_type = modules
    arrays = read_workbook(content)
    # Quaternion [w,x,y,z] -> [w,y,z,x]; vectors [x,y,z] -> [y,z,x].
    segment_q = torch.from_numpy(arrays['Segment Orientation - Quat'][..., [0, 2, 3, 1]].copy())
    sensor_q = torch.from_numpy(arrays['Sensor Orientation - Quat'][:, SENSOR_MASK][..., [0, 2, 3, 1]].copy())
    free_acc = torch.from_numpy(arrays['Sensor Free Acceleration'][:, SENSOR_MASK][..., [1, 2, 0]].copy())
    positions = torch.from_numpy(arrays['Segment Position'][..., [1, 2, 0]].copy())
    quaternion_norms = [q.norm(dim=-1) for q in (segment_q, sensor_q)]
    for norms in quaternion_norms:
        # CIP occasionally stores scaled quaternions (e.g. norm1/3). The
        # official quaternion converter explicitly normalizes them. Zero
        # quaternions on unused workbook columns are omitted by SENSOR_MASK.
        if torch.any(norms < 1e-6):
            raise ValueError('zero quaternion in selected channels')
    offset = product(inverse(sensor_q[:CALIBRATION]), segment_q[:CALIBRATION, SENSOR_MASK])
    dominant = offset.abs().mean(dim=0).argmax(dim=-1)
    for sensor, component in enumerate(dominant):
        offset[:, sensor] *= offset[:, sensor, component:component+1].sign()
    offset = unit(unit(offset).mean(dim=0))
    calibrated = product(sensor_q, offset.repeat(len(sensor_q), 1, 1))
    ori = art.math.quaternion_to_rotation_matrix(calibrated).reshape(-1, 6, 3, 3)
    x = normalize(free_acc, ori).reshape(-1, 6, 12)
    gt = art.math.quaternion_to_rotation_matrix(segment_q).reshape(-1, 23, 3, 3)
    reduced = gt[:, :1].transpose(-1, -2) @ gt[:, JOINT_MASK]
    reduced = reduced[..., :2].transpose(-1, -2).reshape(-1, 11, 6).contiguous()
    # Match official first-frame velocity initialization (reference, including next frame).
    positions[:, :, [0, 2]] -= positions[:, :1, [0, 2]].clone()
    velocity = (positions[1:]-positions[:-1])*FPS
    velocity = torch.cat((velocity[:1], velocity))
    velocity = torch.cat((velocity[:, :1], velocity[:, 1:]-velocity[:, :1]), dim=1) @ ori[:, 0]
    v_init = velocity[:1, VELOCITY_MASK].contiguous()
    p_init = reduced[:1].clone()
    # Canonical mean skeleton (meters), identical for both models and reference.
    canonical = skeleton_type().forward_kinematics(gt)
    features = torch.cat((x.flatten(1), p_init.flatten(1).expand(len(x), -1),
                          v_init.flatten(1).expand(len(x), -1)), dim=1)
    return {'imu': x, 'features': features, 'pose': reduced, 'rotation': gt,
            'position': canonical, 'p_init': p_init, 'v_init': v_init,
            'metadata': {'frames': len(x), 'fps': FPS, 'warmup_excluded': WARMUP,
                         'sensor_mask': SENSOR_MASK, 'sensors': list(SENSORS),
                         'calibration_reference_frames': CALIBRATION,
                         'reference_pose_and_velocity_initialization': True,
                         'temporal_alignment': 'same workbook Frame column',
                         'reference_head_height_feature': False,
                         'sensor_quaternion_norm_min': quaternion_norms[1].min().item(),
                         'scaled_sensor_quaternions': int((torch.abs(quaternion_norms[1]-1)>.02).sum())}}

def assert_untrained(subjects, train_manifest):
    trained = {line.split('_', 1)[0] for line in train_manifest.splitlines() if line.strip()}
    if any('subject'+subject in trained for subject in subjects):
        raise ValueError('evaluation subject occurs in official training manifest')

def select_recordings(archive, subject, allowed):
    names = set(archive.namelist())
    selected = []
    for action in ('sequence', 'circuit', 'task', 'random'):
        options = []
        for rep in range(1, 7):
            base = f'subject{subject}_{action}_rep{rep}'
            key = base+'_trials.pt'
            name = f'MTwAwinda/trials/subject{subject}/subject{subject}_{action}/{base}/{base}.xlsx'
            if name in names and key in allowed:
                options.append((key, name))
        if not options:
            raise ValueError('no preregistered usable recording: '+subject+' '+action)
        selected.append(options[0])
    return selected
