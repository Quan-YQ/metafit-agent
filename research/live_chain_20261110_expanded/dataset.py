"""Expanded public-IMU set: five unseen recordings, six windows, three variants.

Image references describe the prediction geometry, not pose truth. The five
recordings below were not used to fit the fixed ridge baseline.
"""
import argparse
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'examples'))
import fusionposer_expanded as expanded
from fusionposer_research import make_render_frames, predicted_hip_reference
from render_motion_sample import render_frames, world_positions
from check_vision_observation import PROMPT

TRAIN = expanded.OLD_TRAIN + expanded.ADDITIONAL_TRAIN
SOURCES = ('0625173852', '0625174451', '0625181450', '0625181811', '0625182119')
WINDOWS = 6
VARIANTS = (('forward', False), ('reverse', True), ('static', True))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_pair(data_dir, identifier):
    x = np.load(Path(data_dir) / f'motion_{identifier}_x.npz', allow_pickle=False, mmap_mode='r')
    y = np.load(Path(data_dir) / f'motion_{identifier}_y.npz', allow_pickle=False, mmap_mode='r')
    if x.ndim != 2 or x.shape[1] != 430 or y.ndim != 2 or y.shape[1] != 146:
        raise ValueError(f'数据形状无效: {identifier}')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'数据含非有限值: {identifier}')
    return np.array(x[:, 6 * 43:7 * 43]), np.array(y[:, :60])


@lru_cache(maxsize=2)
def prediction_data(data_dir):
    # Fixed existing five-recording/alpha=200 baseline, no new hyperparameter search.
    expanded.OLD = expanded.NEW = Path(data_dir)
    records = {name: read_pair(data_dir, name) for name in TRAIN + SOURCES}
    model = expanded.fit(records, TRAIN)
    model_hash = digest({'means': model[0].tolist(), 'scales': model[1].tolist(),
                         'weights': model[2].tolist(), 'alpha': expanded.ALPHA})
    hashes = {name: {role: sha(Path(data_dir) / f'motion_{name}_{role}.npz')
                     for role in ('x', 'y')} for name in TRAIN + SOURCES}
    return model, model_hash, hashes, records


def draw(joints, frames):
    image = Image.open(BytesIO(render_frames(joints, frames, draw_root=True))).convert('RGB')
    canvas = ImageDraw.Draw(image)
    poses = [world_positions(joints, frame) for frame in frames]
    low = min(p[1] for pose in poses for p in pose)
    high = max(p[1] for pose in poses for p in pose)
    extent = max(max(abs(p[0]-pose[0][0]), abs(p[2]-pose[0][2])) for pose in poses for p in pose)
    scale = min(155/max(high-low, 1), 78/max(extent, 1))
    for col, pose in enumerate(poses):
        for row in (0, 1):
            x, y = 100+col*200, round(row*230+205-(pose[0][1]-low)*scale)
            canvas.ellipse((x-5, y-5, x+5, y+5), fill=(12, 150, 85))
            canvas.text((col*200+12, row*230+8), str(col+1), fill=(37, 73, 130))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def bundle(data_dir, case):
    model, model_hash, hashes, records = prediction_data(str(Path(data_dir).resolve()))
    name, indices = case['recording'], case['rows']
    for identifier in TRAIN + (name,):
        for role in ('x', 'y'):
            if sha(Path(data_dir)/f'motion_{identifier}_{role}.npz') != hashes[identifier][role]:
                raise ValueError('source changed after prediction cache creation')
    x = records[name][0]
    if len(indices) != 6 or any(type(i) is not int or not 0 <= i < len(x) for i in indices):
        raise ValueError('invalid selected IMU rows')
    features = x[indices]
    prediction = expanded.predictions(model, features)
    joints, frames = make_render_frames(prediction.tolist(), features[:, 0].tolist())
    report = {
        'schema': 'fusionposer-frozen-expanded-chain-2',
        'x_sha256': hashes[name]['x'], 'y_sha256': hashes[name]['y'],
        'training_hashes': {key: hashes[key] for key in TRAIN},
        'training_recordings': list(TRAIN), 'training_frames': model[4],
        'predictor': 'fixed_five_recording_ridge_alpha200', 'predictor_sha256': model_hash,
        'recording': name, 'selected_original_rows': indices,
        'frame_order': case['frame_order'], 'case_id': case['id'],
        'predicted_hip_reference': predicted_hip_reference(joints, frames),
        'input': 'Six IMUs plus public head-height input; preprocessed 43 features at window index 6.',
        'source_commit': 'e8ba6f89b2f6cc031b37c55dd72fc674192df86f',
        'reference_scope': 'Geometry of predicted skeleton; no optical target supplied to vision API.',
    }
    return report, draw(joints, frames), b''


def prepare(data_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _, _, hashes, records = prediction_data(str(Path(data_dir).resolve()))
    cases = []
    # Natural contiguous windows per unseen recording; no window was selected by
    # its label. Reversal and repeated-frame variants are constructed controls.
    for source_index, name in enumerate(SOURCES):
        n = len(records[name][0])
        for window in range(WINDOWS):
            start, end = round(window*n/WINDOWS), round((window+1)*n/WINDOWS)-1
            rows = np.linspace(start, end, 6).round().astype(int).tolist()
            for order, reversed_frames in VARIANTS:
                selected = rows[::-1] if reversed_frames and order == 'reverse' else ([rows[0]]*6 if order == 'static' else rows)
                case = {'id': f'imu_x{source_index+1}_{window+1}_{order}', 'recording': name,
                        'rows': selected, 'frame_order': order,
                        'cluster': f'{name}_{window}', 'stage': 'confirmation'}
                report, image, _ = bundle(data_dir, case)
                (output/(case['id']+'.png')).write_bytes(image)
                case.update(reference=report['predicted_hip_reference'],
                    image_sha256=hashlib.sha256(image).hexdigest(), report=report,
                    prediction_report_sha256=digest(report))
                cases.append(case)
    # One pilot item per geometric class, preferring a non-static frame; the
    # remaining cases, including constructed controls, stay in confirmation.
    for label in ('up', 'down', 'no_clear_change'):
        candidates = [c for c in cases if c['reference']['direction'] == label]
        chosen = next((c for c in candidates if c['frame_order'] != 'static'), candidates[0])
        chosen['stage'] = 'pilot'
    for check in hashes:
        for role in ('x', 'y'):
            if sha(Path(data_dir)/f'motion_{check}_{role}.npz') != hashes[check][role]:
                raise ValueError('source changed during dataset generation')
    labels = {label: sum(c['reference']['direction'] == label for c in cases)
              for label in ('up', 'down', 'no_clear_change')}
    manifest = {'schema': 'imu-vision-chain-dataset-2', 'prompt': PROMPT,
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'cases': cases,
        'recordings': list(SOURCES), 'windows_per_recording': WINDOWS,
        'variants': [name for name, _ in VARIANTS], 'class_counts': labels,
        'training_recordings': list(TRAIN),
        'exposure': 'All five source recordings are separate from the five training recordings. '
                    'Forward/reverse/static variants share a source window and are not independent people.',
        'reference': 'Prediction geometry, not physical truth. Static/reverse controls are constructed.',
        'numpy_version': np.__version__}
    save(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT/'data/fusionposer')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = prepare(args.data_dir, args.output)
    print(json.dumps({'cases': len(data['cases']), 'classes': data['class_counts'],
                      'pilot': [c['id'] for c in data['cases'] if c['stage']=='pilot']}))
