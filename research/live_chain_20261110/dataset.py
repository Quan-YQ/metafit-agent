"""Frozen public-IMU inputs; image references describe predictions, not pose truth."""
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
SOURCES = expanded.TEST


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


@lru_cache(maxsize=2)
def prediction_data(data_dir):
    # Fixed existing five-recording/alpha=200 baseline, no new hyperparameter search.
    expanded.OLD = expanded.NEW = Path(data_dir)
    records = {name: expanded.read_pair(name) for name in TRAIN + SOURCES}
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
        'schema': 'fusionposer-frozen-expanded-chain-1',
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
    _, _, _, records = prediction_data(str(Path(data_dir).resolve()))
    cases = []
    # Three clusters per previously exposed development recording. Natural contiguous
    # windows, paired reversal, plus repeated-frame decoys. No pose labels selected.
    for source_index, name in enumerate(SOURCES):
        n = len(records[name][0])
        for window in range(3):
            start, end = round(window*n/3), round((window+1)*n/3)-1
            rows = np.linspace(start, end, 6).round().astype(int).tolist()
            variants = (('forward', rows), ('reverse', rows[::-1]), ('static', [rows[0]]*6))
            for order, selected in variants:
                case = {'id': f'imu_{source_index+1}_{window+1}_{order}', 'recording': name,
                        'rows': selected, 'frame_order': order,
                        'cluster': f'{name}_{window}',
                        'stage': 'confirmation'}
                report, image, _ = bundle(data_dir, case)
                (output/(case['id']+'.png')).write_bytes(image)
                case.update(reference=report['predicted_hip_reference'],
                    image_sha256=hashlib.sha256(image).hexdigest(), report=report,
                    prediction_report_sha256=digest(report))
                cases.append(case)
    # Frozen geometric stratification, before any model response. Three pilot items,
    # one per available class. Confirmation keeps all other items, including failures.
    for label in ('up', 'down', 'no_clear_change'):
        next(c for c in cases if c['reference']['direction'] == label)['stage'] = 'pilot'
    manifest = {'schema': 'imu-vision-chain-dataset-1', 'prompt': PROMPT,
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(), 'cases': cases,
        'exposure': 'Both recordings were used as development data in prior pose experiments; these images are new to this round. Variants share clusters, not independent people.',
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
    print(json.dumps({'cases': len(data['cases']), 'labels': {label: sum(c['reference']['direction']==label for c in data['cases'])
        for label in ('up', 'down', 'no_clear_change')}}))
