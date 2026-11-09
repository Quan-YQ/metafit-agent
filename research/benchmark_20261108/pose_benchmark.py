"""Train an input-aligned small baseline and compare with frozen official DynaIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch

from cip_shared import (JOINT_MASK, WARMUP, assert_untrained, official_modules,
                        prepare, select_recordings)
from fetch_dynaip import PROJECT, REVISION, CHECKPOINT_SHA256

def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def full_rotation(reduced, imu, net):
    # Small model outputs the same 11 rotations in canonical JOINT_MASK order.
    return net._reduced_glb_6d_to_full_glb_mat_xsens(reduced, imu[..., :9].reshape(-1, 6, 3, 3).clone())

def measure(prediction, record, modules):
    skeleton = modules[-1]()
    truth = record['rotation'][WARMUP:]
    pred = prediction[WARMUP:]
    for rotations in (pred, truth):
        if not torch.isfinite(rotations).all():
            raise ValueError('nonfinite pose output')
        residual=rotations.transpose(-1,-2)@rotations-torch.eye(3)
        if residual.abs().max()>1e-3 or (torch.linalg.det(rotations)-1).abs().max()>1e-3:
            raise ValueError('invalid SO3 pose output')
    distance = (skeleton.forward_kinematics(pred)[:, 1:]-record['position'][WARMUP:, 1:]).norm(dim=-1)*100
    difference = pred[:, JOINT_MASK].transpose(-1, -2) @ truth[:, JOINT_MASK]
    cosine = (difference.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    return {'frames': len(truth), 'canonical22_mpjpe_cm': distance.mean().item(),
            'p95_frame_mpjpe_cm': torch.quantile(distance.mean(-1), .95).item(),
            'learned11_geodesic_deg': angle.mean().item()}

def baseline_predict(model, record, net):
    x = (record['features']-model['mean'])/model['scale']
    if model['kind'] == 'ridge':
        reduced = torch.cat((torch.ones(len(x), 1), x), dim=1) @ model['weights']
    else:
        reduced = model['network'](x)
    return full_rotation(reduced.reshape(-1, 11, 6), record['imu'], net)

def macro(model, records, net, modules):
    with torch.inference_mode():
        return float(np.mean([measure(baseline_predict(model, record, net), record, modules)['canonical22_mpjpe_cm']
                              for record in records.values()]))

def train_baselines(records, validation, net, modules, output):
    xs, ys = [], []
    for record in records.values():
        indices = torch.linspace(WARMUP, len(record['imu'])-1, min(3000, len(record['imu'])-WARMUP)).long()
        xs.append(record['features'][indices]); ys.append(record['pose'][indices].flatten(1))
    x = torch.cat(xs); y = torch.cat(ys)
    mean = x.mean(0); scale = x.std(0).clamp_min(1e-5)
    z = (x-mean)/scale
    candidates, trials = [], []
    zd = torch.cat((torch.ones(len(z), 1), z), dim=1).double()
    for alpha in (10., 100., 1000.):
        penalty = torch.eye(zd.shape[1], dtype=torch.float64)*alpha
        penalty[0, 0] = 0
        weights = torch.linalg.solve(zd.T@zd+penalty, zd.T@y.double()).float()
        model = {'kind': 'ridge', 'mean': mean, 'scale': scale, 'weights': weights}
        score = macro(model, validation, net, modules)
        candidates.append((score, 'ridge_'+str(alpha), model))
        trials.append({'name': candidates[-1][1], 'validation_mpjpe_cm': score})
    torch.manual_seed(42)
    network = torch.nn.Sequential(torch.nn.Linear(156, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 66))
    model = {'kind': 'mlp', 'mean': mean, 'scale': scale, 'network': network}
    optimizer = torch.optim.Adam(network.parameters(), lr=.0005, weight_decay=.0001)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(z, y), batch_size=512,
                                          shuffle=True, generator=torch.Generator().manual_seed(42))
    best, best_state, best_epoch, stale = float('inf'), None, None, 0
    history = []
    for epoch in range(1, 31):
        network.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(network(bx), by)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite baseline training')
            loss.backward(); optimizer.step()
        network.eval()
        score = macro(model, validation, net, modules)
        history.append({'epoch': epoch, 'validation_mpjpe_cm': score})
        print(json.dumps({'epoch': epoch, 'validation_mpjpe_cm': round(score, 4)}), flush=True)
        if score < best-1e-6:
            best, best_state, best_epoch, stale = score, {k: v.clone() for k, v in network.state_dict().items()}, epoch, 0
        else:
            stale += 1
            if stale >= 5:
                break
    network.load_state_dict(best_state); network.eval()
    candidates.append((best, 'mlp', model))
    trials.append({'name': 'mlp', 'validation_mpjpe_cm': best, 'best_epoch': best_epoch, 'history': history})
    score, name, selected = min(candidates, key=lambda row: (row[0], row[1]))
    saved = {k: v for k, v in selected.items() if k != 'network'}
    if selected['kind'] == 'mlp':
        saved['state_dict'] = selected['network'].state_dict()
    torch.save(saved, output/'aligned_baseline.pth')
    write_json(output/'selection_frozen.json', {'candidates': trials, 'selected': name,
               'seed': 42, 'training_frames': len(x), 'train_subjects': ['00', '01', '02'],
               'validation_subjects': ['03'], 'test_subjects': ['04'],
               'selection_metric': 'validation recording macro canonical22_mpjpe_cm',
               'baseline_sha256': sha256(output/'aligned_baseline.pth')})
    return selected, name

def timing(predictors, record):
    # Single CPU, identical 600-frame sequences, alternating order, warmed up.
    clip = {k: (v[:600].clone() if isinstance(v, torch.Tensor) and len(v) == len(record['imu']) else v)
            for k, v in record.items()}
    values = {name: [] for name in predictors}
    with torch.inference_mode():
        for function in predictors.values():
            function(clip)
        for repeat in range(5):
            for name in (list(predictors) if repeat % 2 == 0 else list(predictors)[::-1]):
                started = time.perf_counter()
                predictors[name](clip)
                values[name].append((time.perf_counter()-started)*1000)
    return {name: {'sequence_frames': len(clip['imu']), 'repeats_ms': measured,
                   'median_sequence_ms': float(np.median(measured)),
                   'amortized_ms_per_frame': float(np.median(measured)/len(clip['imu'])),
                   'frames_per_second': float(len(clip['imu'])*1000/np.median(measured)),
                   'mode': 'CPU full-sequence throughput; not single-frame online latency'}
            for name, measured in values.items()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('CIP_DATA_DIR'))
    parser.add_argument('--official-dir', type=Path, default=PROJECT/'.tmp/dynaip_official')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data_dir is None:
        raise ValueError('provide --data-dir or CIP_DATA_DIR')
    if args.output.exists():
        raise ValueError('new output directory required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    modules = official_modules(args.official_dir)
    _, poser, _, _, _, _, _ = modules
    net = poser().cpu().eval()
    net.load_state_dict(torch.load(args.official_dir/'weights/DynaIP.pth', map_location='cpu', weights_only=True), strict=True)
    split_dir = args.official_dir/'datasets/split_info/xsens'
    official_train = (split_dir/'cip_train.txt').read_text()
    official_test = (split_dir/'cip_test.txt').read_text()
    assert_untrained(('04', '08'), official_train)
    train, validation, metadata = {}, {}, {}
    def group(subjects, destination, allowed):
        for subject in subjects:
            path = args.data_dir/f'CIP_MTwAwinda_subject{subject}.zip'
            # Published integrity check before use, same constants as preserved CIP study.
            sys.path.insert(0, str(PROJECT/'research/continuous_20261108'))
            import cip_prepare as original
            expected_size, expected_md5 = original.ARCHIVES[subject]
            if path.stat().st_size != expected_size:
                raise ValueError('CIP archive size mismatch')
            digest = hashlib.md5()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(4*1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected_md5:
                raise ValueError('CIP archive MD5 mismatch')
            with zipfile.ZipFile(path) as archive:
                for key, member in select_recordings(archive, subject, set(allowed.splitlines())):
                    content = archive.read(member)
                    record = prepare(content, modules)
                    record['metadata'].update(subject=subject, official_recording=key,
                        workbook_sha256=hashlib.sha256(content).hexdigest(), archive_md5=expected_md5)
                    destination[key] = record
                    metadata[key] = record['metadata']
                    print(json.dumps({'prepared': key, 'frames': len(record['imu'])}), flush=True)
    group(('00', '01', '02'), train, official_train)
    group(('03',), validation, official_train)
    baseline, name = train_baselines(train, validation, net, modules, args.output)
    # The baseline choice and weight digest are on disk before any holdout workbook opens.
    holdout = {}
    group(('04',), holdout, official_test)
    def dyna(record):
        return net.predict(record['imu'].clone(), record['v_init'], record['p_init'])[0]
    def small(record):
        return baseline_predict(baseline, record, net)
    predictions = {'DynaIP': dyna, 'aligned_small': small}
    results = {}
    with torch.inference_mode():
        for key, record in holdout.items():
            results[key] = {model: measure(function(record), record, modules) for model, function in predictions.items()}
    summary = {model: {metric: float(np.mean([row[model][metric] for row in results.values()]))
                       for metric in ('canonical22_mpjpe_cm', 'learned11_geodesic_deg')}
               for model in predictions}
    result = {'schema': 'dynaip-aligned-cip-v1', 'revision': REVISION,
              'checkpoint_sha256': CHECKPOINT_SHA256, 'baseline_selected': name,
              'shared_information': '6x12 normalized IMU + first reference pose(11x6) and velocity(6x3)',
              'calibration': 'first150 reference segment rotations; same-workbook time; exclude first300 scoring frames',
              'position_reference': 'official canonical mean Xsens skeleton in meters; exclude pelvis; no GT rotation overwrite',
              'test_scope': 'subject04: absent from both training sets; same dataset, not external-dataset generalization',
              'prior_test_exposure': 'old subject04 scores existed before this study; no new holdout tuning',
              'recordings': metadata, 'per_recording': results, 'macro': summary,
              'timing': timing(predictions, next(iter(holdout.values()))),
              'parameters': {'DynaIP': sum(p.numel() for p in net.parameters()),
                  'aligned_small': (sum(p.numel() for p in baseline['network'].parameters())
                                    if baseline['kind']=='mlp' else baseline['weights'].numel())},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'torch': torch.__version__, 'device': 'cpu', 'threads': 4,
                              'os': platform.platform(), 'processor': platform.processor()}}
    write_json(args.output/'results.json', result)
    print(json.dumps({'macro': summary, 'timing': result['timing']}, indent=2))

if __name__ == '__main__':
    main()
