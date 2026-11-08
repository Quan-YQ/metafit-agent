"""Frozen, preregistered class-balanced evaluation using the existing guided renderer."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import vision_guided_eval as guided

HERE = Path(__file__).resolve().parent
NAMES = ('49_02', '69_70', '75_01', '86_09')
# Original experiment digest is retained in balanced_manifest.json.
# This portable revision shares the project renderer; image hashes remain checked.
EXPECTED_GUIDED = 'aaf816b8d0f3404c1f9bf82f22544e0dc2359d6251144aa5171d9b941d329c6e'


def make_window(name, window, reverse):
    path = HERE / 'vision_balanced' / (name + '.bvh')
    joints, frames = guided.load_bvh(path)
    if window == 0:
        start, end = 0, len(frames)
    else:
        start, end = (window - 1) * len(frames) // 4, window * len(frames) // 4
    if end - start < 6:
        raise ValueError('window too short')
    original = guided.load_bvh
    try:
        guided.load_bvh = lambda unused: (joints, frames[start:end])
        ref, image = guided.make_case(path, reverse)
    finally:
        guided.load_bvh = original
    return ref, image, start, end


def prepare(path):
    candidates = []
    for name in NAMES:
        for window in range(5):
            for reverse in (False, True):
                ref, image, start, end = make_window(name, window, reverse)
                candidates.append({'name': name, 'window': window, 'reverse': reverse,
                    'start': start, 'end_exclusive': end, 'reference': ref,
                    'image_sha256': hashlib.sha256(image).hexdigest()})
    selected = []
    for category in ('up', 'down', 'no_clear_change'):
        group = [r for r in candidates if r['reference']['direction'] == category]
        chosen, subjects = [], set()
        for r in group:
            if r['name'] not in subjects:
                chosen.append(r); subjects.add(r['name'])
                if len(chosen) == 3: break
        for r in group:
            if len(chosen) == 3: break
            if r not in chosen: chosen.append(r)
        selected.extend(chosen)
    manifest = {'guided_sha256': EXPECTED_GUIDED, 'candidates': candidates, 'selected': selected,
        'counts': dict(Counter(r['reference']['direction'] for r in selected)),
        'rule': 'first 3 per category by fixed name/window/direction order, distinct sources first; paired or overlapping windows are not independent'}
    with path.open('x', encoding='utf-8') as f: json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({'candidates': len(candidates), 'selected': len(selected), 'counts': manifest['counts']}))


def live(manifest_path, output):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key: raise ValueError('missing API key')
    usage_in = usage_out = 0
    with output.open('x', encoding='utf-8') as f:
        for item in manifest['selected']:
            ref, image, start, end = make_window(item['name'], item['window'], item['reverse'])
            if ref != item['reference'] or hashlib.sha256(image).hexdigest() != item['image_sha256'] or (start, end) != (item['start'], item['end_exclusive']):
                raise ValueError('manifest mismatch')
            row = dict(item)
            try:
                body = json.loads(guided.build_payload(image, guided.PROMPT))
                body.update(response_format={'type': 'json_object'}, max_tokens=160)
                response, usage, elapsed = guided.call_once(json.dumps(body, ensure_ascii=False).encode('utf-8'), key)
                if not {'prompt_tokens', 'completion_tokens'}.issubset(usage): raise ValueError('usage missing')
                usage_in += usage['prompt_tokens']; usage_out += usage['completion_tokens']
                obs = guided.compare(response, ref)
                row.update(observation=obs['model_observation']['hip_change'], comparison=obs['comparison'], usage=usage)
                if (usage_in * .3 + usage_out * 1.2) / 1_000_000 * 10 > 1: raise ValueError('budget guard')
            except ValueError:
                row['error'] = 'call_or_format_error'
            f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
            if 'error' in row: raise ValueError('batch stopped after error')
    print(json.dumps({'requests': len(manifest['selected']), 'input_tokens': usage_in, 'output_tokens': usage_out}))


if __name__ == '__main__':
    if hashlib.sha256((HERE / 'vision_guided_eval.py').read_bytes()).hexdigest() != EXPECTED_GUIDED:
        raise ValueError('guided evaluator changed')
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('prepare', 'live'))
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.manifest)
    else: live(args.manifest, args.output)
