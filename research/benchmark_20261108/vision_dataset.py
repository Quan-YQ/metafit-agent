"""Deterministic public CMU images; disjoint source subjects in selection and confirmation."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'examples'))
from render_motion_sample import load_bvh, world_positions, render_frames

CMU_REVISION = '09a07f54f3bbb58797325f009282d0b2048a2871'
DEVELOPMENT = ('49_02', '69_70', '75_01', '86_09')
# These subjects did not enter DEVELOPMENT. Their images are not rendered until
# after model selection is frozen. Prior marker-reader experiments used these
# source recordings, but our 8-way subwindows are new to the vision API study.
CONFIRMATION = ('01_06', '15_10', '40_06', '88_08', '114_16', '136_09', '141_17', '143_18')
SOURCE_SHA256 = {
    '49_02':'cf201c93b7cf1337c8a41d74c96d2062a51b621a9b1e272961e1fe4dd6590924',
    '69_70':'fac827b41e02faeaa68ed6b24b6470e00db22be252805d9b3724c0cc97cfffb5',
    '75_01':'944504bc271fcfa57c918ba755dc98beed21e2a65783f27a72bf5d1bfc4d0f77',
    '86_09':'9c4138c29a481dd3ac87eeaf827c5c8e403351a16a0c2644cca367f8847df4b4',
    '01_06':'8882922f6d51ab25cf7c44ac31e23ad21b5deaebc3332c74b01e2a61cb751b1a',
    '15_10':'03d58e8f3731b4bd1c5b9ac73bc1955dd827f4171a2e16628d0c89f7a935bdde',
    '40_06':'9fb48c767f91189e3c9b293dd2d3a6c1326cc834a357fa27abe197ff0afffd54',
    '88_08':'c09ca52ae671d90f5c3086f7a5841e6a73dc8e07731a98e127171da1fb19c693',
    '114_16':'702952e8521d615e4425e569a95f8913d11d932ec7ea73f809ce96fda747790f',
    '136_09':'f958ad4cfc516b025da94eb1085b7ded2f34e0fad2c0520fb554a29794b28c33',
    '141_17':'50480e797770f6ae881caba3f27746a750bbdae237342bb72b2723df9f0663c7',
    '143_18':'1b55e3b6fe9254c14acc5cc53cd6e8ede16d21682029c4ccf56d374d612f508f'}
PROMPT = ('图片展示同一人物按从左到右排列的六个时刻，上下两排是两个视角。'
          '紫色大点标出髋部，绿色虚线位于第一列髋部上下、整幅骨架高度的5%处。'
          '比较第六列与第一列：第六列紫点在上方绿线之外表示升高up，'
          '在下方绿线之外表示降低down，在两条线之间表示no_clear_change；'
          '看不清或无法判断时答uncertain。只按图像判断，不推测动作名称。'
          '只输出JSON：{"hip_change":"up|down|no_clear_change|uncertain","explanation":"一句话"}。')
CATEGORIES = ('up', 'down', 'no_clear_change')

def get_source(name, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/(name+'.bvh')
    if not path.exists():
        url = f'https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/{CMU_REVISION}/data/{int(name.split("_")[0]):03d}/{name}.bvh'
        with urlopen(Request(url, headers={'User-Agent': 'MetaFit-public-research/1.0'}), timeout=45) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000:
            raise ValueError('oversized CMU recording')
        with path.open('xb') as handle:
            handle.write(content)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA256[name]:
        raise ValueError('public source digest mismatch: '+name)
    return path

def render(joints, frames):
    indices = [round(i*(len(frames)-1)/5) for i in range(6)]
    selected = [frames[i] for i in indices]
    poses = [world_positions(joints, frame) for frame in selected]
    root = next(i for i, joint in enumerate(joints) if joint.name == 'Hips')
    low = min(point[1] for pose in poses for point in pose)
    high = max(point[1] for pose in poses for point in pose)
    span = high-low
    if span <= 1e-9:
        raise ValueError('zero skeleton span')
    extent = max(max(abs(point[0]-pose[0][0]), abs(point[2]-pose[0][2])) for pose in poses for point in pose)
    scale = min(155/max(span, 1), 78/max(extent, 1))
    ratio = (poses[-1][root][1]-poses[0][root][1])/span
    direction = 'up' if ratio > .05 else 'down' if ratio < -.05 else 'no_clear_change'
    image = Image.open(io.BytesIO(render_frames(joints, selected))).convert('RGB')
    draw = ImageDraw.Draw(image)
    for row in range(2):
        bottom = row*230+205
        base = bottom-(poses[0][root][1]-low)*scale
        for y in (round(base-span*scale*.05), round(base+span*scale*.05)):
            for x in range(12, 1192, 12):
                draw.line((x, y, min(x+6, 1198), y), fill=(0, 145, 70), width=1)
        for column, pose in enumerate(poses):
            x = 100+column*200; y = round(bottom-(pose[root][1]-low)*scale)
            draw.ellipse((x-6, y-6, x+6, y+6), fill=(170, 0, 190), outline=(255,255,255), width=1)
    content = io.BytesIO(); image.save(content, format='PNG')
    return content.getvalue(), {'direction': direction, 'height_ratio': ratio, 'threshold': .05}

def prepare(stage, output, cache, freeze=None):
    if output.exists():
        raise ValueError('new image directory required')
    if stage == 'confirmation':
        if freeze is None or not freeze.is_file():
            raise ValueError('freeze candidate selection before generating confirmation images')
        decision = json.loads(freeze.read_text(encoding='utf-8'))
        if not decision.get('candidates') or not decision.get('selection_results_sha256'):
            raise ValueError('invalid frozen selection')
    subjects = DEVELOPMENT if stage == 'selection' else CONFIRMATION
    candidates, images, sources = [], {}, {}
    for name in subjects:
        path = get_source(name, cache)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        joints, frames = load_bvh(path)
        # Eight disjoint windows per recording; opposite directions remain paired.
        for window in range(8):
            start, end = window*len(frames)//8, (window+1)*len(frames)//8
            if end-start < 6:
                continue
            for reverse in (False, True):
                clip = frames[start:end]
                image, reference = render(joints, list(reversed(clip)) if reverse else clip)
                # Keep labels >=1% of full height away from the boundary, avoiding
                # reference rounding ambiguity without using any model response.
                if abs(abs(reference['height_ratio'])-.05) < .01:
                    continue
                case = {'id': f'{name}_w{window}_{"r" if reverse else "f"}', 'source': name,
                        'subject': name.split('_')[0], 'window': window, 'reverse': reverse,
                        'start': start, 'end_exclusive': end, 'reference': reference,
                        'image_sha256': hashlib.sha256(image).hexdigest()}
                candidates.append(case); images[case['id']] = image
    selected = []
    for category in CATEGORIES:
        group = [case for case in candidates if case['reference']['direction']==category]
        # Greedy prefer new source recordings within each class, then new windows.
        chosen = []
        while group and len(chosen)<6:
            used_sources = {case['source'] for case in chosen}
            used_windows = {(case['source'],case['window']) for case in chosen}
            group.sort(key=lambda case: (case['source'] in used_sources,
                       (case['source'],case['window']) in used_windows, case['id']))
            chosen.append(group.pop(0))
        if len(chosen) != 6:
            raise ValueError('insufficient class coverage: '+category)
        selected.extend(chosen)
    # Models see class-interleaved cases, without the reference in the request.
    selected = [selected[class_index*6+index] for index in range(6) for class_index in range(3)]
    output.mkdir(parents=True)
    for case in selected:
        (output/(case['id']+'.png')).write_bytes(images[case['id']])
    manifest = {'schema':'public-skeleton-vision-v1', 'stage':stage, 'prompt':PROMPT,
                'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
                'source_repository':'https://github.com/una-dinosauria/cmu-mocap',
                'source_revision':CMU_REVISION, 'source_sha256':sources, 'cases':selected,
                'counts':dict(Counter(case['reference']['direction'] for case in selected)),
                'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selection_freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest() if freeze else None,
                'scope':'controlled CMU-derived skeletons with guides; windows/reversals are clustered, not independent people',
                'license':'CMU mocap data available for research/use with attribution; source conversion MIT, retain authors and source notices'}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'stage':stage, 'images':len(selected), 'counts':manifest['counts']}))
    return manifest

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('selection','confirmation'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=PROJECT/'.tmp/vision_sources_20261108')
    parser.add_argument('--selection-freeze', type=Path)
    args=parser.parse_args()
    prepare(args.stage,args.output,args.cache,args.selection_freeze)
