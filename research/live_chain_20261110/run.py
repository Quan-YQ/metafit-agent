"""Freeze, pilot, compare and demonstrate a public IMU-to-Agent chain."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

from dataset import PROJECT, PROMPT, bundle, digest, save, sha
from client import Caller, Choice, Observer, read_key, registry
from check_vision_observation import compare
from fusionposer_imu_agent import run_pipeline
from metafit_agent.decision import ModelCallError

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hashes():
    paths = [*sorted(HERE.glob('*.py')), HERE/'models.json',
        PROJECT/'examples/fusionposer_imu_agent.py', PROJECT/'examples/fusionposer_expanded.py',
        PROJECT/'examples/fusionposer_research.py', PROJECT/'examples/render_motion_sample.py',
        PROJECT/'examples/research_motion_choice.py', PROJECT/'examples/check_vision_observation.py',
        PROJECT/'research/benchmark_20261108/vision_benchmark.py',
        *sorted((PROJECT/'metafit_agent').glob('*.py'))]
    return {p.relative_to(PROJECT).as_posix(): sha(p) for p in paths}


def freeze(manifest, output, model_ids):
    data = read(manifest)
    models = registry()
    chosen = [models[key] for key in model_ids]
    if len(set(model_ids)) != len(model_ids) or not model_ids:
        raise ValueError('invalid selected models')
    if hashlib.sha256(data['prompt'].encode()).hexdigest() != data['prompt_sha256'] or data['prompt'] != PROMPT:
        raise ValueError('prompt changed')
    for case in data['cases']:
        if sha(manifest.parent/(case['id']+'.png')) != case['image_sha256']:
            raise ValueError('image changed')
    output.mkdir(parents=True, exist_ok=False)
    # Copy inputs into self-contained run; no absolute paths in distributable evidence.
    (output/'images').mkdir()
    for case in data['cases']:
        (output/'images'/(case['id']+'.png')).write_bytes((manifest.parent/(case['id']+'.png')).read_bytes())
    save(output/'manifest.json', data)
    value = {'schema': 'live-chain-preregistration-1', 'model_ids': model_ids, 'models': chosen,
        'manifest_sha256': sha(output/'manifest.json'), 'code_hashes': code_hashes(),
        'budget_cap_yuan': 10, 'max_output_tokens': 8192,
        'pilot_items_per_model': 3,
        'pilot_bars': {'correct_min': 2, 'invalid_max': 0, 'unknown_cost_max': 0},
        'selection_rule': ['balanced_accuracy_desc', 'accuracy_desc', 'uncertain_rate_asc', 'cost_asc', 'latency_asc'],
        'stops': ['transport_or_auth_error_stops_provider', 'unknown_charge_stops_provider',
                  'pilot_below_bar_excluded', 'ledger_admission_refused_stops_all', 'no_automatic_paid_retry'],
        'decoys': 'Repeated-frame static controls; uncertainty and invalid count as incorrect. Reversal pairs share a source cluster.',
        'blinding': 'Only PNG and frozen prompt sent to vision; no filename, class, reference, pose error, or model label. Programmatic geometry scorer sees class labels only.',
        'heldout_exposure': data['exposure'],
        'cost_envelope': 'One fixed 1200x460 PNG; <=16KiB text; reserve 65,536 input and 8,192 output tokens. Engineering envelope, usage checked after each call; not a provider billing guarantee.',
        'scope': 'Public FusionPoser IMU + head-height → fixed expanded ridge → image observation → simulated event choice → R3 review.'}
    save(output/'freeze.json', value)
    save(output/'dry_run.json', dry_run(output, value))
    return value


def validated(run, *, allow_postprocessing=False):
    frozen = read(run/'freeze.json')
    current = code_hashes()
    if current != frozen['code_hashes']:
        amendment_path = run/'postrun_amendments.json'
        if not allow_postprocessing or not amendment_path.exists():
            raise ValueError('frozen code changed; no further API calls allowed for this run')
        amendment = read(amendment_path)
        allowed = {'research/live_chain_20261110/run.py', 'research/live_chain_20261110/replay.py'}
        changed = {key for key in set(current)|set(frozen['code_hashes'])
                   if current.get(key) != frozen['code_hashes'].get(key)}
        if (not changed <= allowed or amendment['freeze_sha256'] != sha(run/'freeze.json')
                or set(amendment['files']) != changed
                or any(amendment['files'][key] != {'before': frozen['code_hashes'][key], 'after': current[key]} for key in changed)):
            raise ValueError('unregistered postprocessing changes')
    if sha(run/'manifest.json') != frozen['manifest_sha256']:
        raise ValueError('frozen manifest changed')
    manifest = read(run/'manifest.json')
    for case in manifest['cases']:
        if sha(run/'images'/(case['id']+'.png')) != case['image_sha256'] or digest(case['report']) != case['prediction_report_sha256']:
            raise ValueError('frozen image or prediction report changed')
    if read(run/'dry_run.json')['status'] != 'pass':
        raise ValueError('dry run did not pass')
    return frozen, manifest


def dry_run(run, frozen):
    cases = read(run/'manifest.json')['cases']
    for case in cases:
        ref = case['reference']
        for answer in ('up', 'down', 'no_clear_change', 'uncertain'):
            checked = compare(json.dumps({'hip_change': answer, 'explanation': '可见骨架变化。'}), ref)
            if (checked['comparison'] == 'matches_reference') != (answer == ref['direction']):
                raise ValueError('scorer failed')
        if case['frame_order'] == 'static' and ref['direction'] != 'no_clear_change':
            raise ValueError('static decoy failed')
    if len([c for c in cases if c['stage']=='pilot']) != 3:
        raise ValueError('pilot not three items')
    return {'status': 'pass', 'network_calls': 0, 'cases': len(cases),
        'scorer_controls_checked': len(cases)*4, 'static_controls': sum(c['frame_order']=='static' for c in cases),
        'frozen_models': frozen['model_ids']}


def score(records, cases):
    expected = {case['id']: case for case in cases}
    rows = {record['case_id']: record for record in records}
    if len(rows) != len(records) or not set(rows) <= set(expected):
        raise ValueError('duplicate or foreign case')
    counts = Counter(c['reference']['direction'] for c in cases)
    correct = Counter(expected[key]['reference']['direction'] for key, row in rows.items()
        if row.get('comparison') == 'matches_reference')
    return {'planned': len(cases), 'completed': len(rows), 'correct': sum(correct.values()),
        'accuracy': sum(correct.values())/len(cases),
        'balanced_accuracy': statistics.mean(correct[key]/count for key, count in counts.items()),
        'per_class': {key: {'correct': correct[key], 'total': count} for key, count in counts.items()},
        'invalid_or_missing': sum(bool(row.get('error')) for row in records)+len(cases)-len(rows),
        'uncertain': sum(row.get('answer') == 'uncertain' for row in records),
        'unknown_cost': sum(row.get('estimated_yuan') is None for row in records),
        'estimated_yuan': sum(row.get('estimated_yuan') or 0 for row in records),
        'median_ms': statistics.median([r['elapsed_ms'] for r in records]) if records else None}


def comparison(run, stage):
    frozen, manifest = validated(run)
    models = {m['id']: m for m in frozen['models']}
    admitted = list(models)
    if stage == 'confirmation':
        pilot = read(run/'pilot.json')
        admitted = pilot['admitted_models']
        if not admitted:
            raise ValueError('no model passed pilot')
    cases = [c for c in manifest['cases'] if c['stage'] == stage]
    responses = run/(stage+'_responses.json')
    if responses.exists():
        raise ValueError('stage already exists; replay it instead')
    records, failed = [], set()
    stop_all = False
    for index, case in enumerate(cases):
        if stop_all:
            break
        for model_id in admitted[index % len(admitted):]+admitted[:index % len(admitted)]:
            if stop_all:
                break
            if models[model_id]['provider'] in failed:
                continue
            model = models[model_id]
            caller = Caller(model, run/'budget.json', run/'calls')
            identity = stage+'__'+case['id']+'__'+model_id
            observer = Observer(caller, manifest['prompt'], identity)
            result = {'case_id': case['id'], 'model': model_id,
                'answer': None, 'comparison': 'not_completed', 'error': None,
                'elapsed_ms': 0, 'estimated_yuan': None, 'call_file': 'calls/'+identity+'.json'}
            try:
                response = observer.observe((run/'images'/(case['id']+'.png')).read_bytes())
                checked = compare(response, case['reference'])
                result.update(answer=checked['model_observation']['hip_change'],
                    comparison=checked['comparison'])
            except ModelCallError as error:
                result['error'] = error.code
                failed.add(model['provider'])
                if error.code == 'budget_not_admitted':
                    stop_all = True
            except ValueError:
                result['error'] = 'invalid_observation'
            if caller.last_record:
                result.update(elapsed_ms=caller.last_record['elapsed_ms'], estimated_yuan=caller.last_record['estimated_yuan'])
            records.append(result)
            # Crash-safe partial progress; no automatic resume or retry of paid calls.
            with (run/(stage+'_partial.json')).open('w', encoding='utf-8') as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
            print(json.dumps({key: result[key] for key in ('case_id','model','answer','error')}, ensure_ascii=False), flush=True)
    summaries = {key: score([r for r in records if r['model']==key], cases) for key in admitted}
    save(responses, records)
    value = {'stage': stage, 'summary': summaries, 'responses_sha256': sha(responses),
        'admitted_models': [key for key, result in summaries.items() if result['correct'] >= 2 and
            result['invalid_or_missing'] == 0 and result['unknown_cost'] == 0] if stage == 'pilot' else admitted}
    save(run/(stage+'.json'), value)
    return value


def event():
    value = read(PROJECT/'examples/set_completed.json')
    value['exercise_id'] = 'public_imu_research'
    for field in ('planned_reps','completed_reps','weight_kg_per_hand','metrics','set_index','rest_seconds'):
        value.pop(field, None)
    value['discomfort'] = {'status':'none', 'source':'self_report'}
    return value


def chain(run, data_dir, model_id, case_id):
    frozen, manifest = validated(run)
    if (run/'chain_trace.json').exists():
        raise ValueError('chain trace already exists; replay instead')
    model = next(m for m in frozen['models'] if m['id'] == model_id)
    case = next(c for c in manifest['cases'] if c['id'] == case_id)
    def factory():
        report, image, reference = bundle(data_dir, case)
        if report != case['report'] or hashlib.sha256(image).hexdigest() != case['image_sha256']:
            raise ValueError('rerender differs from frozen prediction')
        return report, image, reference
    factory()
    observer = Observer(Caller(model, run/'budget.json', run/'calls'), PROMPT, 'chain_visual')
    # Text decision uses the inexpensive Flash, independently recorded from vision choice.
    decision_model = next(m for m in frozen['models'] if m['id']=='qwen3.8-flash')
    decision = Choice(Caller(decision_model, run/'budget.json', run/'calls'), 'chain_decision')
    value = run_pipeline(event(), observer, decision, experiment_fn=factory)
    value['skeleton_origin'] = 'predicted_by_fixed_five_recording_ridge_alpha200'
    value['notice'] = 'Public IMU and head-height research chain; image geometry is checked against prediction, not motion truth. Simulated event; no App delivery.'
    value.update(schema='metafit-linked-live-run-1', case_id=case_id,
        visual_model=model_id, decision_model=decision_model['id'],
        event=event(), candidate=json.loads(decision.response.content) if decision.response else None,
        image_file='images/'+case_id+'.png',
        freeze_sha256=sha(run/'freeze.json'),
        provider_records={name: sha(run/'calls'/(name+'.json')) for name in ('chain_visual','chain_decision') if (run/'calls'/(name+'.json')).exists()})
    save(run/'chain_trace.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','pilot','confirmation','chain'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--models', nargs='+', default=list(registry()))
    parser.add_argument('--model')
    parser.add_argument('--case')
    parser.add_argument('--data-dir', type=Path, default=PROJECT/'data/fusionposer')
    parser.add_argument('--approve-expansion', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.manifest, args.run, args.models)
    elif args.command == 'confirmation' and not args.approve_expansion:
        parser.error('confirmation requires explicit expansion approval after pilot')
    elif args.command in ('pilot','confirmation'):
        result = comparison(args.run, args.command)
    else:
        result = chain(args.run, args.data_dir, args.model, args.case)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
