"""Zero-API replay: verify archives, rescore all outcomes, optionally rerender IMU."""
import argparse
import hashlib
import json
from pathlib import Path

from dataset import bundle, digest, sha
from run import read, score, validated
from check_vision_observation import compare


def audit(run, data_dir=None):
    frozen, manifest = validated(run)
    checks = {'network_calls': 0, 'images': len(manifest['cases']), 'stages': [], 'chain': None}
    for stage in ('pilot', 'confirmation'):
        if not (run/(stage+'.json')).exists():
            continue
        summary = read(run/(stage+'.json'))
        rows = read(run/(stage+'_responses.json'))
        if sha(run/(stage+'_responses.json')) != summary['responses_sha256']:
            raise ValueError('response manifest changed')
        cases = [c for c in manifest['cases'] if c['stage']==stage]
        for row in rows:
            call = read(run/row['call_file'])
            if call['requested_model'] != row['model']:
                raise ValueError('wrong call model')
            if call['error'] is None:
                case = next(c for c in cases if c['id']==row['case_id'])
                try:
                    checked = compare(call['content'], case['reference'])
                except ValueError:
                    if row['error'] != 'invalid_observation':
                        raise ValueError('invalid output lost from scoring') from None
                else:
                    if checked['comparison'] != row['comparison'] or checked['model_observation']['hip_change'] != row['answer']:
                        raise ValueError('score differs from provider response')
        recalculated = {key: score([r for r in rows if r['model']==key], cases) for key in summary['summary']}
        if recalculated != summary['summary']:
            raise ValueError('summary changed')
        checks['stages'].append({'stage':stage, 'provider_records': len(rows), 'scores_reproduced': True})
    if data_dir:
        for case in manifest['cases']:
            report, image, _ = bundle(data_dir, case)
            if report != case['report'] or hashlib.sha256(image).hexdigest() != case['image_sha256']:
                raise ValueError('source regeneration differs')
        checks['rerendered'] = len(manifest['cases'])
    if (run/'chain_trace.json').exists():
        trace = read(run/'chain_trace.json')
        if trace['freeze_sha256'] != sha(run/'freeze.json'):
            raise ValueError('chain attached to wrong freeze')
        for identity, expected in trace['provider_records'].items():
            if sha(run/'calls'/(identity+'.json')) != expected:
                raise ValueError('chain provider record changed')
        visual = read(run/'calls/chain_visual.json')
        decision = read(run/'calls/chain_decision.json') if (run/'calls/chain_decision.json').exists() else None
        case = next(c for c in manifest['cases'] if c['id']==trace['case_id'])
        observation = trace['visual_observation']
        checked = compare(visual['content'], case['reference'])
        if observation['comparison'] != checked['comparison'] or observation['model_hip_change'] != checked['model_observation']['hip_change']:
            raise ValueError('chain observation changed')
        if trace['visual_requests_started'] != 1 or trace['may_display'] or trace['delivery']:
            raise ValueError('invalid demonstration mode')
        if decision:
            if trace['decision_requests_started'] != 1 or trace['candidate'] != json.loads(decision['content']):
                raise ValueError('chain candidate changed')
        if trace['r3_status'] == 'reviewed':
            # Re-run actual event handling, binding and R3 on recorded responses.
            # Replay explicitly counts no new provider requests.
            from metafit_agent.model_response import ModelResponse, TokenUsage
            from fusionposer_imu_agent import run_pipeline
            class RecordedObserver:
                mode = 'archived_observation_replay'
                requested_model = visual['requested_model']
                calls = requests_started = 0
                usage = visual['usage']
                elapsed_ms = visual['elapsed_ms']
                def observe(self, image):
                    if hashlib.sha256(image).hexdigest() != case['image_sha256']:
                        raise ValueError('replay image differs')
                    return visual['content']
            class RecordedChoice:
                mode = 'archived_choice_replay'
                calls = requests_started = 0
                response = None
                def complete(self, facts):
                    usage = decision['usage']
                    self.response = ModelResponse(decision['content'], decision['requested_model'],
                        decision['returned_model'], decision['provider_request_id'],
                        TokenUsage(usage['input_tokens'], usage['output_tokens']), decision['elapsed_ms'], 'stop')
                    return self.response
            factory = lambda: (case['report'], (run/trace['image_file']).read_bytes(), b'')
            replayed = run_pipeline(trace['event'], RecordedObserver(), RecordedChoice(), experiment_fn=factory)
            for field in ('binding_sha256','binding_status','reference_check','candidate_action','candidate_evidence','r2_status','r3_status','effective_action'):
                if trace[field] != replayed[field]:
                    raise ValueError('R3 replay differs: '+field)
        checks['chain'] = {'r3_status':trace['r3_status'], 'visual_requests': trace['visual_requests_started'],
                           'decision_requests': trace['decision_requests_started'], 'binding_and_review_reproduced': trace['r3_status']=='reviewed'}
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.data_dir), ensure_ascii=False, indent=2))
