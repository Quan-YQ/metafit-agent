"""Research contract tests: leakage, accounting failures and public-image splits."""
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import sys

import pytest

pytest.importorskip('numpy')
pytest.importorskip('PIL')
HERE = Path(__file__).resolve().parents[1]/'research/benchmark_20261108'
sys.path.insert(0,str(HERE))
import vision_benchmark as vision
import vision_dataset as images
import cip_shared as shared

def model(name='qwen3-vl-plus'):
    return next(row for row in vision.registry()['models'] if row['id']==name)

def test_budget_reserves_before_send_and_survives_restart(tmp_path):
    ledger=tmp_path/'budget.json'
    vision.reserve(ledger,9.2,'unknown-network-result')
    with pytest.raises(ValueError,match='exhausted'):
        vision.reserve(ledger,.9,'next')
    assert json.loads(ledger.read_text())['entries'][0]['status']=='started'
    with pytest.raises(ValueError,match='already reserved'):
        vision.reserve(ledger,0,'unknown-network-result')
    vision.settle(ledger,'unknown-network-result',.01)
    vision.reserve(ledger,.9,'next')
    assert sum(row['charged_or_reserved_yuan'] for row in json.loads(ledger.read_text())['entries'])==pytest.approx(.91)

def test_budget_rejects_modified_cap_and_nonfinite(tmp_path):
    ledger=tmp_path/'budget.json'
    ledger.write_text(json.dumps({'cap_yuan':100,'entries':[]}))
    with pytest.raises(ValueError,match='cap changed'):
        vision.reserve(ledger,.1,'x')
    for invalid in (float('nan'),float('inf'),-1):
        with pytest.raises(ValueError):vision.reserve(ledger,invalid,'x')

def test_unknown_billing_cannot_be_settled_above_reservation(tmp_path):
    ledger=tmp_path/'budget.json'
    vision.reserve(ledger,.9,'one')
    with pytest.raises(ValueError,match='reserved bound'):
        vision.settle(ledger,'one',1)
    record=json.loads(ledger.read_text())
    assert record['entries'][0]['charged_or_reserved_yuan']==1
    assert record['halted'] is True
    with pytest.raises(ValueError,match='halted'):
        vision.reserve(ledger,0,'another-run')

def test_fee_uses_official_tier_and_output_tokens():
    assert vision.cost(model(),{'prompt_tokens':2000,'completion_tokens':100})==pytest.approx(.003)
    assert vision.cost(model(),{'prompt_tokens':40000,'completion_tokens':100})==pytest.approx(.0615)
    with pytest.raises(ValueError):vision.cost(model(),{'prompt_tokens':True,'completion_tokens':0})

def response(label, explanation='visible change'):
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'hip_change':label,'explanation':explanation})}}]}

def test_uncertainty_is_different_from_unchanged_and_failure():
    for label in (*images.CATEGORIES,'uncertain'):
        assert vision.observation(response(label))[0]==label
    truncated=response('up');truncated['choices'][0]['finish_reason']='length'
    assert vision.observation(truncated)[:2]==('invalid','truncated_or_nonstop')
    assert vision.observation(response('maybe'))[0]=='invalid'

def test_secret_cannot_enter_stored_explanation_or_error():
    fake='test-secret-value-not-a-real-credential'
    assert fake not in vision.observation(response('up',fake),[fake])[2]
    assert vision.observation({'error':{'message':fake}},[fake])[2]==''
    assert vision.safe_model_name(fake,[fake]) is None
    assert vision.safe_model_name('qwen-model', ['qwen-model']) is None
    assert vision.safe_model_name('glm-4.6v-flash')=='glm-4.6v-flash'

def test_request_contains_image_and_same_question_but_no_reference():
    body=vision.payload(model(),b'public-png',images.PROMPT)
    message=body['messages'][0]['content']
    assert message[0]['text']==images.PROMPT
    assert message[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert 'reference' not in json.dumps(body)
    assert 'Authorization' not in json.dumps(body)

def test_failures_count_in_accuracy_denominator():
    cases=[{'id':str(i),'reference':{'direction':label}} for i,label in enumerate(images.CATEGORIES)]
    rows=[{'case_id':'0','model':'m','observation':'up','elapsed_ms':1,'estimated_yuan':.01},
          {'case_id':'1','model':'m','observation':'uncertain','elapsed_ms':2,'estimated_yuan':0},
          {'case_id':'2','model':'m','observation':'invalid','elapsed_ms':3,'estimated_yuan':None}]
    result=vision.summarize(rows,cases,['m'])['m']
    assert result['accuracy']==pytest.approx(1/3)
    assert result['uncertain_rate']==pytest.approx(1/3)
    assert result['invalid_or_missing_rate']==pytest.approx(1/3)
    assert result['cost_unknown_count']==1

def test_confirmation_requires_freeze_and_disjoint_people(tmp_path):
    assert not {s.split('_')[0] for s in images.DEVELOPMENT} & {s.split('_')[0] for s in images.CONFIRMATION}
    with pytest.raises(ValueError,match='freeze'):
        images.prepare('confirmation',tmp_path/'new',tmp_path/'cache')
    assert not (tmp_path/'new').exists()

def test_training_overlap_guard_is_subject_based():
    shared.assert_untrained(('04','08'),'subject00_task_rep1_trials.pt\nsubject03_task_rep1_trials.pt')
    with pytest.raises(ValueError,match='training'):
        shared.assert_untrained(('04',),'subject04_different_action.pt')

@pytest.mark.parametrize('transient',[False,True])
def test_free_quota_recovery_reuses_paid_calls_and_preserves_failed_attempt(tmp_path,monkeypatch,transient):
    from types import SimpleNamespace
    definitions=vision.registry()
    definitions['models']=[row for row in definitions['models'] if row['id'] in ('glm-4.6v-flash','qwen3-vl-plus')]
    monkeypatch.setattr(vision,'registry',lambda:definitions)
    monkeypatch.setattr(vision,'read_key',lambda _: 'test-placeholder')
    image=b'public-image';image_hash=hashlib.sha256(image).hexdigest()
    (tmp_path/'one.png').write_bytes(image)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'stage':'selection','prompt':images.PROMPT,
            'prompt_sha256':hashlib.sha256(images.PROMPT.encode()).hexdigest(),
            'cases':[{'id':'one','reference':{'direction':'up'},'image_sha256':image_hash}]}),encoding='utf-8')
    prior=tmp_path/'prior';prior.mkdir()
    records=[{'case_id':'one','model':'glm-4.6v-flash','observation':'invalid','error':'http_429',
              'image_sha256':image_hash,'elapsed_ms':1,'estimated_yuan':None},
             {'case_id':'one','model':'qwen3-vl-plus','observation':'up','image_sha256':image_hash,
              'elapsed_ms':2,'estimated_yuan':.001}]
    (prior/'responses.jsonl').write_text('\n'.join(json.dumps(row) for row in records)+'\n',encoding='utf-8')
    (prior/'run.json').write_text(json.dumps({'manifest_sha256':vision.digest(manifest),
              'registry_sha256':vision.digest(HERE/'vision_models.json'),
              'prompt_sha256':hashlib.sha256(images.PROMPT.encode()).hexdigest(),
              'code_sha256':vision.digest(Path(vision.__file__))}))
    (prior/'results.json').write_text(json.dumps({'manifest_sha256':vision.digest(manifest),
              'responses_sha256':vision.digest(prior/'responses.jsonl')}))
    calls=[]
    def send(model,body,key):
        calls.append(model['id'])
        if transient and len(calls)==1:return None,1,'http_429'
        return response('up')|{'usage':{'prompt_tokens':10,'completion_tokens':20}},3,None
    monkeypatch.setattr(vision,'send',send)
    monkeypatch.setattr(vision.time,'sleep',lambda _:None)
    args=SimpleNamespace(manifest=manifest,selection_freeze=None,live=True,output=tmp_path/'recovery',
                         ledger=tmp_path/'budget.json',resume_from=prior,retry_rate_limits=True,free_interval=0,free_attempts=2)
    vision.run(args)
    assert calls==['glm-4.6v-flash']*(2 if transient else 1)
    result=json.loads((args.output/'results.json').read_text())
    assert all(row['correct']==1 for row in result['summary'].values())
    attempts=json.loads((args.output/'inherited_attempts.json').read_text())
    assert len(attempts['explicit_free_429_retries'])==1
    assert attempts['reused_responses']==1
    if transient:
        retry=json.loads((args.output/'quota_attempts.jsonl').read_text())
        assert retry['error']=='http_429' and retry['reserved_yuan']==0

def test_calibration_cannot_consume_scored_reference_frames(monkeypatch):
    torch=pytest.importorskip('torch')
    official=HERE.parents[1]/'.tmp/dynaip_official'
    if not (official/'provenance.json').exists():
        pytest.skip('optional official source download absent')
    import numpy as np
    modules=shared.official_modules(official)
    n=shared.WARMUP+121
    q=np.zeros((n,23,4),dtype=np.float32);q[...,0]=1
    data={'Segment Orientation - Quat':q.copy(),'Sensor Orientation - Quat':q.copy(),
          'Segment Position':np.zeros((n,23,3),dtype=np.float32),
          'Sensor Free Acceleration':np.zeros((n,23,3),dtype=np.float32)}
    monkeypatch.setattr(shared,'read_workbook',lambda _:copy.deepcopy(data))
    before=shared.prepare(b'',modules)
    data['Segment Orientation - Quat'][shared.WARMUP:,shared.JOINT_MASK]=np.array([0,1,0,0])
    after=shared.prepare(b'',modules)
    assert torch.equal(before['imu'],after['imu'])
    assert torch.equal(before['features'],after['features'])
    assert not torch.equal(before['rotation'],after['rotation'])
    assert (shared.G @ np.array([1,2,3])).tolist()==[2,3,1]


def test_prompt_and_existing_summary_stop_before_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(vision, 'read_key', lambda _: 'placeholder')
    monkeypatch.setattr(vision, 'send', lambda *a: pytest.fail('network called'))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'stage':'selection','prompt':'changed','prompt_sha256':'0'*64,'cases':[]}))
    output=tmp_path/'out'
    args=SimpleNamespace(manifest=manifest, selection_freeze=None, live=True, output=output, ledger=tmp_path/'ledger', resume_from=None)
    with pytest.raises(ValueError,match='prompt digest'):
        vision.run(args)
    assert not output.exists() and not args.ledger.exists()
    manifest.write_text(json.dumps({'stage':'selection','prompt':'ok','prompt_sha256':hashlib.sha256(b'ok').hexdigest(),'cases':[]}))
    output.mkdir()
    (output/'results.json').write_text('{}')
    with pytest.raises(ValueError,match='summary exists'):
        vision.run(args)
    assert not args.ledger.exists() and list(output.iterdir())==[output/'results.json']

def test_official_provenance_cannot_be_emptied(tmp_path):
    import cip_shared as shared
    provenance=json.loads((HERE/'source_audit/dynaip_provenance.json').read_text(encoding='utf-8'))
    provenance['files']={}
    (tmp_path/'provenance.json').write_text(json.dumps(provenance),encoding='utf-8')
    with pytest.raises(ValueError,match='provenance differs'):
        shared.official_modules(tmp_path)


def test_resume_rejects_different_code_before_request(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(vision, 'read_key', lambda _: 'placeholder')
    monkeypatch.setattr(vision, 'send', lambda *a: pytest.fail('network called'))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'stage':'selection','prompt':'ok','prompt_sha256':hashlib.sha256(b'ok').hexdigest(),'cases':[]}))
    prior=tmp_path/'prior'; prior.mkdir()
    (prior/'run.json').write_text(json.dumps({'manifest_sha256':vision.digest(manifest),'registry_sha256':vision.digest(HERE/'vision_models.json'),'prompt_sha256':hashlib.sha256(b'ok').hexdigest(),'code_sha256':'old'}))
    (prior/'results.json').write_text('{}')
    args=SimpleNamespace(manifest=manifest,selection_freeze=None,live=True,output=tmp_path/'new',ledger=tmp_path/'ledger',resume_from=prior,retry_rate_limits=False)
    with pytest.raises(ValueError,match='cannot inherit'):
        vision.run(args)
    assert not args.ledger.exists()

def test_cip_rejects_existing_provenance_before_download(tmp_path, monkeypatch):
    import fetch_cip08
    output=tmp_path/'archive.zip'
    output.with_suffix('.provenance.json').write_text('keep')
    monkeypatch.setattr(fetch_cip08,'urlopen',lambda *a,**k: pytest.fail('network called'))
    with pytest.raises(ValueError,match='provenance'):
        fetch_cip08.fetch(output)
    assert output.with_suffix('.provenance.json').read_text()=='keep'
