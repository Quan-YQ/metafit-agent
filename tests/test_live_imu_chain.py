"""Exercise paid-call admission, blinding, failure recording and evidence binding offline."""
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip('numpy')
Image = pytest.importorskip('PIL.Image')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research/live_chain_20261110'))
import client as live
import dataset as imu_data
import run as chain_run
import fusionposer_imu_agent as pipeline
from metafit_agent.decision import ModelCallError
from metafit_agent.model_response import ModelResponse, TokenUsage
from trace_motion_agent import ScriptedObserver
from research_motion_choice import OfflineChoiceModel


def png():
    buffer = BytesIO()
    Image.new('RGB', (1200,460), 'white').save(buffer,format='PNG')
    return buffer.getvalue()


def body():
    return [{'role':'user','content':[{'type':'text','text':'只观察髋部变化。'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png()).decode()}}]}]


@pytest.fixture
def caller(tmp_path, monkeypatch):
    monkeypatch.setattr(live,'read_key',lambda name:'test-secret-value')
    return live.Caller(live.registry()['qwen3.8-flash'], tmp_path/'budget.json',tmp_path/'calls')


def response(content=None):
    return {'model':'qwen3.8-flash','id':'provider-request',
        'usage':{'prompt_tokens':500,'completion_tokens':40},
        'choices':[{'finish_reason':'stop','message':{'content':content or
            '{"hip_change":"up","explanation":"髋部升高。"}'}}]}


def test_real_payload_is_blind_and_usage_settles(caller, monkeypatch):
    sent=[]
    def send(model,payload,key):
        sent.append((model,payload,key))
        return response(), 12, None
    monkeypatch.setattr(live,'send',send)
    out=caller.call(body(),'observation')
    assert out.usage==TokenUsage(500,40)
    assert len(sent)==1 and sent[0][2]=='test-secret-value'
    text=json.dumps(sent[0][1])
    assert 'reference' not in text and 'prediction_report' not in text
    ledger=json.loads(caller.ledger.read_text())
    assert ledger['entries'][0]['status']=='usage_confirmed'
    assert 'test-secret-value' not in (caller.archive/'observation.json').read_text()


def test_provider_timeout_retains_reservation_and_does_not_retry(caller,monkeypatch):
    calls=[]
    monkeypatch.setattr(live,'send',lambda *args:(calls.append(1) or (None,42,'transport_or_json_error')))
    with pytest.raises(ModelCallError,match='transport_or_json_error'):
        caller.call(body(),'failure')
    assert len(calls)==1
    entry=json.loads(caller.ledger.read_text())['entries'][0]
    assert entry['charged_or_reserved_yuan']==entry['reserved_yuan']>0
    assert (caller.archive/'failure.json').exists()
    with pytest.raises(ModelCallError,match='archived_request_exists'):
        caller.call(body(),'failure')
    assert len(calls)==1


def test_unapproved_endpoint_stops_before_auth_or_send(caller,monkeypatch):
    caller.model={**caller.model,'endpoint':'https://wrong.example/'}
    monkeypatch.setattr(live,'send',lambda *args:pytest.fail('must not send'))
    with pytest.raises(ModelCallError,match='unapproved_endpoint'):
        caller.call(body(),'bad')
    assert not caller.ledger.exists()


def test_unknown_usage_is_archived_and_reservation_retained(caller,monkeypatch):
    bad=response(); del bad['usage']
    monkeypatch.setattr(live,'send',lambda *args:(bad,1,None))
    with pytest.raises(ModelCallError,match='usage_or_budget_uncertain'):
        caller.call(body(),'unknown')
    assert json.loads(caller.ledger.read_text())['entries'][0]['status']=='started'


def test_budget_exhausted_blocks_network(caller,monkeypatch):
    from vision_benchmark import reserve
    reserve(caller.ledger,10,'already-used')
    monkeypatch.setattr(live,'send',lambda *args:pytest.fail('must not send'))
    with pytest.raises(ModelCallError,match='budget_not_admitted'):
        caller.call(body(),'no-budget')
    assert caller.requests_started==0


def test_response_secrets_redacted(caller,monkeypatch):
    monkeypatch.setattr(live,'send',lambda *args:(response('test-secret-value'),1,None))
    result=caller.call(body(),'redact')
    assert result.content=='[REDACTED]'
    assert 'test-secret-value' not in (caller.archive/'redact.json').read_text()


@pytest.mark.parametrize('content',[[],[{'type':'text','text':'x'*17000}],
    [{'type':'image_url','image_url':{'url':'data:image/png;base64,bad'}}]])
def test_oversized_or_invalid_input_rejected_without_send(caller,monkeypatch,content):
    monkeypatch.setattr(live,'send',lambda *args:pytest.fail('must not send'))
    if not content:
        # Explicit unsupported item, not an otherwise valid empty context.
        content=[{'type':'video','url':'x'}]
    with pytest.raises(ValueError):
        caller.call([{'role':'user','content':content}],'invalid')
    assert not caller.ledger.exists()


def test_score_keeps_missing_uncertain_and_wrong_in_denominator():
    cases=[{'id':str(i),'reference':{'direction':label}} for i,label in enumerate(('up','down','no_clear_change'))]
    rows=[{'case_id':'0','comparison':'matches_reference','answer':'up','error':None,'estimated_yuan':.1,'elapsed_ms':10},
          {'case_id':'1','comparison':'inconclusive','answer':'uncertain','error':None,'estimated_yuan':.1,'elapsed_ms':20}]
    value=chain_run.score(rows,cases)
    assert value['accuracy']==pytest.approx(1/3) and value['balanced_accuracy']==pytest.approx(1/3)
    assert value['uncertain']==1 and value['invalid_or_missing']==1


def test_injected_predictor_metadata_is_bound_without_changing_app():
    event=chain_run.event()
    report={'x_sha256':'x','y_sha256':'y','selected_original_rows':list(range(6)),
        'predicted_hip_reference':{'direction':'up'},'predictor_sha256':'original'}
    reads=[]
    def changing():
        reads.append(1)
        result=dict(report)
        if len(reads)>1: result['predictor_sha256']='different'
        return result,b'\x89PNG\r\n\x1a\nPREDICTED',b''
    decision=OfflineChoiceModel()
    value=pipeline.run_pipeline(event,ScriptedObserver('up'),decision,experiment_fn=changing)
    assert value['r2_error']=='binding_precheck_failed' and decision.calls==0


def test_actual_adapter_counters_and_model_responses_flow_into_r3():
    report={'x_sha256':'x','y_sha256':'y','selected_original_rows':list(range(6)),
        'predicted_hip_reference':{'direction':'up'},'predictor_sha256':'frozen'}
    class O(ScriptedObserver):
        mode='live_official_vision'
        requests_started=1
    class D(OfflineChoiceModel):
        mode='live_official_choice'
        requests_started=1
    value=pipeline.run_pipeline(chain_run.event(),O('up'),D(),
        experiment_fn=lambda:(dict(report),b'\x89PNG\r\n\x1a\nPREDICTED',b''))
    assert value['r3_status']=='reviewed' and value['binding_status']=='matched_simulation'
    assert value['visual_requests_started']==value['decision_requests_started']==1
    assert value['may_display'] is False and value['delivery'] is False


def test_provider_failure_stops_sibling_models_too(tmp_path,monkeypatch):
    models=list(live.registry().values())[:2]
    cases=[{'id':'one','stage':'pilot','reference':{'direction':'up'}}]
    monkeypatch.setattr(chain_run,'validated',lambda path:({'models':models},{'cases':cases,'prompt':'observe'}))
    sent=[]
    class FakeCaller:
        last_record=None
        def __init__(self,*args): pass
    class FailedObserver:
        def __init__(self,*args): pass
        def observe(self,image):
            sent.append(1)
            raise ModelCallError('http_401')
    monkeypatch.setattr(chain_run,'Caller',FakeCaller)
    monkeypatch.setattr(chain_run,'Observer',FailedObserver)
    (tmp_path/'images').mkdir()
    (tmp_path/'images/one.png').write_bytes(png())
    result=chain_run.comparison(tmp_path,'pilot')
    assert len(sent)==1
    assert not result['admitted_models']


def test_shipped_real_archive_replays_without_network(monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('live_chain_archive_replay',ROOT/'research/live_chain_20261110/replay.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(live,'send',lambda *args:pytest.fail('offline replay must not send'))
    archive=ROOT/'research/live_chain_20261110/results/run_A'
    result=module.audit(archive)
    assert result['network_calls']==0
    assert result['chain']['binding_and_review_reproduced'] is True
    with pytest.raises(ValueError,match='no further API calls'):
        chain_run.validated(archive)


def test_archived_request_hash_proves_image_and_visual_fact_were_sent():
    import hashlib
    from check_vision_observation import PROMPT
    from research_motion_choice import RESEARCH_SYSTEM
    archive=ROOT/'research/live_chain_20261110/results/run_A'
    trace=json.loads((archive/'chain_trace.json').read_text(encoding='utf-8'))
    frozen=json.loads((archive/'freeze.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in frozen['models']}
    records={name:json.loads((archive/'calls'/(name+'.json')).read_text(encoding='utf-8'))
             for name in ('chain_visual','chain_decision')}
    def check_request(model_id,messages,record):
        model=models[model_id]
        payload={'model':model_id,'messages':messages,'temperature':0,'max_tokens':8192,
            'stream':False,'response_format':{'type':'json_object'},**model['request_extra']}
        actual=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert actual==record['request_sha256']
    class O:
        mode=trace['visual_observation']['observer_mode']
        requested_model=trace['visual_model']
        calls=requests_started=0
        usage=elapsed_ms=None
        def observe(self,image):
            check_request(self.requested_model,[{'role':'user','content':[
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}},
                {'type':'text','text':PROMPT}]}],records['chain_visual'])
            return records['chain_visual']['content']
    class D:
        mode='offline_archive_request_verification'
        calls=requests_started=0
        response=None
        def complete(self,facts):
            assert any(f.name=='simulated_model_reported_hip_change' for f in facts)
            safe=[{'fact_id':f.fact_id,'name':f.name,'value':f.value} for f in facts]
            check_request(trace['decision_model'],[
                {'role':'system','content':RESEARCH_SYSTEM},
                {'role':'user','content':json.dumps({'facts':safe},ensure_ascii=False)}],records['chain_decision'])
            r=records['chain_decision']; u=r['usage']
            self.response=ModelResponse(r['content'],r['requested_model'],r['returned_model'],
                r['provider_request_id'],TokenUsage(u['input_tokens'],u['output_tokens']),r['elapsed_ms'],'stop')
            return self.response
    result=pipeline.run_pipeline(trace['event'],O(),D(),experiment_fn=lambda:
        (trace['prediction_research'],(archive/trace['image_file']).read_bytes(),b''))
    assert result['r3_status']=='reviewed'
