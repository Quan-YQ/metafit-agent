"""Evidence-bound research candidates cannot carry model-written claims."""
import json
from pathlib import Path
from copy import deepcopy
from metafit_agent.decision import ModelCallError, UNKNOWN_REASON
from metafit_agent.model_response import ModelResponse
from metafit_agent.research_evidence_guard import evaluate

EVENT=json.loads((Path(__file__).resolve().parents[1]/'examples'/'set_completed.json').read_text(encoding='utf-8'))

class Model:
    def __init__(self,action='notify',evidence='completed_reps',bad=False):
        self.action,self.evidence,self.bad,self.calls=action,evidence,bad,0
    def complete(self,facts):
        self.calls+=1
        if self.bad: raise ModelCallError('synthetic_error')
        ids=[f.fact_id for f in facts if f.name==self.evidence]
        value={'suggested_action':self.action,'message':None if self.action=='silent' else '你动作非常完美，加大负重。',
               'reason':'模拟自报无不适，状态未确认。','evidence':ids or ['fact_999']}
        return ModelResponse(json.dumps(value,ensure_ascii=False),'fake',None,None,None,0,'stop')

def base():
    event=deepcopy(EVENT); event['discomfort']={'status':'none','source':'self_report'}
    return event

def test_untrusted_text_is_rebuilt_from_cited_fact():
    r=evaluate(base(),Model())
    assert r['candidate_text']=='模拟记录显示本组完成了10次。'
    assert r['effective_action']=='notify' and not r['delivery'] and not r['may_display']
    assert '负重' not in r['candidate_text']

def test_quiet_and_defer_avoid_immediate_notification():
    r=evaluate(base(),Model(),do_not_disturb=True)
    assert r['effective_action']=='defer' and not r['delivery']
    r=evaluate(base(),Model(action='silent'))
    assert r['effective_action']=='silent' and r['candidate_text'] is None

def test_missing_or_wrong_evidence_stops():
    assert evaluate(base(),Model(evidence='exercise_id'))['reason_code']=='no_cited_completed_reps'
    assert evaluate(base(),Model(evidence='not_available'))['reason_code']=='evidence_unknown_id'

def test_discomfort_bypasses_model_and_errors_are_not_delivered():
    e=base();e['discomfort']={'status':'reported','source':'self_report'}
    model=Model();r=evaluate(e,model)
    assert r['review_status']=='safety_pending' and model.calls==0 and not r['delivery']
    assert '停止当前动作' in r['safety_message']
    r=evaluate(base(),Model(bad=True))
    assert r['review_status']=='stopped' and r['reason_code']=='synthetic_error' and r['candidate_text'] is None

def test_known_reason_can_be_corrected_only_for_simulated_none_status():
    class Ambiguous(Model):
        def complete(self,facts):
            self.calls+=1
            f=next(f for f in facts if f.name=='completed_reps')
            raw={'suggested_action':'notify','message':'未经验证的文字','reason':UNKNOWN_REASON,'evidence':[f.fact_id]}
            return ModelResponse(json.dumps(raw,ensure_ascii=False),'fake',None,None,None,0,'stop')
    r=evaluate(base(),Ambiguous(),do_not_disturb=True)
    assert r['reason_normalized'] is True and r['effective_action']=='defer'
    assert r['candidate_text']=='模拟记录显示本组完成了10次。' and not r['delivery']
    unknown=base();unknown['discomfort']={'status':'unknown','source':'self_report'}
    r=evaluate(unknown,Ambiguous(),do_not_disturb=True)
    assert r['reason_normalized'] is False and r['review_status']=='stopped'

def test_invalid_quiet_state_and_missing_reps_fail_closed():
    for state in ('false', 0, None):
        r=evaluate(base(),Model(),do_not_disturb=state)
        assert r['review_status']=='stopped' and r['candidate_text'] is None
        assert not r['delivery'] and not r['may_display']
    event=base();event.pop('completed_reps',None)
    r=evaluate(event,Model())
    assert r['review_status']=='stopped' and r['candidate_text'] is None

def test_invalid_model_structure_fails_closed():
    class Broken(Model):
        def complete(self,facts):
            return ModelResponse('{"suggested_action":"notify","suggested_action":"silent"}', 'fake',None,None,None,0,'stop')
    r=evaluate(base(),Broken())
    assert r['review_status']=='stopped' and r['candidate_text'] is None
    assert r['reason_code']=='invalid_model_json' and not r['delivery']
