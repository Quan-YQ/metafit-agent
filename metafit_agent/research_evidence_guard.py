"""Research-only deterministic evidence-bound candidate text; never delivers."""
from dataclasses import replace
import json
from .decision import UNKNOWN_REASON, _parse_candidate, decide_simulated_event
from .decision_review import review_outcome

class _NormalizeKnownReason:
    """Correct one known wording mismatch; leave action, evidence and message untouched."""
    def __init__(self, inner):
        self.inner=inner
        self.changed=False

    def complete(self, facts):
        response=self.inner.complete(facts)
        candidate,error=_parse_candidate(response.content)
        status=[f.value for f in facts if f.origin=='event' and f.name=='discomfort_status']
        if error is None and candidate is not None and status==['none'] and candidate.reason==UNKNOWN_REASON:
            value={'suggested_action':candidate.suggested_action,'message':candidate.message,
                   'reason':'模拟数据记录为自报无不适，报告时效及当前状态未确认。',
                   'evidence':list(candidate.evidence)}
            self.changed=True
            return replace(response,content=json.dumps(value,ensure_ascii=False))
        return response

def evaluate(event, model, *, do_not_disturb=False):
    """Run existing candidate/check flow, then rebuild visible candidate from cited facts.

    This isolation does not modify R5. It does not check physical truth or clinical safety.
    """
    wrapped=_NormalizeKnownReason(model)
    outcome=decide_simulated_event(event,wrapped)
    review=review_outcome(outcome,do_not_disturb=do_not_disturb)
    result={'simulation':True,'may_display':False,'delivery':False,
            'decision_status':outcome.status,'review_status':review.status,
            'effective_action':review.effective_action,'reason_code':review.code,
            'candidate_text':None,'evidence':[],
            'reason_normalized':wrapped.changed,
            'safety_message':outcome.safety_message if review.status=='safety_pending' else None,
            'notice':'研究模拟，不代表事实真实或可用于训练指导；不投递。'}
    if review.status != 'reviewed' or review.effective_action=='silent':
        return result
    cited=set(outcome.candidate.evidence)
    reps=[f for f in outcome.facts if f.fact_id in cited and f.name=='completed_reps'
          and f.origin=='event' and type(f.value) is int and 0<=f.value<=10000]
    if len(reps)!=1:
        result.update(review_status='stopped',effective_action=None,reason_code='no_cited_completed_reps')
        return result
    fact=reps[0]
    result['candidate_text']=f'模拟记录显示本组完成了{fact.value}次。'
    result['evidence']=[fact.fact_id]
    return result
