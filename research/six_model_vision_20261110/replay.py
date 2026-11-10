"""Audit compact six-model archive offline, without keys or API requests."""
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def audit():
    for name,expected in read('analysis_method_hashes.json').items():
        if sha((ROOT/'analysis_sources'/(name+'.txt')).read_bytes())!=expected:
            raise ValueError('analysis source differs: '+name)
    freeze=read('freeze.json')
    for name,expected in freeze['hashes'].items():
        path=ROOT/name
        if not path.exists():path=ROOT/'frozen_sources'/(name+'.txt')
        if sha(path.read_bytes())!=expected:raise ValueError('frozen file differs: '+name)
    manifest=read('manifest.json');cases={c['id']:c for c in manifest['cases']}
    config=read('models.json');models={m['provider']:m for m in config['models']}
    saved=read('paired_results.json');ledger=read('budget.json')
    entries={e['id']:e for e in ledger['entries']}
    if len(entries)!=len(ledger['entries']):raise ValueError('duplicate ledger request')
    rows={};original_hashes={}
    for provider,model in models.items():
        rows[provider]={}
        path=ROOT/'responses'/(provider+'.jsonl')
        for line in path.read_text(encoding='utf-8').splitlines():
            wrapper=json.loads(line);r=wrapper['record'];ident=r['case_id'];case=cases[ident]
            raw=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
            if wrapper['original_newline']=='crlf':raw=raw.replace(b'\n',b'\r\n')
            elif wrapper['original_newline']!='lf':raise ValueError('unknown original newline')
            if sha(raw)!=wrapper['original_sha256']:raise ValueError('record content differs')
            if ident in rows[provider] or r['provider']!=provider or r['model']!=model['model']:
                raise ValueError('duplicate or wrong model response')
            image=(ROOT/'images'/(ident+'.png')).read_bytes()
            if sha(image)!=case['image_sha256'] or r['image_sha256']!=case['image_sha256']:
                raise ValueError('image differs')
            if sha(case['prompt'].encode('utf-8'))!=r['prompt_sha256']:
                raise ValueError('prompt differs')
            msg=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}},
                                           {'type':'text','text':case['prompt']}]}]
            body={'model':model['model'],'messages':msg,'stream':False,'response_format':{'type':'json_object'},**model['extra']}
            if provider=='kimi':body['max_completion_tokens']=model['max_output']
            else:body.update(max_tokens=model['max_output'],temperature=0)
            if sha(canonical(body))!=r['request_sha256']:raise ValueError('request hash differs')
            if r['request_id_local'] not in entries:raise ValueError('request missing from budget')
            if not r['provider_error'] and r['answer']!='invalid':
                parsed=json.loads(r['raw_content'])
                if parsed.get('hip_change')!=r['answer']:raise ValueError('answer differs')
            rows[provider][ident]=r
            original_hashes['responses/'+provider+'/'+ident+'.json']=wrapper['original_sha256']
    for name,expected in read('inheritance.json')['inherited_response_sha256'].items():
        if original_hashes.get(name)!=expected:raise ValueError('pilot response differs')
    common=set.intersection(*(set(group) for group in rows.values()))
    if sha('\n'.join(sorted(common)).encode())!=saved['paired_coverage']['paired_ids_sha256']:
        raise ValueError('paired subset differs')
    groups={'cmu_all':lambda c:c['style']!='imu_prediction',
            'cmu_guided':lambda c:c['style']=='guided',
            'cmu_unguided':lambda c:c['style']=='unguided',
            'imu':lambda c:c['style']=='imu_prediction'}
    for provider in models:
        for name,predicate in groups.items():
            subset=[cases[c] for c in common if predicate(cases[c])]
            counts=Counter(c['label'] for c in subset)
            correct=sum(rows[provider][c['id']]['answer']==c['label'] for c in subset)
            per=[sum(rows[provider][c['id']]['answer']==label for c in subset if c['label']==label)/n
                 for label,n in counts.items()]
            expected=saved['models'][provider]['metrics'][name]
            if expected['cases']!=len(subset) or expected['correct']!=correct:
                raise ValueError('score differs')
            if per and not math.isclose(expected['balanced_accuracy'],statistics.mean(per),abs_tol=1e-12):
                raise ValueError('class average differs')
    known=sum(e['charged_or_reserved_yuan'] for e in ledger['entries'] if e['status']=='usage_confirmed')
    unknown=sum(e['charged_or_reserved_yuan'] for e in ledger['entries'] if e['status']!='usage_confirmed')
    if not math.isclose(known,saved['budget']['spent_estimate_yuan'],abs_tol=1e-10) or not math.isclose(unknown,saved['budget']['unconfirmed_reserved_yuan'],abs_tol=1e-10):
        raise ValueError('budget differs')
    return {'status':'passed','network_calls':0,'responses':sum(map(len,rows.values())),
            'paired_images':len(common),'models':len(models),'known_estimated_yuan':known,
            'unknown_reserved_yuan':unknown}

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
