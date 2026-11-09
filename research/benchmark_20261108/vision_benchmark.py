"""Budgeted, resumable image API comparison; no secrets in artifacts or errors."""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vision_dataset import CATEGORIES, PROJECT

HERE = Path(__file__).resolve().parent
DEFAULT_LEDGER = PROJECT/'outputs/benchmark_20261108/vision_budget.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def read_key(name):
    value = os.environ.get(name, '').strip()
    if not value and os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as handle:
                value = winreg.QueryValueEx(handle, name)[0].strip()
        except (OSError, AttributeError):
            pass
    return value

def save_exclusive(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)

@contextmanager
def ledger_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock')
    descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()

def ledger_write(path, ledger):
    temporary = path.with_suffix('.pending')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(ledger, handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)

def reserve(path, amount, request_id):
    if not math.isfinite(amount) or amount < 0:
        raise ValueError('invalid reservation')
    with ledger_lock(path):
        ledger = json.loads(path.read_text()) if path.exists() else {'cap_yuan':10.0, 'entries':[]}
        if ledger.get('halted'):
            raise ValueError('budget ledger halted; review billing before any further request')
        if ledger['cap_yuan'] != 10.0:
            raise ValueError('budget ledger cap changed; explicit user decision required')
        consumed = sum(entry['charged_or_reserved_yuan'] for entry in ledger['entries'])
        if consumed+amount > ledger['cap_yuan']+1e-10:
            raise ValueError('budget exhausted before request')
        if any(entry['id']==request_id for entry in ledger['entries']):
            raise ValueError('request already reserved; do not retry unknown billing')
        ledger['entries'].append({'id':request_id, 'reserved_yuan':amount,
                  'charged_or_reserved_yuan':amount, 'status':'started', 'started_at':now()})
        ledger_write(path, ledger)

def settle(path, request_id, actual):
    if not math.isfinite(actual) or actual < 0:
        raise ValueError('invalid estimated cost')
    with ledger_lock(path):
        ledger = json.loads(path.read_text())
        entry = next(row for row in ledger['entries'] if row['id']==request_id)
        if entry['status'] != 'started':
            raise ValueError('request already settled')
        if actual > entry['reserved_yuan']+1e-8:
            entry.update(charged_or_reserved_yuan=actual, status='usage_exceeds_reservation', finished_at=now())
            ledger['halted'] = True
            ledger_write(path, ledger)
            raise ValueError('usage exceeds reserved bound; stop for price review')
        entry.update(charged_or_reserved_yuan=actual, status='usage_confirmed', finished_at=now())
        ledger_write(path, ledger)

def cost(model, usage):
    incoming, outgoing = usage['prompt_tokens'], usage['completion_tokens']
    if not all(isinstance(value, int) and not isinstance(value, bool) and value>=0 for value in (incoming,outgoing)):
        raise ValueError('invalid usage counts')
    input_price, output_price = model['input_per_million'], model['output_per_million']
    if model.get('tiers'):
        tier = next((tier for tier in model['tiers'] if incoming<=tier[0]), None)
        if tier is None:
            raise ValueError('usage beyond documented tier')
        _, input_price, output_price = tier
    return (incoming*input_price+outgoing*output_price)/1_000_000

def payload(model, image, prompt):
    if len(image)>2_000_000:
        raise ValueError('oversized image')
    return {'model':model['id'], 'messages':[{'role':'user','content':[
        {'type':'text','text':prompt},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}}]}],
        'temperature':0, 'max_tokens':4096, **model['request_extra']}

def send(model, body, key):
    # Endpoints are validated by main before any key is attached. No retries,
    # redirects, URL query credentials or request/error body logging.
    class RejectRedirects(__import__('urllib.request', fromlist=['HTTPRedirectHandler']).HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    import urllib.request
    opener = urllib.request.build_opener(RejectRedirects)
    request = Request(model['endpoint'], data=json.dumps(body,ensure_ascii=False).encode(),
                      headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'},method='POST')
    started=time.perf_counter()
    try:
        with opener.open(request,timeout=45) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000:
            return None,(time.perf_counter()-started)*1000,'response_too_large'
        return json.loads(raw),(time.perf_counter()-started)*1000,None
    except HTTPError as error:
        return None,(time.perf_counter()-started)*1000,'http_'+str(error.code)
    except (URLError,TimeoutError,OSError,ValueError):
        return None,(time.perf_counter()-started)*1000,'transport_or_json_error'

def observation(response, keys=()):
    try:
        choice=response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            return 'invalid','truncated_or_nonstop',''
        text=choice['message']['content'].strip()
        if text.startswith('```') and text.endswith('```'):
            text=re.sub(r'^```(?:json)?\s*','',text)[:-3].strip()
        value=json.loads(text)
        label=value['hip_change']
        if label not in (*CATEGORIES,'uncertain'):
            raise ValueError('invalid label')
        explanation=value.get('explanation','')
        if not isinstance(explanation,str):
            raise ValueError('invalid explanation')
        for key in keys:
            if key:
                explanation=explanation.replace(key,'[REDACTED]')
        explanation=re.sub(r'(?i)(?:bearer\s+|sk-)[A-Za-z0-9._-]{8,}','[REDACTED]',explanation)
        return label,None,explanation[:300]
    except (KeyError,IndexError,TypeError,AttributeError,ValueError):
        return 'invalid','invalid_observation',''

def safe_model_name(value, keys=()):
    if not isinstance(value,str) or not re.fullmatch(r'(?:glm-|GLM-|qwen)[a-zA-Z0-9._/-]{1,100}',value):
        return None
    if any(key and key in value for key in keys):
        return None
    return value

def summarize(rows, cases, model_ids):
    expected={case['id']:case for case in cases}
    summary={}
    for model in model_ids:
        group=[row for row in rows if row['model']==model]
        if len({row['case_id'] for row in group})!=len(group):
            raise ValueError('duplicate model-case result')
        correct=sum(row['observation']==expected[row['case_id']]['reference']['direction'] for row in group)
        uncertain=sum(row['observation']=='uncertain' for row in group)
        invalid=sum(row['observation']=='invalid' for row in group)
        per_class={category:{'total':sum(case['reference']['direction']==category for case in cases),
                   'correct':sum(row['observation']==category and expected[row['case_id']]['reference']['direction']==category for row in group)}
                   for category in CATEGORIES}
        elapsed=[row['elapsed_ms'] for row in group]
        summary[model]={'planned':len(cases),'completed':len(group),'correct':correct,
            'accuracy':correct/len(cases),'uncertain_rate':uncertain/len(cases),
            'invalid_or_missing_rate':(invalid+len(cases)-len(group))/len(cases),
            'per_class':per_class, 'latency_median_ms':statistics.median(elapsed) if elapsed else None,
            'latency_p95_ms':sorted(elapsed)[math.ceil(.95*len(elapsed))-1] if elapsed else None,
            'estimated_yuan':sum(row.get('estimated_yuan') or 0 for row in group),
            'cost_unknown_count':sum(row.get('estimated_yuan') is None for row in group),
            'confusion':{category:{answer:sum(expected[row['case_id']]['reference']['direction']==category and row['observation']==answer for row in group)
                         for answer in (*CATEGORIES,'uncertain','invalid')} for category in CATEGORIES}}
    return summary

def registry():
    data=json.loads((HERE/'vision_models.json').read_text(encoding='utf-8'))
    endpoints={'bigmodel':'https://open.bigmodel.cn/api/paas/v4/chat/completions',
               'dashscope':'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'}
    for model in data['models']:
        if model['endpoint'] != endpoints.get(model['provider']):
            raise ValueError('unapproved API endpoint')
    return data

def run(args):
    definitions=registry()
    models=definitions['models']
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if hashlib.sha256(manifest['prompt'].encode('utf-8')).hexdigest()!=manifest['prompt_sha256']:
        raise ValueError('prompt digest differs')
    if args.output is not None and (args.output/'results.json').exists():
        raise ValueError('summary exists; use archived result or a new run')
    if manifest['stage']=='confirmation':
        if args.selection_freeze is None or digest(args.selection_freeze)!=manifest['selection_freeze_sha256']:
            raise ValueError('confirmation does not match frozen selection')
        frozen=json.loads(args.selection_freeze.read_text(encoding='utf-8'))
        models=[model for model in models if model['id'] in frozen['candidates']]
        if frozen['registry_sha256'] != digest(HERE/'vision_models.json') or frozen['prompt_sha256'] != manifest['prompt_sha256']:
            raise ValueError('model configuration or prompt changed after selection')
    elif manifest['stage']!='selection':
        raise ValueError('unknown dataset stage')
    print(json.dumps({'keys':{name:bool(read_key(name)) for name in sorted({model['key_env'] for model in models})},
                      'models':[model['id'] for model in models],'live':args.live}),flush=True)
    if not args.live:
        return
    for model in models:
        if not read_key(model['key_env']):
            raise ValueError('required key missing: '+model['key_env'])
    if args.output is None:
        raise ValueError('output required for live run')
    args.output.mkdir(parents=True,exist_ok=True)
    rows_file=args.output/'responses.jsonl'
    rows=[json.loads(line) for line in rows_file.read_text(encoding='utf-8').splitlines()] if rows_file.exists() else []
    config={'manifest_sha256':digest(args.manifest),'registry_sha256':digest(HERE/'vision_models.json'),
            'prompt_sha256':manifest['prompt_sha256'],'stage':manifest['stage'],
            'code_sha256':digest(Path(__file__)),'started_at':now(),'budget_cap_yuan':10}
    if args.resume_from:
        previous=json.loads((args.resume_from/'run.json').read_text())
        previous_result=json.loads((args.resume_from/'results.json').read_text())
        previous_rows=args.resume_from/'responses.jsonl'
        if any(previous[key]!=config[key] for key in ('manifest_sha256','registry_sha256','prompt_sha256','code_sha256')):
            raise ValueError('cannot inherit responses for different images, prompt or models')
        if digest(previous_rows)!=previous_result['responses_sha256'] or digest(args.manifest)!=previous_result['manifest_sha256']:
            raise ValueError('inherited evidence hash differs')
        if rows_file.exists():
            raise ValueError('inherit only into a new run directory')
        inherited=[json.loads(line) for line in previous_rows.read_text(encoding='utf-8').splitlines()]
        excluded=[];rows=[]
        for row in inherited:
            if row.get('error')=='http_429' and args.retry_rate_limits:
                definition=next(model for model in models if model['id']==row['model'])
                if definition['reserve_yuan']!=0:
                    raise ValueError('do not retry paid requests with unknown billing')
                excluded.append(row)
            else:
                rows.append(row)
        save_exclusive(args.output/'inherited_attempts.json',{'parent_results_sha256':digest(args.resume_from/'results.json'),
                       'reused_responses':len(rows),'explicit_free_429_retries':excluded})
        with rows_file.open('x',encoding='utf-8') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        config['inherited_results_sha256']=digest(args.resume_from/'results.json')
    config_file=args.output/'run.json'
    if config_file.exists():
        old=json.loads(config_file.read_text())
        if any(old[key]!=config[key] for key in ('manifest_sha256','registry_sha256','code_sha256')):
            raise ValueError('resume configuration differs')
    else:
        save_exclusive(config_file,config)
    done={(row['case_id'],row['model']) for row in rows}
    failed_providers=set()
    all_keys=[read_key(model['key_env']) for model in models]
    last_free_start=None
    with rows_file.open('a',encoding='utf-8') as handle:
        for index,case in enumerate(manifest['cases']):
            ordered=models[index%len(models):]+models[:index%len(models)]
            for model in ordered:
                if (case['id'],model['id']) in done or model['provider'] in failed_providers:
                    continue
                image=(args.manifest.parent/(case['id']+'.png')).read_bytes()
                if hashlib.sha256(image).hexdigest()!=case['image_sha256']:
                    raise ValueError('image digest differs')
                request_id=hashlib.sha256((str(args.output.resolve())+case['id']+model['id']).encode()).hexdigest()
                total_started=time.monotonic()
                attempt_limit=getattr(args,'free_attempts',1) if model['reserve_yuan']==0 else 1
                base_request_id=request_id
                for attempt in range(attempt_limit):
                    request_id=base_request_id+(f'-retry{attempt}' if attempt else '')
                    reserve(args.ledger,model['reserve_yuan'],request_id)
                    # Schedule free requests independently of measured HTTP latency.
                    # Only an explicit --free-attempts allows bounded retries,
                    # and only HTTP429 on a documented zero-price model qualifies.
                    if model['provider']=='bigmodel':
                        interval=max(args.free_interval,55 if attempt else 0)
                        if last_free_start is not None:
                            time.sleep(max(0,interval-(time.monotonic()-last_free_start)))
                        last_free_start=time.monotonic()
                    response,elapsed,error=send(model,payload(model,image,manifest['prompt']),read_key(model['key_env']))
                    if error!='http_429' or attempt+1==attempt_limit:
                        break
                    failed={'case_id':case['id'],'model':model['id'],'request_id':request_id,
                            'error':error,'elapsed_ms':elapsed,'finished_at':now(),'reserved_yuan':0}
                    with (args.output/'quota_attempts.jsonl').open('a',encoding='utf-8') as attempts:
                        attempts.write(json.dumps(failed)+'\n');attempts.flush();os.fsync(attempts.fileno())
                    print(json.dumps({'model':model['id'],'case_id':case['id'],'free_quota_retry':attempt+1}),flush=True)
                row={'case_id':case['id'],'model':model['id'],'image_sha256':case['image_sha256'],
                     'elapsed_ms':round(elapsed,3),'observation':'invalid','estimated_yuan':None,
                     'request_id':request_id,'finished_at':now()}
                row.update(attempts=attempt+1,elapsed_with_scheduling_ms=round((time.monotonic()-total_started)*1000,3))
                if error:
                    row['error']=error
                    failed_providers.add(model['provider'])
                else:
                    usage=response.get('usage',{}) if isinstance(response,dict) else {}
                    try:
                        estimated=cost(model,usage)
                        settle(args.ledger,request_id,estimated)
                        row.update(usage={key:usage[key] for key in ('prompt_tokens','completion_tokens')},estimated_yuan=estimated)
                    except (ValueError,KeyError,TypeError):
                        row['error']='usage_or_budget_error'
                        failed_providers.add(model['provider'])
                    label,parse_error,explanation=observation(response,all_keys)
                    row.update(observation=label,explanation=explanation)
                    if parse_error:row['parse_error']=parse_error
                    returned=response.get('model','') if isinstance(response,dict) else ''
                    returned=safe_model_name(returned,all_keys)
                    if returned is not None:
                        row['returned_model']=returned
                handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
                rows.append(row)
                print(json.dumps({key:row.get(key) for key in ('model','case_id','observation','elapsed_ms','error','estimated_yuan')}),flush=True)
    summary=summarize(rows,manifest['cases'],[model['id'] for model in models])
    result={'schema':'vision-comparison-v1','stage':manifest['stage'],'summary':summary,
            'manifest_sha256':digest(args.manifest),'responses_sha256':digest(rows_file),
            'registry_sha256':digest(HERE/'vision_models.json'),'finished_at':now(),
            'notice':'18 balanced controlled images per model; source-clustered windows; API latency includes network and reasoning; gross list-price estimate, not invoice'}
    result_file=args.output/'results.json'
    if result_file.exists():
        raise ValueError('summary exists; use archived result or a new run')
    save_exclusive(result_file,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

def freeze(args):
    result=json.loads(args.results.read_text(encoding='utf-8'))
    if result['stage']!='selection' or result['registry_sha256']!=digest(HERE/'vision_models.json'):
        raise ValueError('invalid selection result')
    summary=result['summary']
    if len(summary)<3 or any(row['completed']!=row['planned'] or row['cost_unknown_count'] for row in summary.values()):
        raise ValueError('incomplete model comparison; do not select on partial calls')
    ranked=sorted(summary,key=lambda model:(-summary[model]['accuracy'],summary[model]['uncertain_rate'],
                  summary[model]['invalid_or_missing_rate'],summary[model]['estimated_yuan'],
                  summary[model]['latency_median_ms'],model))
    decision={'schema':'vision-selection-v1','candidates':ranked[:2], 'ranking':ranked,
              'selection_results_sha256':digest(args.results),'registry_sha256':result['registry_sha256'],
              'prompt_sha256':json.loads((args.results.parent/'run.json').read_text())['prompt_sha256'],
              'rule':'accuracy descending, uncertainty ascending, invalid ascending, cost ascending, latency ascending; top2',
              'frozen_at':now()}
    save_exclusive(args.output,decision)
    print(json.dumps(decision,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    runner=sub.add_parser('run')
    runner.add_argument('--manifest',type=Path,required=True)
    runner.add_argument('--output',type=Path)
    runner.add_argument('--selection-freeze',type=Path)
    runner.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    runner.add_argument('--live',action='store_true')
    runner.add_argument('--resume-from',type=Path,help='inherit identical completed calls from an earlier run')
    runner.add_argument('--retry-rate-limits',action='store_true',help='explicitly retry only free HTTP429 responses; preserve attempts')
    runner.add_argument('--free-interval',type=float,default=20,help='minimum seconds between free-provider starts')
    runner.add_argument('--free-attempts',type=int,default=1,help='explicit bounded attempts for free HTTP429 only; paid requests always1')
    selector=sub.add_parser('freeze')
    selector.add_argument('--results',type=Path,required=True)
    selector.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    if arguments.mode=='run' and not 0<=arguments.free_interval<=55:
        parser.error('free interval must be in [0,55]')
    if arguments.mode=='run' and not 1<=arguments.free_attempts<=4:
        parser.error('free attempts must be in [1,4]')
    run(arguments) if arguments.mode=='run' else freeze(arguments)
