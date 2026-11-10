"""Official multimodal endpoints, sanitized archives, shared persistent CNY ledger."""
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'research/benchmark_20261108'))
sys.path.insert(0, str(PROJECT))
from vision_benchmark import read_key, reserve, settle, send, now
from metafit_agent.decision import ModelCallError
from metafit_agent.model_response import ModelResponse, TokenUsage

ENDPOINTS = {
    'dashscope': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    'ark': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
}
MAX_OUTPUT = 8192
INPUT_RESERVATION = 65536


def registry():
    value = json.loads(Path(__file__).with_name('models.json').read_text(encoding='utf-8'))
    for model in value['models']:
        if model['endpoint'] != ENDPOINTS.get(model['provider']):
            raise ValueError('unapproved provider endpoint')
    return {model['id']: model for model in value['models']}


def sanitize(text):
    if not isinstance(text, str):
        return None
    for name in ('DASHSCOPE_API_KEY', 'ARK_API_KEY', 'ZHIPUAI_API_KEY'):
        key = read_key(name)
        if key:
            text = text.replace(key, '[REDACTED]')
    return re.sub(r'(?i)(?:bearer\s+|sk-|ark-)[A-Za-z0-9._-]{8,}', '[REDACTED]', text)


def input_bound(body):
    # Engineering guard: fixed 1200x460 PNG, one image, <=16K UTF-8 text.
    # Reserve 65K input tokens (far above this image + prompt), check returned usage.
    text_bytes = 0
    images = 0
    for message in body['messages']:
        content = message['content']
        if isinstance(content, str):
            text_bytes += len(content.encode())
        else:
            for item in content:
                if item['type'] == 'text':
                    text_bytes += len(item['text'].encode())
                elif item['type'] == 'image_url':
                    from io import BytesIO
                    from PIL import Image
                    raw = base64.b64decode(item['image_url']['url'].split(',', 1)[1], validate=True)
                    if len(raw) > 2_000_000:
                        raise ValueError('image too large')
                    with Image.open(BytesIO(raw)) as picture:
                        if picture.format != 'PNG' or picture.size != (1200, 460):
                            raise ValueError('unexpected image size')
                    images += 1
                else:
                    raise ValueError('unsupported content')
    if text_bytes > 16384 or images > 1:
        raise ValueError('request exceeds frozen input scope')
    return INPUT_RESERVATION


class Caller:
    def __init__(self, model, ledger, archive):
        self.model, self.ledger, self.archive = model, Path(ledger), Path(archive)
        self.requests_started = 0
        self.last_record = None

    def call(self, messages, identity):
        model = self.model
        if model['endpoint'] != ENDPOINTS.get(model['provider']):
            raise ModelCallError('unapproved_endpoint')
        key = read_key(model['key_env'])
        if not key:
            raise ModelCallError('missing_api_key')
        body = {'model': model['id'], 'messages': messages, 'temperature': 0,
            'max_tokens': MAX_OUTPUT, 'stream': False,
            'response_format': {'type': 'json_object'}, **model['request_extra']}
        bound = input_bound(body)
        request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        record_path = self.archive/(identity+'.json')
        if record_path.exists():
            raise ModelCallError('archived_request_exists_use_replay')
        reservation = (bound*model['input_per_million']+MAX_OUTPUT*model['output_per_million'])/1e6
        request_id = hashlib.sha256((str(self.archive.resolve())+identity+request_hash).encode()).hexdigest()
        try:
            reserve(self.ledger, reservation, request_id)
        except (OSError, ValueError):
            raise ModelCallError('budget_not_admitted') from None
        self.requests_started += 1
        response, elapsed, error = send(model, body, key)
        record = {'schema': 'live-chain-provider-call-1', 'identity': identity,
            'requested_model': model['id'], 'provider': model['provider'],
            'request_sha256': request_hash, 'request_id_local': request_id,
            'created_at': now(), 'elapsed_ms': round(elapsed, 3),
            'error': error, 'usage': None, 'estimated_yuan': None,
            'reservation_yuan': reservation, 'returned_model': None,
            'provider_request_id': None, 'finish_reason': None, 'content': None}
        if response is not None:
            try:
                usage = response['usage']
                inp, out = usage['prompt_tokens'], usage['completion_tokens']
                if type(inp) is not int or type(out) is not int or inp < 0 or out < 0:
                    raise ValueError('invalid usage')
                record['usage'] = {'input_tokens': inp, 'output_tokens': out}
                record['estimated_yuan'] = (inp*model['input_per_million']+out*model['output_per_million'])/1e6
                if inp > bound or out > MAX_OUTPUT:
                    raise ValueError('usage exceeds input or output envelope')
                settle(self.ledger, request_id, record['estimated_yuan'])
            except (ValueError, KeyError, TypeError, OSError):
                record['error'] = 'usage_or_budget_uncertain'
            try:
                choice = response['choices'][0]
                content = choice['message']['content']
                if not isinstance(content, str) or len(content.encode()) > 32768:
                    raise ValueError('invalid response content')
                record['content'] = sanitize(content)
                returned = response.get('model')
                record['returned_model'] = sanitize(returned) if isinstance(returned, str) and len(returned) < 150 else None
                rid = response.get('id')
                record['provider_request_id'] = sanitize(rid) if isinstance(rid, str) and len(rid) < 180 else None
                record['finish_reason'] = sanitize(choice.get('finish_reason'))
                if record['finish_reason'] != 'stop':
                    record['error'] = record['error'] or 'nonstop_or_truncated'
            except (KeyError, IndexError, TypeError, ValueError):
                record['error'] = record['error'] or 'invalid_provider_response'
        self.archive.mkdir(parents=True, exist_ok=True)
        with record_path.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        self.last_record = record
        if record['error']:
            raise ModelCallError(record['error'])
        usage = record['usage']
        return ModelResponse(record['content'], model['id'], record['returned_model'],
            record['provider_request_id'], TokenUsage(usage['input_tokens'], usage['output_tokens']),
            elapsed, record['finish_reason'])


class Observer:
    mode = 'live_official_vision'

    def __init__(self, caller, prompt, identity):
        self.caller, self.prompt, self.identity = caller, prompt, identity
        self.requested_model = caller.model['id']
        self.calls = 0
        self.usage = self.elapsed_ms = None

    @property
    def requests_started(self):
        return self.caller.requests_started

    def observe(self, image):
        if self.calls:
            raise ModelCallError('visual_call_limit')
        self.calls += 1
        result = self.caller.call([{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+base64.b64encode(image).decode()}},
            {'type': 'text', 'text': self.prompt}]}], self.identity)
        self.usage = {'prompt_tokens': result.usage.input_tokens, 'completion_tokens': result.usage.output_tokens}
        self.elapsed_ms = result.elapsed_ms
        return result.content


class Choice:
    mode = 'live_official_choice'

    def __init__(self, caller, identity):
        self.caller, self.identity = caller, identity
        self.calls = 0
        self.response = None

    @property
    def requests_started(self):
        return self.caller.requests_started

    def complete(self, facts):
        if self.calls:
            raise ModelCallError('decision_call_limit')
        self.calls += 1
        from research_motion_choice import RESEARCH_SYSTEM
        safe = [{'fact_id': fact.fact_id, 'name': fact.name, 'value': fact.value} for fact in facts]
        self.response = self.caller.call([
            {'role': 'system', 'content': RESEARCH_SYSTEM},
            {'role': 'user', 'content': json.dumps({'facts': safe}, ensure_ascii=False)}], self.identity)
        return self.response
