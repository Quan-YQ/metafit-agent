"""An oversized request must deliver 413 without waiting for its claimed body."""
import http.client
import json

from metafit_agent.simulation_api import MAX_BODY
from test_r5_simulation_api import request, service


def test_repeated_oversized_requests_deliver_the_declared_error(service):
    for _ in range(128):
        status, body = request(service[1], 'POST', '/sim/v0/sessions', raw=b' ' * (MAX_BODY + 1))
        assert status == 413 and body['code'] == 'request_too_large'


def test_huge_declared_body_is_rejected_without_waiting_for_upload(service):
    conn = http.client.HTTPConnection('127.0.0.1', service[1].server_port, timeout=2)
    try:
        conn.putrequest('POST', '/sim/v0/sessions')
        conn.putheader('Content-Type', 'application/json')
        conn.putheader('Content-Length', str(10**12))
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 413
        assert json.loads(response.read())['code'] == 'request_too_large'
    finally:
        conn.close()
