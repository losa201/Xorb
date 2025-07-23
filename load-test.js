import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 500 },  // ramp up to 500 virtual users
    { duration: '3m', target: 500 },
    { duration: '1m', target: 0 },    // ramp down
  ],
};

export default function () {
  const res = http.post('https://api.xorb.local/workflow/start', JSON.stringify({
    tenant_id: 'test-tenant',
    task: 'scan',
    target: 'example.com'
  }), { headers: { 'Content-Type': 'application/json' } });
  check(res, { 'status was 200': (r) => r.status === 200 });
  sleep(1);
}
