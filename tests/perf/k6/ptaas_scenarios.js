/**
 * K6 Performance Test Scenarios for PTaaS on AMD EPYC 7002
 * Target: P95 < 2s, Error rate < 0.5%, Fairness index ≥ 0.7
 * Load: 10 tenants × 8 concurrent jobs = 80 in-flight jobs
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend, Gauge } from 'k6/metrics';
import { randomIntBetween, randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics for PTaaS performance tracking
export const errorRate = new Rate('ptaas_errors');
export const requestDuration = new Trend('ptaas_request_duration');
export const activeJobs = new Gauge('ptaas_active_jobs');
export const fairnessIndex = new Gauge('ptaas_fairness_index');
export const jobsCompleted = new Counter('ptaas_jobs_completed');
export const jobsFailed = new Counter('ptaas_jobs_failed');
export const queueTime = new Trend('ptaas_queue_time_ms');
export const executionTime = new Trend('ptaas_execution_time_ms');

// Test configuration for EPYC 7002 target
export const options = {
  scenarios: {
    // Main load test: 10 tenants with mixed workload
    epyc_mixed_load: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '2m', target: 20 },   // Ramp up
        { duration: '5m', target: 80 },   // Target load (10 tenants × 8 jobs)
        { duration: '10m', target: 80 },  // Sustained load
        { duration: '2m', target: 20 },   // Ramp down
        { duration: '1m', target: 0 },    // Cool down
      ],
      gracefulRampDown: '30s',
    },

    // Spike test for EPYC capacity
    epyc_spike_test: {
      executor: 'ramping-vus',
      startTime: '22m',
      startVUs: 80,
      stages: [
        { duration: '30s', target: 160 }, // 2x spike
        { duration: '1m', target: 160 },  // Sustained spike
        { duration: '30s', target: 80 },  // Back to normal
      ],
      gracefulRampDown: '30s',
    },

    // Tenant fairness validation
    fairness_test: {
      executor: 'per-vu-iterations',
      vus: 10,  // 10 tenants
      iterations: 50,  // 50 jobs per tenant
      startTime: '25m',
      maxDuration: '10m',
    }
  },

  thresholds: {
    // Performance targets for EPYC 7002
    'ptaas_request_duration': ['p(95)<2000'], // P95 < 2s
    'ptaas_errors': ['rate<0.005'],           // Error rate < 0.5%
    'ptaas_fairness_index': ['value>=0.7'],  // Fairness ≥ 0.7
    'http_req_duration': ['p(90)<1500', 'p(95)<2000'], // HTTP latency targets
    'http_req_failed': ['rate<0.01'],        // HTTP error rate < 1%
  },

  // EPYC-optimized settings
  batch: 20,           // Batch requests for efficiency
  batchPerHost: 10,    // Per-host batching
  discardResponseBodies: false, // Keep response for validation
};

// Base URL for PTaaS API
const BASE_URL = __ENV.PTAAS_BASE_URL || 'http://localhost:8000';
const API_TOKEN = __ENV.PTAAS_API_TOKEN || 'test-token';

// Test data: tenant configurations
const TENANTS = [
  'enterprise-corp', 'startup-inc', 'government-agency', 'financial-services',
  'healthcare-org', 'education-inst', 'retail-chain', 'manufacturing-co',
  'telecom-provider', 'cloud-vendor'
];

// Scan profiles with different resource requirements
const SCAN_PROFILES = {
  quick: { weight: 0.70, timeout: 30 },      // 70% - fast scans
  comprehensive: { weight: 0.05, timeout: 300 }, // 5% - slow scans
  stealth: { weight: 0.15, timeout: 120 },   // 15% - medium scans
  'web-focused': { weight: 0.10, timeout: 90 } // 10% - medium scans
};

// Target types for scanning
const TARGET_TYPES = [
  'single_host', 'network_range', 'domain', 'web_application', 'api_endpoint'
];

// Generate realistic test targets
function generateTarget() {
  const targetType = randomItem(TARGET_TYPES);

  switch (targetType) {
    case 'single_host':
      return `192.168.${randomIntBetween(1, 255)}.${randomIntBetween(1, 254)}`;
    case 'network_range':
      return `10.${randomIntBetween(0, 255)}.${randomIntBetween(0, 255)}.0/24`;
    case 'domain':
      return `test${randomIntBetween(1, 1000)}.example.com`;
    case 'web_application':
      return `https://webapp${randomIntBetween(1, 100)}.testdomain.com`;
    case 'api_endpoint':
      return `https://api${randomIntBetween(1, 50)}.service.com/v1`;
    default:
      return '192.168.1.100';
  }
}

// Select scan profile based on weights (70% fast, 25% medium, 5% slow)
function selectScanProfile() {
  const rand = Math.random();
  let cumulative = 0;

  for (const [profile, config] of Object.entries(SCAN_PROFILES)) {
    cumulative += config.weight;
    if (rand <= cumulative) {
      return profile;
    }
  }
  return 'quick'; // fallback
}

// Generate PTaaS scan request
function createScanRequest(tenantId) {
  const scanProfile = selectScanProfile();
  const targetCount = randomIntBetween(1, 5); // 1-5 targets per scan
  const targets = [];

  for (let i = 0; i < targetCount; i++) {
    targets.push(generateTarget());
  }

  return {
    targets: targets.map(target => ({
      host: target,
      ports: [22, 80, 443, 8080],
      scan_profile: scanProfile
    })),
    scan_type: randomItem(['discovery', 'vulnerability_scan', 'compliance_check']),
    priority: randomItem(['high', 'medium', 'low']),
    metadata: {
      tenant_id: tenantId,
      compliance_frameworks: randomItem([['PCI-DSS'], ['HIPAA'], ['SOX'], ['ISO-27001']]),
      scan_options: {
        stealth_mode: Math.random() > 0.7,
        deep_scan: scanProfile === 'comprehensive'
      }
    }
  };
}

// Main test function
export default function() {
  const tenantId = randomItem(TENANTS);
  const scanRequest = createScanRequest(tenantId);

  // Headers with tenant identification
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_TOKEN}`,
    'X-Tenant-ID': tenantId,
    'X-Request-ID': `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  };

  const startTime = Date.now();

  // Create PTaaS scan session
  const response = http.post(
    `${BASE_URL}/api/v1/ptaas/sessions`,
    JSON.stringify(scanRequest),
    {
      headers: headers,
      timeout: '60s',
      tags: {
        tenant: tenantId,
        scan_profile: scanRequest.targets[0].scan_profile,
        scenario: __ENV.SCENARIO || 'mixed_load'
      }
    }
  );

  const requestTime = Date.now() - startTime;
  requestDuration.add(requestTime, { tenant: tenantId });

  // Validate response
  const isSuccess = check(response, {
    'status is 201': (r) => r.status === 201,
    'response has session_id': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.session_id !== undefined;
      } catch (e) {
        return false;
      }
    },
    'response time < 5s': (r) => r.timings.duration < 5000,
  });

  if (!isSuccess) {
    errorRate.add(1, { tenant: tenantId });
    jobsFailed.add(1, { tenant: tenantId });
    return;
  }

  const sessionData = JSON.parse(response.body);
  const sessionId = sessionData.session_id;

  // Monitor scan progress
  let scanCompleted = false;
  let attempts = 0;
  const maxAttempts = 60; // 60 attempts × 2s = 2 minutes max wait

  while (!scanCompleted && attempts < maxAttempts) {
    sleep(2); // Poll every 2 seconds
    attempts++;

    const statusResponse = http.get(
      `${BASE_URL}/api/v1/ptaas/sessions/${sessionId}`,
      {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        timeout: '10s',
        tags: { tenant: tenantId, operation: 'status_check' }
      }
    );

    if (statusResponse.status === 200) {
      const statusData = JSON.parse(statusResponse.body);

      if (statusData.status === 'completed') {
        scanCompleted = true;
        jobsCompleted.add(1, { tenant: tenantId });

        // Extract performance metrics from response
        if (statusData.metrics) {
          if (statusData.metrics.queue_time_ms) {
            queueTime.add(statusData.metrics.queue_time_ms, { tenant: tenantId });
          }
          if (statusData.metrics.execution_time_ms) {
            executionTime.add(statusData.metrics.execution_time_ms, { tenant: tenantId });
          }
        }

      } else if (statusData.status === 'failed') {
        scanCompleted = true;
        errorRate.add(1, { tenant: tenantId });
        jobsFailed.add(1, { tenant: tenantId });
      }

      // Update active jobs gauge
      if (statusData.active_jobs !== undefined) {
        activeJobs.add(statusData.active_jobs);
      }

      // Update fairness metrics if available
      if (statusData.fairness_index !== undefined) {
        fairnessIndex.add(statusData.fairness_index);
      }
    }
  }

  if (!scanCompleted) {
    console.warn(`Scan ${sessionId} timeout after ${maxAttempts * 2} seconds`);
    errorRate.add(1, { tenant: tenantId, reason: 'timeout' });
  }

  // Brief pause between requests to simulate realistic usage
  sleep(randomIntBetween(1, 3));
}

// Setup function for test initialization
export function setup() {
  console.log('Starting PTaaS EPYC 7002 Performance Test');
  console.log(`Target URL: ${BASE_URL}`);
  console.log(`Tenants: ${TENANTS.length}`);
  console.log('Scan Profile Distribution:');
  for (const [profile, config] of Object.entries(SCAN_PROFILES)) {
    console.log(`  ${profile}: ${(config.weight * 100).toFixed(1)}%`);
  }

  // Verify API connectivity
  const healthResponse = http.get(`${BASE_URL}/api/v1/health`, {
    timeout: '10s'
  });

  if (healthResponse.status !== 200) {
    throw new Error(`API health check failed: ${healthResponse.status}`);
  }

  console.log('API health check passed');
  return { startTime: Date.now() };
}

// Teardown function for test cleanup
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`PTaaS Performance Test completed in ${duration.toFixed(1)} seconds`);

  // Final metrics summary would be handled by k6's built-in reporting
}

// Handle summary for custom reporting
export function handleSummary(data) {
  // Extract key metrics for EPYC validation
  const p95_latency = data.metrics.ptaas_request_duration?.p95 || 0;
  const error_rate = data.metrics.ptaas_errors?.rate || 0;
  const fairness_index = data.metrics.ptaas_fairness_index?.value || 0;

  const summary = {
    timestamp: new Date().toISOString(),
    test_environment: 'EPYC-7002',
    performance_targets: {
      p95_latency_ms: {
        target: 2000,
        actual: p95_latency,
        passed: p95_latency < 2000
      },
      error_rate: {
        target: 0.005,
        actual: error_rate,
        passed: error_rate < 0.005
      },
      fairness_index: {
        target: 0.7,
        actual: fairness_index,
        passed: fairness_index >= 0.7
      }
    },
    raw_metrics: data.metrics
  };

  return {
    'ptaas_perf_summary.json': JSON.stringify(summary, null, 2),
    stdout: `
┌─────────────────────────────────────────────────────────────┐
│                PTaaS EPYC 7002 Performance Results         │
├─────────────────────────────────────────────────────────────┤
│ P95 Latency:    ${p95_latency.toFixed(1)}ms (target: <2000ms) ${p95_latency < 2000 ? '✅' : '❌'}   │
│ Error Rate:     ${(error_rate * 100).toFixed(3)}% (target: <0.5%) ${error_rate < 0.005 ? '✅' : '❌'}    │
│ Fairness Index: ${fairness_index.toFixed(3)} (target: ≥0.7) ${fairness_index >= 0.7 ? '✅' : '❌'}       │
└─────────────────────────────────────────────────────────────┘
    `
  };
}
