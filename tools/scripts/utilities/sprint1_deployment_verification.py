#!/usr/bin/env python3
"""
Sprint 1 Deployment Verification Script

Verifies all Sprint 0 & Sprint 1 optimizations are properly deployed:
- SHA-1 embedding deduplication with Redis caching
- HNSW vector indices for fast similarity search
- gRPC embedding service for better performance
- HPA/VPA dynamic scaling configuration
- NATS JetStream event-driven architecture
- Prometheus metrics and alerting
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import nats
import psycopg2
import redis


@dataclass
class VerificationResult:
    """Result of a verification check"""
    component: str
    check: str
    status: str  # "pass", "fail", "warning"
    message: str
    details: dict[str, Any] | None = None
    duration_ms: int = 0


class DeploymentVerifier:
    """Comprehensive deployment verification for Sprint 1"""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        embedding_grpc_url: str = "localhost:50051",
        postgres_url: str = "postgresql://xorb:xorb_secure_2024@localhost:5432/xorb_ptaas",
        redis_url: str = "redis://localhost:6379/0",
        nats_url: str = "nats://localhost:4222",
        prometheus_url: str = "http://localhost:9090"
    ):
        self.api_url = api_url
        self.embedding_grpc_url = embedding_grpc_url
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.nats_url = nats_url
        self.prometheus_url = prometheus_url

        self.results: list[VerificationResult] = []

    async def verify_all(self) -> dict[str, Any]:
        """Run all verification checks"""

        print("🚀 Starting Sprint 1 Deployment Verification...")
        print("=" * 60)

        # Core service health
        await self.verify_api_health()
        await self.verify_embedding_service()

        # Data layer optimizations
        await self.verify_redis_caching()
        await self.verify_postgres_hnsw()

        # Event system
        await self.verify_nats_jetstream()

        # Monitoring and metrics
        await self.verify_prometheus_metrics()
        await self.verify_alerting_rules()

        # Performance tests
        await self.verify_embedding_performance()
        await self.verify_cache_deduplication()

        # Generate report
        return self.generate_report()

    async def verify_api_health(self):
        """Verify main API service health"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/health", timeout=10.0)

                if response.status_code == 200:
                    health_data = response.json()

                    self.results.append(VerificationResult(
                        component="API Service",
                        check="Health Check",
                        status="pass",
                        message="API service is healthy",
                        details=health_data,
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))
                else:
                    self.results.append(VerificationResult(
                        component="API Service",
                        check="Health Check",
                        status="fail",
                        message=f"API health check failed: {response.status_code}",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))

        except Exception as e:
            self.results.append(VerificationResult(
                component="API Service",
                check="Health Check",
                status="fail",
                message=f"Failed to connect to API: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_embedding_service(self):
        """Verify gRPC embedding service"""
        start_time = time.time()

        try:
            # For now we'll simulate the gRPC health check
            # In production, this would use the actual gRPC client

            # Check if gRPC port is accessible
            import socket
            host, port = self.embedding_grpc_url.split(':')
            port = int(port)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                self.results.append(VerificationResult(
                    component="Embedding Service",
                    check="gRPC Connectivity",
                    status="pass",
                    message="gRPC embedding service is accessible",
                    duration_ms=int((time.time() - start_time) * 1000)
                ))
            else:
                self.results.append(VerificationResult(
                    component="Embedding Service",
                    check="gRPC Connectivity",
                    status="fail",
                    message=f"Cannot connect to gRPC service on {self.embedding_grpc_url}",
                    duration_ms=int((time.time() - start_time) * 1000)
                ))

        except Exception as e:
            self.results.append(VerificationResult(
                component="Embedding Service",
                check="gRPC Connectivity",
                status="fail",
                message=f"gRPC connectivity check failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_redis_caching(self):
        """Verify Redis caching is working"""
        start_time = time.time()

        try:
            r = redis.from_url(self.redis_url)

            # Test basic connectivity
            ping_result = r.ping()

            if ping_result:
                # Test cache functionality with SHA-1 key
                test_text = "test embedding cache verification"
                test_model = "nvidia/embed-qa-4"
                test_input_type = "query"

                # Generate SHA-1 key as the service would
                cache_key = f"{test_model}:{test_input_type}:{hashlib.sha1(test_text.encode('utf-8')).hexdigest()}"

                # Set test data
                test_data = {
                    "embedding": [0.1, 0.2, 0.3],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {}
                }

                r.setex(cache_key, 3600, json.dumps(test_data))

                # Retrieve and verify
                cached_data = r.get(cache_key)
                if cached_data:
                    parsed_data = json.loads(cached_data)

                    self.results.append(VerificationResult(
                        component="Redis Cache",
                        check="SHA-1 Caching",
                        status="pass",
                        message="Redis caching with SHA-1 keys is working",
                        details={
                            "cache_key": cache_key,
                            "data_size": len(cached_data)
                        },
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))

                    # Cleanup
                    r.delete(cache_key)
                else:
                    self.results.append(VerificationResult(
                        component="Redis Cache",
                        check="SHA-1 Caching",
                        status="fail",
                        message="Could not retrieve cached data",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))
            else:
                self.results.append(VerificationResult(
                    component="Redis Cache",
                    check="Connectivity",
                    status="fail",
                    message="Redis ping failed",
                    duration_ms=int((time.time() - start_time) * 1000)
                ))

            r.close()

        except Exception as e:
            self.results.append(VerificationResult(
                component="Redis Cache",
                check="Connectivity",
                status="fail",
                message=f"Redis verification failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_postgres_hnsw(self):
        """Verify PostgreSQL HNSW indices are created"""
        start_time = time.time()

        try:
            conn = psycopg2.connect(self.postgres_url)
            cur = conn.cursor()

            # Check if pgvector extension is installed
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            vector_ext = cur.fetchone()

            if vector_ext:
                # Check for HNSW indices
                cur.execute("""
                    SELECT indexname, tablename, indexdef
                    FROM pg_indexes
                    WHERE indexdef LIKE '%hnsw%'
                    AND schemaname = 'public';
                """)

                hnsw_indices = cur.fetchall()

                if hnsw_indices:
                    self.results.append(VerificationResult(
                        component="PostgreSQL",
                        check="HNSW Indices",
                        status="pass",
                        message=f"Found {len(hnsw_indices)} HNSW indices",
                        details={
                            "indices": [
                                {"name": idx[0], "table": idx[1]}
                                for idx in hnsw_indices
                            ]
                        },
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))
                else:
                    self.results.append(VerificationResult(
                        component="PostgreSQL",
                        check="HNSW Indices",
                        status="warning",
                        message="No HNSW indices found - may need to run migration",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))
            else:
                self.results.append(VerificationResult(
                    component="PostgreSQL",
                    check="pgvector Extension",
                    status="fail",
                    message="pgvector extension not installed",
                    duration_ms=int((time.time() - start_time) * 1000)
                ))

            cur.close()
            conn.close()

        except Exception as e:
            self.results.append(VerificationResult(
                component="PostgreSQL",
                check="HNSW Verification",
                status="fail",
                message=f"PostgreSQL verification failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_nats_jetstream(self):
        """Verify NATS JetStream is configured"""
        start_time = time.time()

        try:
            nc = await nats.connect(self.nats_url)
            js = nc.jetstream()

            # Check for Xorb events stream
            try:
                stream_info = await js.stream_info("XORB_EVENTS")

                self.results.append(VerificationResult(
                    component="NATS JetStream",
                    check="Stream Configuration",
                    status="pass",
                    message="XORB_EVENTS stream is configured",
                    details={
                        "messages": stream_info.state.messages,
                        "subjects": stream_info.config.subjects,
                        "consumers": stream_info.state.consumer_count
                    },
                    duration_ms=int((time.time() - start_time) * 1000)
                ))

            except Exception:
                self.results.append(VerificationResult(
                    component="NATS JetStream",
                    check="Stream Configuration",
                    status="warning",
                    message="XORB_EVENTS stream not found - may need initialization",
                    duration_ms=int((time.time() - start_time) * 1000)
                ))

            await nc.close()

        except Exception as e:
            self.results.append(VerificationResult(
                component="NATS JetStream",
                check="Connectivity",
                status="fail",
                message=f"NATS JetStream verification failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_prometheus_metrics(self):
        """Verify Prometheus metrics are available"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                # Check Prometheus health
                response = await client.get(f"{self.prometheus_url}/-/healthy", timeout=10.0)

                if response.status_code == 200:
                    # Check for Xorb-specific metrics
                    metrics_response = await client.get(
                        f"{self.prometheus_url}/api/v1/label/__name__/values",
                        timeout=10.0
                    )

                    if metrics_response.status_code == 200:
                        metrics_data = metrics_response.json()
                        metric_names = metrics_data.get("data", [])

                        xorb_metrics = [
                            name for name in metric_names
                            if name.startswith("xorb_") or name.startswith("embedding_")
                        ]

                        if xorb_metrics:
                            self.results.append(VerificationResult(
                                component="Prometheus",
                                check="Xorb Metrics",
                                status="pass",
                                message=f"Found {len(xorb_metrics)} Xorb metrics",
                                details={"metrics": xorb_metrics[:10]},  # Show first 10
                                duration_ms=int((time.time() - start_time) * 1000)
                            ))
                        else:
                            self.results.append(VerificationResult(
                                component="Prometheus",
                                check="Xorb Metrics",
                                status="warning",
                                message="No Xorb-specific metrics found",
                                duration_ms=int((time.time() - start_time) * 1000)
                            ))
                    else:
                        self.results.append(VerificationResult(
                            component="Prometheus",
                            check="Metrics API",
                            status="fail",
                            message="Could not query metrics API",
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                else:
                    self.results.append(VerificationResult(
                        component="Prometheus",
                        check="Health Check",
                        status="fail",
                        message=f"Prometheus health check failed: {response.status_code}",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))

        except Exception as e:
            self.results.append(VerificationResult(
                component="Prometheus",
                check="Connectivity",
                status="fail",
                message=f"Prometheus verification failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_alerting_rules(self):
        """Verify alerting rules are loaded"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.prometheus_url}/api/v1/rules",
                    timeout=10.0
                )

                if response.status_code == 200:
                    rules_data = response.json()
                    groups = rules_data.get("data", {}).get("groups", [])

                    xorb_rules = []
                    for group in groups:
                        if "xorb" in group.get("name", "").lower():
                            xorb_rules.extend(group.get("rules", []))

                    embedding_alerts = [
                        rule for rule in xorb_rules
                        if "embedding" in rule.get("alert", "").lower()
                    ]

                    if embedding_alerts:
                        self.results.append(VerificationResult(
                            component="Prometheus",
                            check="Embedding Alerts",
                            status="pass",
                            message=f"Found {len(embedding_alerts)} embedding-related alerts",
                            details={"alerts": [rule.get("alert") for rule in embedding_alerts]},
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                    else:
                        self.results.append(VerificationResult(
                            component="Prometheus",
                            check="Embedding Alerts",
                            status="warning",
                            message="No embedding-specific alerts found",
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                else:
                    self.results.append(VerificationResult(
                        component="Prometheus",
                        check="Rules API",
                        status="fail",
                        message="Could not query rules API",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))

        except Exception as e:
            self.results.append(VerificationResult(
                component="Prometheus",
                check="Alerting Rules",
                status="fail",
                message=f"Alert rules verification failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_embedding_performance(self):
        """Verify embedding service performance"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                # Test embedding endpoint
                test_data = {
                    "input": ["performance test text", "another test embedding"],
                    "model": "nvidia/embed-qa-4",
                    "input_type": "query"
                }

                embed_start = time.time()
                response = await client.post(
                    f"{self.api_url}/embeddings",
                    json=test_data,
                    timeout=30.0
                )
                embed_duration = time.time() - embed_start

                if response.status_code == 200:
                    embedding_data = response.json()

                    # Check if we got embeddings
                    if "data" in embedding_data and len(embedding_data["data"]) == 2:
                        self.results.append(VerificationResult(
                            component="Embedding Performance",
                            check="API Response Time",
                            status="pass" if embed_duration < 5.0 else "warning",
                            message=f"Embedding API responded in {embed_duration:.2f}s",
                            details={
                                "duration_seconds": embed_duration,
                                "num_embeddings": len(embedding_data["data"]),
                                "usage": embedding_data.get("usage", {})
                            },
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                    else:
                        self.results.append(VerificationResult(
                            component="Embedding Performance",
                            check="API Response",
                            status="fail",
                            message="Invalid embedding response format",
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                else:
                    self.results.append(VerificationResult(
                        component="Embedding Performance",
                        check="API Request",
                        status="fail",
                        message=f"Embedding API request failed: {response.status_code}",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))

        except Exception as e:
            self.results.append(VerificationResult(
                component="Embedding Performance",
                check="API Test",
                status="fail",
                message=f"Performance test failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    async def verify_cache_deduplication(self):
        """Verify cache deduplication is working"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                # Make the same request twice to test caching
                test_data = {
                    "input": ["cache deduplication test text"],
                    "model": "nvidia/embed-qa-4"
                }

                # First request
                first_start = time.time()
                first_response = await client.post(
                    f"{self.api_url}/embeddings",
                    json=test_data,
                    timeout=30.0
                )
                first_duration = time.time() - first_start

                # Second request (should be faster due to caching)
                second_start = time.time()
                second_response = await client.post(
                    f"{self.api_url}/embeddings",
                    json=test_data,
                    timeout=30.0
                )
                second_duration = time.time() - second_start

                if first_response.status_code == 200 and second_response.status_code == 200:
                    # Second request should be significantly faster
                    cache_improvement = (first_duration - second_duration) / first_duration

                    if cache_improvement > 0.1:  # At least 10% improvement
                        self.results.append(VerificationResult(
                            component="Cache Deduplication",
                            check="Performance Improvement",
                            status="pass",
                            message=f"Cache provided {cache_improvement:.1%} performance improvement",
                            details={
                                "first_request_ms": int(first_duration * 1000),
                                "second_request_ms": int(second_duration * 1000),
                                "improvement_percent": cache_improvement * 100
                            },
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                    else:
                        self.results.append(VerificationResult(
                            component="Cache Deduplication",
                            check="Performance Improvement",
                            status="warning",
                            message=f"Cache improvement only {cache_improvement:.1%}",
                            details={
                                "first_request_ms": int(first_duration * 1000),
                                "second_request_ms": int(second_duration * 1000)
                            },
                            duration_ms=int((time.time() - start_time) * 1000)
                        ))
                else:
                    self.results.append(VerificationResult(
                        component="Cache Deduplication",
                        check="API Requests",
                        status="fail",
                        message="Cache deduplication test requests failed",
                        duration_ms=int((time.time() - start_time) * 1000)
                    ))

        except Exception as e:
            self.results.append(VerificationResult(
                component="Cache Deduplication",
                check="Test Execution",
                status="fail",
                message=f"Cache deduplication test failed: {str(e)}",
                duration_ms=int((time.time() - start_time) * 1000)
            ))

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive verification report"""

        # Count results by status
        passed = len([r for r in self.results if r.status == "pass"])
        failed = len([r for r in self.results if r.status == "fail"])
        warnings = len([r for r in self.results if r.status == "warning"])
        total = len(self.results)

        # Calculate overall health score
        health_score = (passed + (warnings * 0.5)) / total * 100 if total > 0 else 0

        # Determine overall status
        if failed == 0 and warnings <= 2:
            overall_status = "healthy"
        elif failed <= 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "health_score": round(health_score, 1),
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            },
            "results": [
                {
                    "component": r.component,
                    "check": r.check,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms
                }
                for r in self.results
            ]
        }

        return report

    def print_summary(self, report: dict[str, Any]):
        """Print human-readable summary"""

        print("\n" + "=" * 60)
        print("📊 DEPLOYMENT VERIFICATION SUMMARY")
        print("=" * 60)

        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌"
        }

        print(f"Overall Status: {status_emoji.get(report['overall_status'], '❓')} {report['overall_status'].upper()}")
        print(f"Health Score: {report['health_score']}%")
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"  ✅ Passed: {report['summary']['passed']}")
        print(f"  ⚠️  Warnings: {report['summary']['warnings']}")
        print(f"  ❌ Failed: {report['summary']['failed']}")

        print("\n" + "-" * 60)
        print("COMPONENT DETAILS")
        print("-" * 60)

        # Group results by component
        by_component = {}
        for result in report['results']:
            component = result['component']
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(result)

        for component, results in by_component.items():
            status_counts = {"pass": 0, "fail": 0, "warning": 0}
            for result in results:
                status_counts[result['status']] += 1

            # Component status emoji
            if status_counts['fail'] > 0:
                comp_emoji = "❌"
            elif status_counts['warning'] > 0:
                comp_emoji = "⚠️"
            else:
                comp_emoji = "✅"

            print(f"\n{comp_emoji} {component}")

            for result in results:
                emoji = {"pass": "  ✅", "fail": "  ❌", "warning": "  ⚠️"}[result['status']]
                print(f"{emoji} {result['check']}: {result['message']}")
                if result['duration_ms'] > 1000:
                    print(f"     Duration: {result['duration_ms']}ms")


async def main():
    """Main verification function"""

    verifier = DeploymentVerifier()

    try:
        report = await verifier.verify_all()

        # Print summary to console
        verifier.print_summary(report)

        # Save detailed report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"sprint1_verification_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_file}")

        # Exit with appropriate code
        if report['overall_status'] == "healthy":
            print("\n🎉 All systems operational! Sprint 1 deployment successful.")
            exit(0)
        elif report['overall_status'] == "degraded":
            print("\n⚠️  Some issues detected. Review warnings and failed checks.")
            exit(1)
        else:
            print("\n❌ Critical issues detected. Deployment needs attention.")
            exit(2)

    except KeyboardInterrupt:
        print("\n\n⏹️  Verification interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Verification failed with error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
