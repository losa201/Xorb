#!/usr/bin/env python3
"""
Performance Benchmark Suite for Xorb 2.0
Tests connection pooling, NVIDIA embeddings, and overall system performance
"""

import asyncio
import time
import statistics
import sys
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import requests
import json

class XorbPerformanceBenchmark:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: Dict[str, Any] = {}
        
    async def benchmark_database_connections(self, concurrent_requests: int = 50) -> Dict[str, float]:
        """Benchmark database connection pool performance"""
        print(f"🔗 Benchmarking database connections ({concurrent_requests} concurrent)")
        
        # Import here to avoid import issues
        sys.path.append('/root/Xorb')
        from packages.xorb_core.xorb_core.database.connection_pool import get_connection_pool
        
        pool = await get_connection_pool()
        
        async def single_query():
            start = time.time()
            result = await pool.execute_query("SELECT 1 as test")
            return time.time() - start
        
        # Warm up
        for _ in range(5):
            await single_query()
        
        # Benchmark concurrent queries
        start_time = time.time()
        tasks = [single_query() for _ in range(concurrent_requests)]
        query_times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        stats = {
            'total_time': total_time,
            'avg_query_time': statistics.mean(query_times),
            'median_query_time': statistics.median(query_times),
            'p95_query_time': statistics.quantiles(query_times, n=20)[18],  # 95th percentile
            'queries_per_second': concurrent_requests / total_time,
            'min_time': min(query_times),
            'max_time': max(query_times)
        }
        
        print(f"  ✅ {concurrent_requests} queries in {total_time:.2f}s")
        print(f"  📊 QPS: {stats['queries_per_second']:.1f}")
        print(f"  📊 Avg latency: {stats['avg_query_time']*1000:.1f}ms")
        print(f"  📊 P95 latency: {stats['p95_query_time']*1000:.1f}ms")
        
        return stats
    
    def benchmark_api_endpoints(self, concurrent_requests: int = 20) -> Dict[str, Any]:
        """Benchmark API endpoint performance"""
        print(f"🌐 Benchmarking API endpoints ({concurrent_requests} concurrent)")
        
        def test_health_endpoint():
            start = time.time()
            try:
                response = requests.get(f"{self.base_url}/health", timeout=10)
                duration = time.time() - start
                return {
                    'duration': duration,
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'duration': time.time() - start,
                    'status_code': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            start_time = time.time()
            futures = [executor.submit(test_health_endpoint) for _ in range(concurrent_requests)]
            results = [future.result() for future in futures]
            total_time = time.time() - start_time
        
        successful_requests = [r for r in results if r['success']]
        success_rate = len(successful_requests) / len(results)
        
        if successful_requests:
            response_times = [r['duration'] for r in successful_requests]
            stats = {
                'total_time': total_time,
                'success_rate': success_rate,
                'requests_per_second': len(successful_requests) / total_time,
                'avg_response_time': statistics.mean(response_times),
                'median_response_time': statistics.median(response_times),
                'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                'min_time': min(response_times),
                'max_time': max(response_times)
            }
        else:
            stats = {
                'total_time': total_time,
                'success_rate': 0,
                'requests_per_second': 0,
                'error': 'All requests failed'
            }
        
        print(f"  ✅ {len(successful_requests)}/{len(results)} requests successful")
        print(f"  📊 Success rate: {success_rate:.1%}")
        if successful_requests:
            print(f"  📊 RPS: {stats['requests_per_second']:.1f}")
            print(f"  📊 Avg response: {stats['avg_response_time']*1000:.1f}ms")
        
        return stats
    
    def benchmark_nvidia_embeddings(self, batch_sizes: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """Benchmark NVIDIA embeddings performance"""
        print("🧠 Benchmarking NVIDIA embeddings")
        
        test_texts = [
            "SQL injection vulnerability in web application",
            "Cross-site scripting attack vector analysis",
            "Buffer overflow in network parsing code",
            "Authentication bypass using JWT manipulation",
            "Directory traversal vulnerability assessment",
            "Memory corruption in image processing",
            "Privilege escalation through file permissions",
            "Session hijacking via cookie manipulation",
            "Command injection in file upload feature",
            "Race condition in multi-threaded application"
        ]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            batch_texts = test_texts[:batch_size]
            
            payload = {
                "input": batch_texts,
                "model": "nvidia/embed-qa-4",
                "input_type": "passage",
                "encoding_format": "float"
            }
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    json=payload,
                    headers={"Authorization": "Bearer demo-token"},
                    timeout=30
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings_per_second = batch_size / duration
                    
                    results[f"batch_{batch_size}"] = {
                        'duration': duration,
                        'embeddings_per_second': embeddings_per_second,
                        'texts_per_embedding': batch_size,
                        'avg_time_per_embedding': duration / batch_size,
                        'embedding_dimension': len(data['data'][0]['embedding']) if data['data'] else 0,
                        'total_tokens': data.get('usage', {}).get('total_tokens', 0)
                    }
                    
                    print(f"    ✅ {batch_size} embeddings in {duration:.2f}s ({embeddings_per_second:.1f} emb/s)")
                else:
                    print(f"    ❌ Failed with status {response.status_code}")
                    results[f"batch_{batch_size}"] = {'error': f"HTTP {response.status_code}"}
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results[f"batch_{batch_size}"] = {'error': str(e)}
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        print("💾 Checking memory usage")
        
        try:
            import psutil
            process = psutil.Process()
            
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            stats = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': memory_percent,
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
            }
            
            print(f"  📊 RSS Memory: {stats['rss_mb']:.1f} MB")
            print(f"  📊 Memory %: {stats['memory_percent']:.1f}%")
            print(f"  📊 Threads: {stats['num_threads']}")
            
            return stats
            
        except ImportError:
            print("  ⚠️  psutil not available, skipping memory benchmark")
            return {'error': 'psutil not available'}
        except Exception as e:
            print(f"  ❌ Memory benchmark failed: {e}")
            return {'error': str(e)}
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🚀 Xorb 2.0 Performance Benchmark Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Database benchmarks
        try:
            self.results['database'] = await self.benchmark_database_connections()
        except Exception as e:
            print(f"  ❌ Database benchmark failed: {e}")
            self.results['database'] = {'error': str(e)}
        
        print()
        
        # API benchmarks
        self.results['api'] = self.benchmark_api_endpoints()
        print()
        
        # NVIDIA embeddings benchmarks
        self.results['embeddings'] = self.benchmark_nvidia_embeddings()
        print()
        
        # Memory usage
        self.results['memory'] = self.benchmark_memory_usage()
        print()
        
        total_time = time.time() - start_time
        self.results['benchmark_duration'] = total_time
        
        print("🎯 Benchmark Summary")
        print("=" * 30)
        print(f"  ⏱️  Total time: {total_time:.1f}s")
        
        # Performance score calculation
        score = self.calculate_performance_score()
        self.results['performance_score'] = score
        
        print(f"  📊 Performance Score: {score}/100")
        print(f"  📊 Status: {'🟢 Excellent' if score >= 80 else '🟡 Good' if score >= 60 else '🔴 Needs Improvement'}")
        
        return self.results
    
    def calculate_performance_score(self) -> int:
        """Calculate overall performance score"""
        score = 0
        
        # Database performance (30 points)
        if 'database' in self.results and 'queries_per_second' in self.results['database']:
            qps = self.results['database']['queries_per_second']
            if qps >= 100:
                score += 30
            elif qps >= 50:
                score += 20
            elif qps >= 20:
                score += 15
            else:
                score += 10
        
        # API performance (25 points)
        if 'api' in self.results and 'success_rate' in self.results['api']:
            success_rate = self.results['api']['success_rate']
            rps = self.results['api'].get('requests_per_second', 0)
            
            if success_rate >= 0.99 and rps >= 50:
                score += 25
            elif success_rate >= 0.95 and rps >= 30:
                score += 20
            elif success_rate >= 0.90:
                score += 15
            else:
                score += 5
        
        # Embeddings performance (25 points)
        if 'embeddings' in self.results:
            batch_10 = self.results['embeddings'].get('batch_10', {})
            if 'embeddings_per_second' in batch_10:
                eps = batch_10['embeddings_per_second']
                if eps >= 5:
                    score += 25
                elif eps >= 3:
                    score += 20
                elif eps >= 1:
                    score += 15
                else:
                    score += 10
        
        # Memory efficiency (20 points)
        if 'memory' in self.results and 'memory_percent' in self.results['memory']:
            mem_percent = self.results['memory']['memory_percent']
            if mem_percent <= 5:
                score += 20
            elif mem_percent <= 10:
                score += 15
            elif mem_percent <= 20:
                score += 10
            else:
                score += 5
        
        return min(score, 100)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"/root/Xorb/benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Results saved to {filename}")

async def main():
    """Run the benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xorb 2.0 Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    benchmark = XorbPerformanceBenchmark(base_url=args.url)
    results = await benchmark.run_full_benchmark()
    
    if args.output:
        benchmark.save_results(args.output)
    else:
        benchmark.save_results()
    
    return results['performance_score']

if __name__ == "__main__":
    try:
        score = asyncio.run(main())
        sys.exit(0 if score >= 60 else 1)  # Exit with error if score < 60
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)