#!/usr/bin/env python3
"""
Simple validation script for Xorb 2.0 EPYC deployment
"""

import subprocess
import time


def run_command(cmd, description):
    """Run a command and return result"""
    print(f"Testing: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True, result.stdout.strip()
        else:
            print(f"❌ {description}: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: Timeout")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description}: {str(e)}")
        return False, str(e)

def main():
    """Validate EPYC deployment"""
    print("🚀 Xorb 2.0 EPYC Deployment Validation")
    print("=" * 50)

    tests = [
        # Kubernetes cluster validation
        ("kubectl cluster-info", "Kubernetes cluster status"),
        ("kubectl get nodes", "Node availability"),

        # Namespace validation
        ("kubectl get namespace xorb-prod", "Xorb production namespace"),

        # DaemonSet validation
        ("kubectl get daemonset -n xorb-prod epyc-cpu-governor-optimizer", "EPYC CPU Governor DaemonSet"),
        ("kubectl get pods -n xorb-prod -l app.kubernetes.io/name=epyc-cpu-governor", "EPYC optimizer pods"),

        # Configuration validation
        ("kubectl get configmap -n xorb-prod epyc-optimization-config", "EPYC configuration"),
        ("kubectl get service -n xorb-prod epyc-cpu-governor-metrics", "Metrics service"),

        # RBAC validation
        ("kubectl get clusterrole epyc-system-optimizer", "EPYC system optimizer role"),
        ("kubectl get serviceaccount -n xorb-prod epyc-system-optimizer", "Service account"),

        # Pod logs validation
        ("kubectl logs -n xorb-prod --selector=app.kubernetes.io/name=epyc-cpu-governor --tail=5", "EPYC optimizer logs"),
    ]

    passed = 0
    total = len(tests)

    for cmd, desc in tests:
        success, output = run_command(cmd, desc)
        if success:
            passed += 1

        # Add delay between tests
        time.sleep(1)

    print("\n" + "=" * 50)
    print("🏁 Validation Summary")
    print(f"✅ Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All validations passed! EPYC deployment is healthy.")

        # Show current EPYC optimizer status
        print("\n📊 Current EPYC Optimizer Status:")
        success, pod_name = run_command(
            "kubectl get pods -n xorb-prod -l app.kubernetes.io/name=epyc-cpu-governor -o jsonpath='{.items[0].metadata.name}'",
            "Get EPYC pod name"
        )

        if success and pod_name:
            print(f"Pod: {pod_name}")
            success, logs = run_command(
                f"kubectl logs -n xorb-prod {pod_name} --tail=3",
                "Recent logs"
            )
            if success:
                for line in logs.split('\n'):
                    if line.strip():
                        print(f"  {line}")

        # Show resource usage
        print("\n💾 Resource Status:")
        run_command(
            "kubectl top pods -n xorb-prod --no-headers 2>/dev/null || echo 'Metrics not available'",
            "Pod resource usage"
        )

        return True
    else:
        print("⚠️  Some validations failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
