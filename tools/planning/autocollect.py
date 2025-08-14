#!/usr/bin/env python3
import os
import re
import json
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

class RepoScanner:
    def __init__(self):
        self.root = Path.cwd()
        self.matrix = []
        self.redacted = {}
        self.patterns = {
            'token': r'[a-zA-Z0-9]{20,}',
            'password': r'password=\S+',
            'secret': r'secret=\S+',
            'key': r'\.pem|\.key|\.p12'
        }

    def redact(self, value):
        if not value:
            return value
        for name, pattern in self.patterns.items():
            if re.search(pattern, str(value)):
                prefix = str(value)[:8] if isinstance(value, str) else 'bin'
                key = f"{name}_{hashlib.sha256(prefix.encode()).hexdigest()[:12]}"
                self.redacted[key] = value
                return f"<REDACTED:{key}>"
        return value

    def add_entry(self, key, value, confidence, source, notes=""):
        self.matrix.append({
            'key': key,
            'discovered_value': self.redact(value),
            'confidence': confidence,
            'source': source,
            'notes': notes
        })

    def scan_repo_shape(self):
        dirs = ['proto', 'services', 'platform', 'infra', 'compose', 'docs/architecture']
        for d in dirs:
            exists = self.root.joinpath(d).exists()
            self.add_entry(f"platform.current_state.repo_layout.{d}", exists, 1.0, d)

    def scan_adrs(self):
        for i in range(1, 5):
            path = f"docs/architecture/ADR-00{i}.md"
            if self.root.joinpath(path).exists():
                self.add_entry(f"platform.current_state.adrs_present.ADR-00{i}", True, 1.0, path)
                # Parse key decisions
                content = self.root.joinpath(path).read_text()
                if i == 1:
                    lang = re.search(r'Language Matrix.*?\|\s*(\w+)', content)
                    self.add_entry("platform.current_state.services.languages", lang.group(1) if lang else "unknown", 0.8, path)

    def scan_nats(self):
        nats_config = self.root.joinpath("infra/kubernetes/nats-config.yaml")
        if nats_config.exists():
            content = nats_config.read_text()
            streams = re.findall(r'stream: (\w+)', content)
            subjects = re.findall(r'subject: (\w+)', content)
            self.add_entry("platform.runtime_infra.nats.streams", streams, 0.9, "infra/k8s")
            self.add_entry("platform.runtime_infra.nats.subjects", subjects, 0.9, "infra/k8s")

    def scan_redis(self):
        redis_usage = []
        for path in self.root.rglob("*.py"):
            if "redis" in path.read_text():
                if "pubsub" in path.read_text():
                    redis_usage.append("pubsub")
                if "cache" in path.read_text():
                    redis_usage.append("cache")
        self.add_entry("platform.runtime_infra.redis.roles", list(set(redis_usage)), 0.8, "code")

    def scan_dbs(self):
        db_drivers = ["postgres", "neo4j", "minio", "s3"]
        for driver in db_drivers:
            for path in self.root.rglob(f"*{driver}*"):
                self.add_entry(f"platform.runtime_infra.datastores.{driver}", True, 0.7, str(path))

    def scan_ci(self):
        ci_dir = self.root.joinpath(".github/workflows")
        if ci_dir.exists():
            for wf in ci_dir.glob("*.yml"):
                content = wf.read_text()
                if "security" in content:
                    self.add_entry(f"dev.workflow.ci_pipelines.{wf.stem}", "security", 0.8, str(wf))

    def scan_scanners(self):
        scanners = ["nmap", "nuclei", "openvas", "rust-scanner"]
        for scanner in scanners:
            count = len(list(self.root.rglob(f"*{scanner}*")))
            self.add_entry(f"ptaas.domain.scanners.{scanner}", count > 0, 0.7 if count else 0.0, "code")

    def write_outputs(self):
        # Write CSV
        with open(self.root.joinpath("tools/planning/inputs_matrix.csv"), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['key', 'discovered_value', 'confidence', 'source', 'notes'])
            writer.writeheader()
            for row in self.matrix:
                writer.writerow(row)
        # Write JSON
        with open(self.root.joinpath("tools/planning/inputs_matrix.json"), 'w') as f:
            json.dump(self.matrix, f, indent=2)
        # Write redacted values
        with open(self.root.joinpath("tools/planning/redacted.json"), 'w') as f:
            json.dump(self.redacted, f, indent=2)

    def run(self):
        self.scan_repo_shape()
        self.scan_adrs()
        self.scan_nats()
        self.scan_redis()
        self.scan_dbs()
        self.scan_ci()
        self.scan_scanners()
        self.write_outputs()

if __name__ == "__main__":
    scanner = RepoScanner()
    scanner.run()
