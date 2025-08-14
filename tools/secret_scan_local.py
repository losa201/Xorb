import os
import re
import json
from pathlib import Path

def scan_secrets():
    patterns = {
        'github_token': r'ghp_[A-Za-z0-9]{36}',
        'aws_akia': r'AKIA[0-9A-Z]{16}',
        'jwt': r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
        'credential': r'(?:password|secret|token)=[^\s]{8,}'
    }

    findings = []
    for path in Path('.').rglob('*'):
        if any(p in str(path) for p in ['reports/security/', '.git', 'node_modules']):
            continue

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            for name, pattern in patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    findings.append({
                        'file': str(path),
                        'type': name,
                        'match': match.group(),
                        'line': content[:match.start()].count('\n') + 1
                    })
        except Exception:
            continue

    return findings

if __name__ == '__main__':
    results = scan_secrets()
    print(json.dumps(results, indent=2))
    exit(1 if results else 0)
