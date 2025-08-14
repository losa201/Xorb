#!/usr/bin/env python3
"""
Detect Private Key - Pre-commit hook to detect private keys in staged files
"""

import sys
import re

PRIVATE_KEY_PATTERNS = [
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"-----BEGIN [A-Z ]*PRIVATE KEY BLOCK-----",
    r"ssh-rsa [A-Za-z0-9+/]{100,}",
    r"ssh-dss [A-Za-z0-9+/]{100,}",
    r"ssh-ed25519 [A-Za-z0-9+/]{100,}",
    r"ecdsa-sha2-[A-Za-z0-9+/]{100,}",
]

def main():
    """Main function to check for private keys in files."""
    if len(sys.argv) < 2:
        print("No files provided")
        return 1

    violations = []
    for filename in sys.argv[1:]:
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for pattern in PRIVATE_KEY_PATTERNS:
                if re.search(pattern, content):
                    violations.append(filename)
                    break
        except Exception:
            # Skip files that can't be read
            continue

    if violations:
        print("❌ Private key detected in:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
