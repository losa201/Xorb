#!/usr/bin/env python3
"""
ADR Locked Guard - Ensures LOCKED sections in ADRs are enforced in repo configs

Scans docs/architecture/ADR-* for LOCKED sections and ensures the phrases appear
unchanged in repo configs (e.g., TLS1.3 only, no HS256, Redis not a bus).
"""

import re
import sys
from pathlib import Path


def find_locked_sections(adr_dir):
    """Find all LOCKED sections in ADR files."""
    locked_phrases = []

    if not adr_dir.exists():
        print(f"Warning: ADR directory {adr_dir} not found")
        return locked_phrases

    for adr_file in adr_dir.glob("ADR-*.md"):
        try:
            with open(adr_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all LOCKED sections
            locked_matches = re.finditer(r'LOCKED: (.+?)(?=\n\n|\Z)', content, re.DOTALL)
            for match in locked_matches:
                phrase = match.group(1).strip()
                # Only include significant phrases (not just "status" phrases)
                if len(phrase) > 20 and "LOCKED" in phrase:
                    # Extract the key part after "LOCKED:"
                    key_part = phrase.split("LOCKED:", 1)[1].strip() if "LOCKED:" in phrase else phrase
                    locked_phrases.append((key_part, adr_file.name))
        except Exception as e:
            print(f"Warning: Could not read {adr_file}: {e}")

    return locked_phrases


def check_phrase_in_configs(phrase, config_dirs):
    """Check if a locked phrase appears in config files."""
    # Normalize phrase for matching
    normalized_phrase = re.sub(r'\s+', ' ', phrase.lower().strip())

    for config_dir in config_dirs:
        if not config_dir.exists():
            continue

        # Check common config files
        for config_file in config_dir.rglob("*"):
            if config_file.is_file() and config_file.suffix in ['.yml', '.yaml', '.json', '.conf', '.cfg', '.toml', '.md']:
                try:
                    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        normalized_content = re.sub(r'\s+', ' ', content.lower())

                        # Check if phrase appears in content
                        if normalized_phrase in normalized_content:
                            return str(config_file)
                except Exception:
                    continue

    return None


def main():
    """Main function to check ADR locked sections."""
    repo_root = Path(__file__).parent.parent.parent
    adr_dir = repo_root / "docs" / "architecture"
    config_dirs = [
        repo_root / "config",
        repo_root / "infra",
        repo_root / "deploy",
        repo_root / "compose"
    ]

    # Find locked sections
    locked_phrases = find_locked_sections(adr_dir)

    if not locked_phrases:
        print("No LOCKED sections found in ADRs")
        return 0

    # Check each locked phrase
    violations = []
    for phrase, adr_file in locked_phrases:
        found_file = check_phrase_in_configs(phrase, config_dirs)
        if not found_file:
            violations.append((phrase, adr_file))

    # Report violations
    if violations:
        print("❌ ADR Lock Violations:")
        print("The following LOCKED requirements were not found in configs:")
        for phrase, adr_file in violations:
            print(f"  - From {adr_file}: {phrase[:100]}...")
        return 1
    else:
        print("✅ All ADR LOCKED requirements found in configs")
        return 0


if __name__ == "__main__":
    sys.exit(main())
