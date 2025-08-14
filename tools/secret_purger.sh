#!/bin/bash
set -e

# Dry-run mode by default (no push)
DRY_RUN=1
if [ "$1" = "PUSH=1" ]; then
  DRY_RUN=0
fi

# Create backup branch if it doesn't exist
if ! git show-ref -q --verify refs/heads/backup/pre-purge; then
  git checkout -b backup/pre-purge
fi

# Check for git filter-repo availability
if command -v git-filter-repo >/dev/null 2>&1; then
  echo "Using git filter-repo"
  git filter-repo --path-glob 'reports/security/*.json' --invert-paths
else
  echo "Using git filter-branch"
  git filter-branch --tree-filter 'rm -f reports/security/*.json || true' HEAD
fi

# Push backup if requested
if [ $DRY_RUN -eq 0 ]; then
  git push origin backup/pre-purge
fi

exit 0
