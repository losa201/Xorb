# GitHub Security Push Protection

This document explains GitHub's push protection, why our push was rejected, and how to remediate it.

## Quick Fix Now

If your push was rejected due to a secret in `reports/security/safety_report.json` or elsewhere:

1.  **Scrub History:**
    Run the provided script to remove the secret from Git history. This rewrites your commits.
    ```bash
    bash tools/secrets/remediate_git_history.sh
    ```
2.  **Re-push:**
    After scrubbing, force-push your branch and tags.
    ```bash
    git push --force-with-lease origin <your_branch_name>
    git push --force --tags
    ```
    ⚠️ Ensure `<your_branch_name>` is not a protected branch like `main`.

## Ongoing Prevention

Prevent future issues with a multi-layered approach:

1.  **Pre-commit Hooks:**
    Install hooks to scan for secrets before committing.
    ```bash
    make precommit-install
    ```
2.  **CI Scans:**
    Every push and pull request is automatically scanned using `gitleaks` in GitHub Actions.
3.  **Allowlist:**
    Placeholders and test data are allowlisted in `.gitleaks.toml`. Use these patterns, never real secrets.

## Safe Patterns (Placeholders Only)

Use these placeholder values for examples and tests:

- **GitHub PAT:** `ghp_placeholderPAT1234567890123456789012345`
- **JWT Secret:** `your_jwt_secret_32_chars_min`
- **AWS Key ID:** `AKIAIOSFODNN7EXAMPLE`
- **Slack Token:** `xoxb-placeholder-slack-token-123456789012`
- **Generic Password:** `test_password`

Never commit real secrets. Rotate any accidentally committed secrets immediately.
