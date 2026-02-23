#!/usr/bin/env python3
"""
Lightweight staged-file secret checker for git pre-commit.
Blocks commits when likely secrets are detected.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)postgres(?:ql)?://[^\s{}]+"),
    re.compile(r"(?i)mongodb(?:\+srv)?://[^\s{}]+"),
    re.compile(r"(?i)mysql://[^\s{}]+"),
    re.compile(r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----"),
]

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".pkl",
    ".pyc",
}

SKIP_FILES = {
    ".env.example",
    "scripts/precommit_secret_check.py",
}


def get_staged_files() -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("Unable to list staged files; skipping secret check.", file=sys.stderr)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_staged_content(path: str) -> str:
    result = subprocess.run(
        ["git", "show", f":{path}"],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def main() -> int:
    findings: list[str] = []
    for file_path in get_staged_files():
        if file_path.replace("\\", "/") in SKIP_FILES:
            continue
        p = Path(file_path)
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        content = get_staged_content(file_path)
        if not content:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append(f"{file_path}: matched `{pattern.pattern}`")
                break

    if findings:
        print("Commit blocked: possible secrets detected in staged content.")
        for finding in findings:
            print(f"- {finding}")
        print("Move credentials to .env and commit only .env.example.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
