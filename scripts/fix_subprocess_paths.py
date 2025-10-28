#!/usr/bin/env python3
"""
Audit and patch all subprocess.run([sys.executable, ...]) calls so they use sys.executable instead.
Run once from project root:  python scripts/fix_subprocess_paths.py
"""

import re, sys
from pathlib import Path

scripts = Path("scripts")
for file in scripts.glob("*.py"):
    text = file.read_text(encoding="utf-8")
    new_text = re.sub(
        r'subprocess\.run\(\s*\[\s*["\']python["\']',
        'subprocess.run([sys.executable',
        text,
    )
    if text != new_text:
        print(f"🩹 Patched subprocess calls in {file.name}")
        if "import sys" not in new_text:
            # insert import sys safely after other imports
            new_text = re.sub(r"(^import .*$)", r"\1\nimport sys", new_text, count=1, flags=re.MULTILINE)
        file.write_text(new_text, encoding="utf-8")
print("✅ Audit complete — all scripts now use sys.executable for subprocess calls.")
