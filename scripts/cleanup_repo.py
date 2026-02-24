#!/usr/bin/env python3
"""Cross-platform cleanup: move large dirs into an archive folder for review."""
import os
import shutil
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)

ARCHIVE_DIR = f"archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

def main():
    archive = os.path.join(ROOT, ARCHIVE_DIR)
    os.makedirs(archive, exist_ok=True)
    candidates = ["htmlcov", "docs", "notebooks", "data", "dev", ".pytest_cache"]
    moved = []
    for d in candidates:
        path = os.path.join(ROOT, d)
        if os.path.exists(path):
            target = os.path.join(archive, d)
            print(f"Archiving {d} -> {target}")
            shutil.move(path, target)
            moved.append((d, target))
    if not moved:
        print("No candidate directories found to archive.")
    else:
        print(f"Archived into {archive}. Inspect before committing.")

if __name__ == '__main__':
    main()
