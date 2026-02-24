#!/bin/sh
# Script to move large/unnecessary folders to `archive/` to keep repo small
set -e

ARCHIVE_DIR=archive_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ARCHIVE_DIR"

# List of candidate folders to archive (safe defaults)
for d in htmlcov docs notebooks data dev .pytest_cache; do
  if [ -d "$d" ]; then
    echo "Archiving $d -> $ARCHIVE_DIR/$d"
    mv "$d" "$ARCHIVE_DIR/"
  fi
done

echo "Archived into $ARCHIVE_DIR. Inspect before committing."
