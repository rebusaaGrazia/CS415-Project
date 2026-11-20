
from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data"
CLEANED = ROOT / "data_cleaned"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

if CLEANED.exists():
    raise FileExistsError(
        "data_cleaned already exists. "
        "Move or rename it before running again."
    )

removed = 0
kept = 0

for split in ("Training", "Testing"):
    groups = defaultdict(list)

    for path in sorted((SOURCE / split).rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups[digest].append(path)

    for paths in groups.values():
        original = paths[0]
        destination = CLEANED / original.relative_to(SOURCE)

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)

        kept += 1
        removed += len(paths) - 1

print("Unique images retained:", kept)
print("Redundant copies excluded:", removed)
print("Cleaned dataset:", CLEANED)
