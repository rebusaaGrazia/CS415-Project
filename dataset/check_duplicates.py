
from pathlib import Path
from collections import defaultdict
import hashlib

DATA_DIR = Path(__file__).resolve().parent.parent / "data_cleaned"
EXTENSIONS = {".jpg", ".jpeg", ".png"}

hashes = defaultdict(list)

for split in ("Training", "Testing"):
    for path in (DATA_DIR / split).rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[file_hash].append(path)

duplicate_groups = [
    paths for paths in hashes.values()
    if len(paths) > 1
]

cross_split_groups = [
    paths for paths in duplicate_groups
    if any("Training" in p.parts for p in paths)
    and any("Testing" in p.parts for p in paths)
]

print("Total unique file hashes:", len(hashes))
print("Duplicate groups:", len(duplicate_groups))
print("Cross-split duplicate groups:", len(cross_split_groups))

for group in cross_split_groups[:5]:
    print("\nCross-split duplicate:")
    for path in group:
        print(" ", path.relative_to(DATA_DIR))



training_duplicates = 0
testing_duplicates = 0
cross_class_duplicates = 0

for paths in duplicate_groups:
    splits = {p.relative_to(DATA_DIR).parts[0] for p in paths}
    classes = {p.relative_to(DATA_DIR).parts[1] for p in paths}

    if splits == {"Training"}:
        training_duplicates += 1
    elif splits == {"Testing"}:
        testing_duplicates += 1

    if len(classes) > 1:
        cross_class_duplicates += 1

print("\nDuplicate group breakdown:")
print("Within Training:", training_duplicates)
print("Within Testing:", testing_duplicates)
print("Across different classes:", cross_class_duplicates)
