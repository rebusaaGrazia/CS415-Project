
from pathlib import Path

# Update this path if your downloaded dataset
# is located elsewhere.
DATA_DIR = Path(__file__).resolve().parent.parent / "data_cleaned"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

print("Dataset inspection\n")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}"
    )

for split in ["Training", "Testing"]:
    split_dir = DATA_DIR / split

    if not split_dir.exists():
        print(f"Missing: {split_dir}")
        continue

    print(f"{split}:")

    total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"  {class_dir.name}: {len(images)}")
        total += len(images)

    print(f"  Total: {total}\n")
