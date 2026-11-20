
from pathlib import Path
from collections import Counter
import json

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_cleaned"
OUTPUT_DIR = ROOT / "dataset"
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42
VAL_RATIO = 0.15
EXTENSIONS = {".jpg", ".jpeg", ".png"}

classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Collect cleaned training images and their labels.
images = []
labels = []

for class_name in classes:
    folder = DATA_DIR / "Training" / class_name

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    images.extend(files)
    labels.extend([class_name] * len(files))

# Keep approximately 15% for validation.
train_paths, val_paths, train_labels, val_labels = train_test_split(
    images,
    labels,
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=labels
)

# Collect the original cleaned testing data.
test_paths = []
test_labels = []

for class_name in classes:
    folder = DATA_DIR / "Testing" / class_name

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    test_paths.extend(files)
    test_labels.extend([class_name] * len(files))

# Ensure the three subsets have no overlapping file paths.
assert not (set(train_paths) & set(val_paths))
assert not (set(train_paths) & set(test_paths))
assert not (set(val_paths) & set(test_paths))

# Save portable paths relative to data_cleaned.
def relative_paths(paths):
    return sorted(str(p.relative_to(DATA_DIR)) for p in paths)

split_data = {
    "seed": SEED,
    "validation_ratio": VAL_RATIO,
    "classes": classes,
    "train": relative_paths(train_paths),
    "val": relative_paths(val_paths),
    "test": relative_paths(test_paths)
}

with open(OUTPUT_DIR / "split.json", "w") as f:
    json.dump(split_data, f, indent=2)

train_counts = Counter(train_labels)
val_counts = Counter(val_labels)
test_counts = Counter(test_labels)

print("\nClass distribution:")
print(f"{'Class':<14} {'Train':>7} {'Val':>7} {'Test':>7}")

for name in classes:
    print(
        f"{name:<14} "
        f"{train_counts[name]:>7} "
        f"{val_counts[name]:>7} "
        f"{test_counts[name]:>7}"
    )

print("\nTotal train:", len(train_paths))
print("Total validation:", len(val_paths))
print("Total test:", len(test_paths))

# -------------------------------
# FIGURE: Class distribution
# -------------------------------
x = list(range(len(classes)))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    [i - width for i in x],
    [train_counts[c] for c in classes],
    width,
    label="Training"
)

ax.bar(
    x,
    [val_counts[c] for c in classes],
    width,
    label="Validation"
)

ax.bar(
    [i + width for i in x],
    [test_counts[c] for c in classes],
    width,
    label="Testing"
)

ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Number of images")
ax.set_title("Class Distribution After Duplicate Removal and Splitting")
ax.legend()

fig.tight_layout()

figure_path = FIGURES_DIR / "class_distribution.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSplit saved to:", OUTPUT_DIR / "split.json")
print("Figure saved to:", figure_path)
