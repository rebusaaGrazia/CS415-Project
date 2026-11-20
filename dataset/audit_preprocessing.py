
from pathlib import Path
from collections import Counter
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataloader import (
    BrainTumorDataset,
    DATA_DIR,
    SPLIT_FILE,
    crop_black_margins,
    pad_to_square,
)

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "dataset" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(SPLIT_FILE) as f:
    split_data = json.load(f)

# Check that every file belongs to exactly one split.
train_set = set(split_data["train"])
val_set = set(split_data["val"])
test_set = set(split_data["test"])

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

crop_removed = []
failures = []
examples = []

for split in ("train", "val", "test"):
    dataset = BrainTumorDataset(split)

    for index, relative_path in enumerate(dataset.paths):
        try:
            path = DATA_DIR / relative_path

            with Image.open(path) as img:
                original = img.convert("RGB")

            cropped = crop_black_margins(original)
            squared = pad_to_square(cropped)

            original_area = original.width * original.height
            cropped_area = cropped.width * cropped.height

            removed_percentage = (
                100 * (1 - cropped_area / original_area)
            )

            crop_removed.append(removed_percentage)

            if cropped.width <= 0 or cropped.height <= 0:
                raise ValueError("Invalid cropped dimensions")

            tensor, label = dataset[index]

            if tensor.shape != (3, 224, 224):
                raise ValueError("Unexpected tensor shape")

            if not torch.isfinite(tensor).all():
                raise ValueError("Non-finite tensor values")

            # Record unusually aggressive crops.
            if removed_percentage > 50:
                examples.append((
                    relative_path,
                    removed_percentage
                ))

        except Exception as error:
            failures.append((relative_path, str(error)))

    print(f"{split}: checked {len(dataset)} images")

crop_removed = np.asarray(crop_removed)

print("\nAudit results:")
print("Images checked:", len(crop_removed))
print("Processing failures:", len(failures))
print("Mean cropped area (%):", round(crop_removed.mean(), 2))
print("Median cropped area (%):", round(np.median(crop_removed), 2))
print("Maximum cropped area (%):", round(crop_removed.max(), 2))
print("Images with >50% area removed:", len(examples))

if examples:
    print("\nExamples requiring visual inspection:")
    for path, percentage in sorted(
        examples, key=lambda x: -x[1]
    )[:10]:
        print(f"{path}: {percentage:.1f}% removed")

if failures:
    print("\nProcessing failures:")
    for path, error in failures[:10]:
        print(path, error)

# Generate a histogram for the progress report.
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(crop_removed, bins=30, edgecolor="black")
ax.set_xlabel("Image area removed by cropping (%)")
ax.set_ylabel("Number of MRI images")
ax.set_title("Distribution of Cropped Image Area")

fig.tight_layout()
output = FIGURES_DIR / "cropping_distribution.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nFigure saved to:", output)

assert len(failures) == 0, "Some images failed preprocessing"
assert len(crop_removed) == 7013, "Unexpected number of images"

print("Full-dataset preprocessing audit passed!")
