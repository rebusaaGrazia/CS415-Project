
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataloader import (
    DATA_DIR,
    SPLIT_FILE,
    crop_black_margins,
)

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "dataset" / "figures"

with open(SPLIT_FILE) as f:
    splits = json.load(f)

results = []

# Find the most aggressively cropped images.
for split in ("train", "val", "test"):
    for relative_path in splits[split]:
        path = DATA_DIR / relative_path

        with Image.open(path) as img:
            original = img.convert("RGB")

        cropped = crop_black_margins(original)

        removed = 100 * (
            1 - (
                cropped.width * cropped.height
            ) / (
                original.width * original.height
            )
        )

        results.append((removed, relative_path))

results.sort(reverse=True)

top_examples = results[:12]

fig, axes = plt.subplots(12, 2, figsize=(8, 28))

for row, (removed, relative_path) in enumerate(top_examples):
    with Image.open(DATA_DIR / relative_path) as img:
        original = img.convert("RGB")

    cropped = crop_black_margins(original)

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(
        f"Original — {Path(relative_path).name}",
        fontsize=9
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(cropped)
    axes[row, 1].set_title(
        f"Cropped — {removed:.1f}% area removed",
        fontsize=9
    )
    axes[row, 1].axis("off")

fig.suptitle(
    "Visual Inspection of Aggressive MRI Cropping",
    fontsize=14
)
fig.tight_layout()

output = FIGURES_DIR / "extreme_cropping_examples.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figure saved to:", output)
