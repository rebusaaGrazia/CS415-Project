
from pathlib import Path
import json
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_cleaned"
FIGURES_DIR = ROOT / "dataset" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "dataset" / "split.json") as f:
    splits = json.load(f)

classes = ["glioma", "meningioma", "notumor", "pituitary"]


def crop_black_margins(image, threshold=10):
    gray = np.asarray(image.convert("L"))
    mask = gray > threshold

    if not mask.any():
        return image.copy()

    rows, cols = np.where(mask)

    return image.crop((
        int(cols.min()),
        int(rows.min()),
        int(cols.max()) + 1,
        int(rows.max()) + 1
    ))



def pad_to_square(image):
    width, height = image.size
    target_size = max(width, height)

    left = (target_size - width) // 2
    top = (target_size - height) // 2

    right = target_size - width - left
    bottom = target_size - height - top

    return ImageOps.expand(
        image,
        border=(left, top, right, bottom),
        fill=(0, 0, 0)
    )



fig, axes = plt.subplots(4, 4, figsize=(13, 12))

for row, class_name in enumerate(classes):
    candidates = [
        p for p in splits["train"]
        if Path(p).parent.name == class_name
    ]

    image_path = DATA_DIR / sorted(candidates)[0]

    with Image.open(image_path) as img:
        original = img.convert("RGB")

    cropped = crop_black_margins(original)
    squared = pad_to_square(cropped)

    resized = squared.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    versions = [original, cropped, squared, resized]
    titles = [
        "Original",
        "Cropped",
        "Square padding",
        "Resized 224×224"
    ]

    for col, (image, title) in enumerate(zip(versions, titles)):
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(f"{class_name} — {title}", fontsize=10)
        ax.axis("off")

    print(
        f"{class_name}: "
        f"{original.size} -> {cropped.size} "
        f"-> {squared.size} -> {resized.size}"
    )

fig.suptitle("MRI Preprocessing Examples", fontsize=14)
fig.tight_layout()

output = FIGURES_DIR / "preprocessing_examples.png"
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure saved to:", output)
