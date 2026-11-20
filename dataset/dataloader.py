
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_cleaned"
SPLIT_FILE = ROOT / "dataset" / "split.json"

# Match the class order saved in split.json.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    size = max(width, height)

    left = (size - width) // 2
    top = (size - height) // 2

    return ImageOps.expand(
        image,
        border=(
            left,
            top,
            size - width - left,
            size - height - top
        ),
        fill=(0, 0, 0)
    )


def get_transform(training=False, augment=False):
    operations = [
        transforms.Lambda(crop_black_margins),
        transforms.Lambda(pad_to_square),
        transforms.Resize((224, 224)),
    ]

    if training and augment:
        operations.append(
            transforms.RandomAffine(
                degrees=8,
                translate=(0.03, 0.03),
                fill=0
            )
        )

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return transforms.Compose(operations)


class BrainTumorDataset(Dataset):
    def __init__(self, split, augment=False):
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split")

        with open(SPLIT_FILE) as f:
            split_data = json.load(f)

        self.paths = split_data[split]
        self.transform = get_transform(
            training=(split == "train"),
            augment=augment
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        relative_path = self.paths[index]
        path = DATA_DIR / relative_path

        class_name = Path(relative_path).parent.name
        label = CLASS_TO_IDX[class_name]

        with Image.open(path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        return image, label


def create_dataloaders(batch_size=16, augment=False):
    train_dataset = BrainTumorDataset(
        "train", augment=augment
    )
    val_dataset = BrainTumorDataset("val")
    test_dataset = BrainTumorDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
