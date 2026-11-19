"""
Step 5: PyTorch Dataset + DataLoaders. Everyone (baseline, SE, CBAM) imports from here:

    from dataset import get_loaders
    train_loader, val_loader, test_loader = get_loaders()

Run `python dataset.py` to execute the checkpoints.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config as C


class BrainMRIDataset(Dataset):
    """Reads one of splits/{train,val,test}.csv and returns (tensor[3,H,W], label_idx)."""

    def __init__(self, csv_path, transform=None, root=C.PROC_DIR):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.filepath).convert("RGB")  # 1 gray channel -> 3 channels
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row.label_idx)


def get_transforms(train, augment=C.AUGMENT):
    """Images are already cropped+resized offline. Augmentation (if any) is TRAIN ONLY."""
    t = []
    if train and augment:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    t += [transforms.ToTensor(), transforms.Normalize(C.MEAN, C.STD)]
    return transforms.Compose(t)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def get_loaders(batch_size=C.BATCH_SIZE, num_workers=2, augment=C.AUGMENT, seed=C.SEED):
    g = torch.Generator()
    g.manual_seed(seed)  # same seed -> same shuffling order
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin, worker_init_fn=_seed_worker)

    train_ds = BrainMRIDataset(C.SPLITS_DIR / "train.csv", get_transforms(True, augment))
    val_ds = BrainMRIDataset(C.SPLITS_DIR / "val.csv", get_transforms(False))
    test_ds = BrainMRIDataset(C.SPLITS_DIR / "test.csv", get_transforms(False))

    train_loader = DataLoader(train_ds, shuffle=True, generator=g, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------- checkpoints
def _check(cond, msg, level="fail"):
    tag = "OK" if cond else ("FAIL" if level == "fail" else "WARN")
    print(f"  [{tag}] {msg}")
    return cond


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("== STEP 5: dataset / dataloader checkpoints ==")
    ok = True
    tr, va, te = get_loaders(num_workers=0)
    xb, yb = next(iter(tr))

    ok &= _check(xb.shape == (min(C.BATCH_SIZE, len(tr.dataset)), 3, C.IMG_SIZE, C.IMG_SIZE),
                 f"batch shape is (B, 3, {C.IMG_SIZE}, {C.IMG_SIZE}): {tuple(xb.shape)}")
    ok &= _check(xb.dtype == torch.float32, f"dtype float32: {xb.dtype}")
    ok &= _check(yb.dtype == torch.int64, f"labels are int64: {yb.dtype}")
    ok &= _check(int(yb.min()) >= 0 and int(yb.max()) < len(C.CLASSES), f"labels in 0..{len(C.CLASSES) - 1}: {yb.unique().tolist()}")
    print(f"  batch mean {xb.mean():.2f}, std {xb.std():.2f}, min {xb.min():.2f}, max {xb.max():.2f}")
    ok &= _check(-3.0 < float(xb.min()) and float(xb.max()) < 3.0,
                 "values are normalised (roughly within -2.2 .. 2.7), not 0..255")
    _check(yb.unique().numel() >= 3, f"first training batch mixes classes ({yb.unique().numel()} of 4): shuffling works",
           level="warn")

    # reproducibility: two loaders with the same seed give the same first batch
    _, yb2 = next(iter(get_loaders(num_workers=0)[0]))
    ok &= _check(torch.equal(yb, yb2), "same seed -> same shuffled first batch")
    # val/test are never shuffled
    y_a = next(iter(va))[1]
    y_b = next(iter(va))[1]
    ok &= _check(torch.equal(y_a, y_b), "validation order is fixed (shuffle=False)")
    # sizes
    for name, ld in (("train", tr), ("val", va), ("test", te)):
        n = sum(len(y) for _, y in ld)
        ok &= _check(n == len(ld.dataset), f"{name}: loader yields all {len(ld.dataset)} images (got {n})")
    # label <-> folder consistency for the first 20 rows
    df = tr.dataset.df.head(20)
    ok &= _check(all(Path(r.filepath).parts[1] == C.CLASSES[r.label_idx] for r in df.itertuples()),
                 "label_idx matches the class folder of the file")

    # visual check
    C.REPORTS_DIR.mkdir(exist_ok=True)
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    for ax, img, y in zip(axes.flat, xb, yb):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(C.CLASSES[int(y)], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(C.REPORTS_DIR / "batch_check.png", dpi=80)
    print(f"  -> OPEN {C.REPORTS_DIR / 'batch_check.png'}: do titles match what you see (denormalised batch)?")
    print("\nAll hard checkpoints passed." if ok else "\nSome checkpoints FAILED.")
    raise SystemExit(0 if ok else 1)