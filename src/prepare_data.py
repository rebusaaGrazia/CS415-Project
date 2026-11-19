"""
Data preparation for the Brain Tumor MRI project (Nickparvar V2).

Run the steps in order (or `all`):
    python prepare_data.py download     # step 1
    python prepare_data.py audit        # step 2
    python prepare_data.py preprocess   # step 3
    python prepare_data.py splits       # step 4   (add --force to rewrite existing CSVs)
    python prepare_data.py all

Every step ends with CHECKPOINTS printed as [OK] / [WARN] / [FAIL].
[FAIL] stops the script (exit code 1). [WARN] means: read it and decide / tell the team.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import config as C

# ----------------------------------------------------------------------------- checkpoint helpers
FAILS = []


def check(cond, msg, level="fail"):
    """Print a checkpoint. level='fail' stops the step at the end; level='warn' only warns."""
    tag = "OK" if cond else ("FAIL" if level == "fail" else "WARN")
    print(f"  [{tag}] {msg}")
    if not cond and level == "fail":
        FAILS.append(msg)
    return cond


def finish(step):
    if FAILS:
        print(f"\n{step}: {len(FAILS)} checkpoint(s) FAILED:")
        for m in FAILS:
            print("   -", m)
        sys.exit(1)
    print(f"\n{step}: no failed checkpoints (read any [WARN] lines above)\n")


# ----------------------------------------------------------------------------- shared helpers
def find_dataset_root(base):
    """Return the folder that contains both 'Training' and 'Testing' (any capitalisation), or None."""
    base = Path(base)
    for dirpath, dirnames, _ in os.walk(base):
        low = {d.lower(): d for d in dirnames}
        if "training" in low and "testing" in low:
            return Path(dirpath)
    return None


def get_split_dir(root, split):
    for d in Path(root).iterdir():
        if d.is_dir() and d.name.lower() == split.lower():
            return d
    raise FileNotFoundError(f"{split} folder not found in {root}")


def list_images(root):
    """Yield (split, label, path) for every image in root/{Training,Testing}/<class>/."""
    for split in ("Training", "Testing"):
        sdir = get_split_dir(root, split)
        for cls in C.CLASSES:
            cdir = next((d for d in sdir.iterdir() if d.is_dir() and d.name.lower() == cls), None)
            if cdir is None:
                continue
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in C.IMG_EXTS:
                    yield split, cls, f


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


# ============================================================================= STEP 1: download
def step_download():
    print("== STEP 1: download ==")
    root = find_dataset_root(C.RAW_DIR) if C.RAW_DIR.exists() else None
    if root is None:
        import kagglehub  # pip install kagglehub  (needs Kaggle login/token on first use)
        src = Path(kagglehub.dataset_download(C.KAGGLE_ID))
        src_root = find_dataset_root(src)
        check(src_root is not None, f"Training/ and Testing/ found inside the download ({src})")
        if src_root is None:
            finish("download")
        C.RAW_DIR.mkdir(parents=True, exist_ok=True)
        for sub in src_root.iterdir():
            if sub.is_dir():
                shutil.copytree(sub, C.RAW_DIR / sub.name, dirs_exist_ok=True)
        root = find_dataset_root(C.RAW_DIR)
    else:
        print(f"  raw data already in {C.RAW_DIR}, skipping download")

    print("-- checkpoints --")
    check(root is not None, f"dataset root found: {root}")
    if root is not None:
        for split in ("Training", "Testing"):
            sdir = get_split_dir(root, split)
            names = {d.name.lower() for d in sdir.iterdir() if d.is_dir()}
            check(names == set(C.CLASSES), f"{split}/ has exactly the 4 class folders {sorted(names)}")
    finish("download")


# ============================================================================= STEP 2: audit
def step_audit():
    print("== STEP 2: audit ==")
    root = find_dataset_root(C.RAW_DIR)
    check(root is not None, "raw data found (run step 1 first)")
    if root is None:
        finish("audit")

    rows = []
    for split, label, path in list_images(root):
        try:
            with Image.open(path) as im:
                im.load()
                w, h, mode = im.size[0], im.size[1], im.mode
            ok = True
        except Exception:
            w = h = 0
            mode, ok = "ERR", False
        rows.append(dict(split=split, label=label, path=str(path), key=f"{split}/{label}/{path.stem}",
                         w=w, h=h, mode=mode, ok=ok, md5=md5(path)))
    df = pd.DataFrame(rows)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.REPORTS_DIR / "audit.csv", index=False)

    counts = df.pivot_table(index="label", columns="split", values="path", aggfunc="count", margins=True)
    print("\nImages per class and split:")
    print(counts.to_string(), "\n")
    print("Image modes:", df["mode"].value_counts().to_dict())
    print("Distinct image sizes:", df.groupby(["w", "h"]).ngroups,
          "| most common:", df.groupby(["w", "h"]).size().sort_values(ascending=False).head(3).to_dict(), "\n")

    print("-- checkpoints --")
    check(len(df) > 0, f"found {len(df)} images")
    check(set(df.label.unique()) == set(C.CLASSES), "all 4 classes present")
    bad = df[~df.ok]
    check(len(bad) == 0, f"no corrupted/unreadable files ({len(bad)} found)")
    if len(bad):
        print(bad.path.to_string())

    for split in ("Training", "Testing"):
        per = df[df.split == split].groupby("label").size()
        exp = C.EXPECTED_PER_CLASS[split]
        check((per == exp).all(),
              f"{split}: {exp} images per class as written in the proposal (found {per.to_dict()}). "
              f"IF THIS DIFFERS, TELL THE TEAM: the proposal numbers must be updated.", level="warn")
        check(per.min() / per.max() >= 0.8,
              f"{split}: classes roughly balanced (min/max = {per.min() / per.max():.2f})", level="warn")

    # byte-identical duplicates (does NOT catch near-duplicates, only exact copies)
    tr, te = df[df.split == "Training"], df[df.split == "Testing"]
    leak = set(tr.md5) & set(te.md5)
    check(len(leak) == 0, f"no identical image appears in both Training and Testing ({len(leak)} found; remove them if > 0)",
          level="warn")
    check(tr.md5.duplicated().sum() == 0, f"no exact duplicates inside Training ({tr.md5.duplicated().sum()} found)",
          level="warn")
    check(te.md5.duplicated().sum() == 0, f"no exact duplicates inside Testing ({te.md5.duplicated().sum()} found)",
          level="warn")

    # visual check: look at this image yourself!
    fig, axes = plt.subplots(len(C.CLASSES), 6, figsize=(14, 9))
    rng = np.random.RandomState(0)
    for r, cls in enumerate(C.CLASSES):
        sub = tr[tr.label == cls]
        for c, p in enumerate(rng.choice(sub.path.values, 6, replace=False)):
            axes[r, c].imshow(read_rgb(p))
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(cls, loc="left")
    plt.tight_layout()
    plt.savefig(C.REPORTS_DIR / "audit_samples.png", dpi=80)
    plt.close()
    print(f"  -> OPEN {C.REPORTS_DIR / 'audit_samples.png'} and look: do the images match their folder names?")
    print(f"  -> full table saved to {C.REPORTS_DIR / 'audit.csv'}")
    finish("audit")


# ============================================================================= STEP 3: preprocess
def crop_brain(rgb):
    """Crop the black margins. Returns (cropped, fraction_of_image_kept, success)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = (gray > C.CROP_THRESH).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb, 1.0, False
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    frac = (w * h) / (rgb.shape[0] * rgb.shape[1])
    if frac < C.MIN_CROP_AREA_FRAC:
        return rgb, 1.0, False
    return rgb[y:y + h, x:x + w], frac, True


def resize_square(img):
    if C.RESIZE_MODE == "pad":
        h, w = img.shape[:2]
        s = max(h, w)
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        top, left = (s - h) // 2, (s - w) // 2
        canvas[top:top + h, left:left + w] = img
        img = canvas
    interp = cv2.INTER_AREA if img.shape[0] > C.IMG_SIZE else cv2.INTER_CUBIC
    return cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=interp)


def step_preprocess():
    print("== STEP 3: preprocess (crop -> resize -> save PNG) ==")
    root = find_dataset_root(C.RAW_DIR)
    check(root is not None, "raw data found (run step 1 first)")
    if root is None:
        finish("preprocess")

    records, seen = [], set()
    for split, label, path in list_images(root):
        rgb = read_rgb(path)
        cropped, frac, ok = crop_brain(rgb)
        out = resize_square(cropped)
        out_path = C.PROC_DIR / split / label / f"{path.stem}.png"
        if out_path in seen:
            raise RuntimeError(f"two raw images map to the same output file: {out_path}")
        seen.add(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(out_path)  # PNG = lossless
        records.append(dict(split=split, label=label, raw=str(path), out=str(out_path), frac=frac, ok=ok))
    rec = pd.DataFrame(records)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rec.to_csv(C.REPORTS_DIR / "preprocess_log.csv", index=False)

    print("\nFraction of the original image kept by the crop:")
    print(rec.frac.describe().round(3).to_string(), "\n")

    print("-- checkpoints --")
    n_raw = len(rec)
    n_proc = sum(1 for _ in C.PROC_DIR.rglob("*.png"))
    check(n_proc == n_raw, f"processed file count ({n_proc}) == raw image count ({n_raw})")
    for split in ("Training", "Testing"):
        for cls in C.CLASSES:
            a = (rec[(rec.split == split) & (rec.label == cls)]).shape[0]
            b = len(list((C.PROC_DIR / split / cls).glob("*.png")))
            check(a == b, f"{split}/{cls}: {b} processed == {a} raw")

    bad_shape = 0
    for p in C.PROC_DIR.rglob("*.png"):
        with Image.open(p) as im:
            if im.size != (C.IMG_SIZE, C.IMG_SIZE) or im.mode != "RGB":
                bad_shape += 1
    check(bad_shape == 0, f"every processed image is {C.IMG_SIZE}x{C.IMG_SIZE} RGB ({bad_shape} wrong)")

    n_fail = int((~rec.ok).sum())
    check(n_fail <= 0.01 * n_raw, f"crop fell back to the original image for {n_fail} images (<=1% expected)",
          level="warn")
    n_aggr = int((rec.frac < 0.30).sum())
    check(n_aggr == 0, f"{n_aggr} images lost more than 70% of their area in the crop (look at them in the figure)",
          level="warn")

    # visual check: 4 random + the 4 most aggressively cropped images
    rng = np.random.RandomState(1)
    pick = list(rng.choice(len(rec), 4, replace=False)) + list(rec.frac.nsmallest(4).index)
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    for i, idx in enumerate(pick):
        r = rec.loc[idx]
        axes[0, i].imshow(read_rgb(r.raw))
        axes[0, i].set_title(f"{r.label}\nkept {r.frac:.0%}", fontsize=8)
        axes[1, i].imshow(read_rgb(r.out))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].text(-0.05, 0.5, "before", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.05, 0.5, "after", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    plt.tight_layout()
    plt.savefig(C.REPORTS_DIR / "preprocess_check.png", dpi=80)
    plt.close()
    print(f"  -> OPEN {C.REPORTS_DIR / 'preprocess_check.png'}: is the whole brain still visible, with no black "
          f"border and nothing cut off? (first 4 columns random, last 4 = worst crops)")
    finish("preprocess")


# ============================================================================= STEP 4: splits

def build_tables():    #Duplicate-aware train/validation/test split.
    rows = []

    # Collect all processed images and calculate their hashes
    for folder in ("Training", "Testing"):
        for cls in C.CLASSES:
            class_dir = C.PROC_DIR / folder / cls

            for f in sorted(class_dir.glob("*.png")):
                rows.append({
                    "filepath": f.relative_to(C.PROC_DIR).as_posix(),
                    "label": cls,
                    "label_idx": C.CLASS_TO_IDX[cls],
                    "folder": folder,
                    "md5": md5(f)
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No processed images found.")

    # Check whether identical images have different labels
    label_counts = df.groupby("md5")["label"].nunique()

    conflicts = label_counts[label_counts > 1]

    if len(conflicts):
        raise RuntimeError(
            f"{len(conflicts)} identical images have conflicting labels. "
            "Check the dataset labels before splitting."
        )

    # Never allow an identical image in original Training and Testing
    training_hashes = set(
        df.loc[df.folder == "Training", "md5"]
    )

    testing_hashes = set(
        df.loc[df.folder == "Testing", "md5"]
    )

    overlap = training_hashes & testing_hashes

    if overlap:
        raise RuntimeError(
            f"{len(overlap)} identical images occur in both "
            "original Training and Testing folders. "
            "Resolve this leakage before splitting."
        )

    trainval = df[df.folder == "Training"].copy()
    test_df = df[df.folder == "Testing"].copy()

    val_hashes = set()

    # Split unique image groups separately for each class
    for cls in C.CLASSES:

        class_df = trainval[trainval.label == cls]

        unique_hashes = sorted(class_df.md5.unique())

        if len(unique_hashes) < 2:
            raise RuntimeError(
                f"Class '{cls}' has fewer than 2 unique image groups. "
                "Cannot create separate training and validation sets."
            )

        # Split groups, NOT individual images
        _, class_val_hashes = train_test_split(
            unique_hashes,
            test_size=C.VAL_FRACTION,
            random_state=C.SEED,
            shuffle=True
        )

        val_hashes.update(class_val_hashes)

    # All copies of an image stay together
    val_df = trainval[trainval.md5.isin(val_hashes)].copy()

    train_df = trainval[
        ~trainval.md5.isin(val_hashes)
    ].copy()

    # Format output tables
    def fin(d, name):
        d = d.sort_values("filepath").reset_index(drop=True)

        d.insert(len(d.columns), "split", name)

        return d.drop(columns=["folder", "md5"])

    return (
        fin(train_df, "train"),
        fin(val_df, "val"),
        fin(test_df, "test")
    )


def step_splits(force=False):
    print("== STEP 4: splits ==")
    n_png = sum(1 for _ in C.PROC_DIR.rglob("*.png")) if C.PROC_DIR.exists() else 0
    check(n_png > 0, "processed images found (run step 3 first)")
    if n_png == 0:
        finish("splits")

    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {n: C.SPLITS_DIR / f"{n}.csv" for n in ("train", "val", "test")}
    new = dict(zip(("train", "val", "test"), build_tables()))

    if all(p.exists() for p in paths.values()) and not force:
        print("  split CSVs already exist -> NOT rewriting them (frozen). Use --force to overwrite.")
        tables = {n: pd.read_csv(p) for n, p in paths.items()}
    else:
        for n, d in new.items():
            d.to_csv(paths[n], index=False)
        tables = new
        print(f"  wrote {', '.join(p.name for p in paths.values())} to {C.SPLITS_DIR}")

    train, val, test = tables["train"], tables["val"], tables["test"]
    print("\nImages per class and split:")
    summary = pd.concat({n: t.groupby("label").size() for n, t in tables.items()}, axis=1)
    summary.loc["TOTAL"] = summary.sum()
    print(summary.to_string(), "\n")

    print("-- checkpoints --")
    # 1. no overlap
    s = {n: set(t.filepath) for n, t in tables.items()}
    check(not (s["train"] & s["val"]), "train and val share no file")
    check(not (s["train"] & s["test"]), "train and test share no file")
    check(not (s["val"] & s["test"]), "val and test share no file")
    # 2. sizes
    n_training = sum(1 for _ in (C.PROC_DIR / "Training").rglob("*.png"))
    n_testing = sum(1 for _ in (C.PROC_DIR / "Testing").rglob("*.png"))
    check(len(train) + len(val) == n_training, f"train ({len(train)}) + val ({len(val)}) == Training folder ({n_training})")
    check(len(test) == n_testing, f"test ({len(test)}) == Testing folder ({n_testing})")
    check(abs(len(val) - C.VAL_FRACTION * n_training) <= 4,
          f"val is ~{C.VAL_FRACTION:.0%} of Training ({len(val)} vs {C.VAL_FRACTION * n_training:.0f})")
    # 3. stratification
    ptr = train.label.value_counts(normalize=True)
    pva = val.label.value_counts(normalize=True)
    worst = float((ptr - pva).abs().max())
    check(worst < 0.01, f"class proportions of train and val match (max difference {worst:.3%})")
    # 4. test only comes from the Testing folder, labels are consistent
    check(test.filepath.str.startswith("Testing/").all(), "all test files come from the Testing/ folder")
    check(train.filepath.str.startswith("Training/").all() and val.filepath.str.startswith("Training/").all(),
          "all train/val files come from the Training/ folder")
    for n, t in tables.items():
        check((t.label.map(C.CLASS_TO_IDX) == t.label_idx).all(), f"{n}: label_idx matches the fixed class mapping")
    # 5. files exist
    missing = [f for t in tables.values() for f in t.filepath if not (C.PROC_DIR / f).exists()]
    check(not missing, f"every CSV path exists on disk ({len(missing)} missing)")
    # 6. reproducible: regenerating gives the same CSVs
    same = all(new[n][["filepath", "label_idx"]].equals(tables[n][["filepath", "label_idx"]]) for n in tables)
    check(same, f"regenerating the split with SEED={C.SEED} gives identical CSVs (reproducible)", level="warn")
    # 7. leakage via identical bytes (uses the audit table if it exists)
    audit = C.REPORTS_DIR / "audit.csv"
    if audit.exists():
        a = pd.read_csv(audit).set_index("key").md5
        sets = {}
        for n, t in tables.items():
            for fp in t.filepath:
                k = f"{Path(fp).parent.as_posix()}/{Path(fp).stem}"
                if k in a.index:
                    sets.setdefault(a[k], set()).add(n)
        n_leak = sum(1 for v in sets.values() if len(v) > 1)
        check(n_leak == 0, f"no byte-identical image is shared between train/val/test ({n_leak} found)", level="warn")
    else:
        print("  [SKIP] duplicate check across sets (run the audit step to enable it)")
    print("\n  -> now COMMIT splits/*.csv to git and do not regenerate them.")
    finish("splits")


# ============================================================================= main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "audit", "preprocess", "splits", "all"])
    ap.add_argument("--force", action="store_true", help="overwrite existing split CSVs")
    a = ap.parse_args()
    if a.step in ("download", "all"):
        step_download()
    if a.step in ("audit", "all"):
        step_audit()
    if a.step in ("preprocess", "all"):
        step_preprocess()
    if a.step in ("splits", "all"):
        step_splits(force=a.force)