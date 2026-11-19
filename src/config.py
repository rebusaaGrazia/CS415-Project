"""Shared configuration. EVERYONE on the team imports these values; nobody hardcodes their own."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where the data lives. Override with the env var BRAIN_DATA_DIR (e.g. a Google Drive path in Colab).
DATA_DIR = Path(os.environ.get("BRAIN_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
SPLITS_DIR = ROOT / "splits"      # small CSVs: these ARE committed to git
REPORTS_DIR = ROOT / "reports"    # audit tables and check images

KAGGLE_ID = "masoudnickparvar/brain-tumor-mri-dataset"

# Classes (alphabetical = the folder names in the dataset). The index order is FIXED for the whole project.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# What the proposal says the dataset contains (used only to warn if reality differs)
EXPECTED_PER_CLASS = {"Training": 1400, "Testing": 400}

# Preprocessing
IMG_SIZE = 224
CROP_THRESH = 10            # pixels <= this (0-255) count as black background
MIN_CROP_AREA_FRAC = 0.10   # if the crop keeps less than 10% of the image, the crop failed -> keep original
RESIZE_MODE = "pad"         # "pad": pad the crop to a square with black, then resize (keeps the shape)
                            # "stretch": resize directly to IMG_SIZE x IMG_SIZE (distorts the aspect ratio)

# Splits
SEED = 42
VAL_FRACTION = 0.15

# Loading
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics (match the pretrained ResNet18)
STD = [0.229, 0.224, 0.225]
AUGMENT = False                # agree with the team BEFORE turning this on; train only, same for all 7 models