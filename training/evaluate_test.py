
from pathlib import Path
import csv
import json
import sys
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "implementation"))
sys.path.insert(0, str(ROOT / "dataset"))

from attention_resnet import AttentionResNet18
from dataloader import create_dataloaders

TRAIN_RESULTS = ROOT / "training" / "results"
CHECKPOINT_DIR = ROOT / "training" / "checkpoints"
TEST_DIR = TRAIN_RESULTS / "test"

SEEDS = [42, 123, 2026]

CONFIGURATIONS = [
    ("baseline", "none", None),
    ("se_early", "se", "early"),
    ("se_middle", "se", "middle"),
    ("se_late", "se", "late"),
    ("cbam_early", "cbam", "early"),
    ("cbam_middle", "cbam", "middle"),
    ("cbam_late", "cbam", "late"),
]

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

# ==========================================
# 2. VERIFY ALL 21 SAVED CHECKPOINTS
# ==========================================

experiments = []

for config, attention, placement in CONFIGURATIONS:
    for seed in SEEDS:
        name = f"{config}_seed{seed}"

        checkpoint = CHECKPOINT_DIR / f"{name}_best.pth"
        metadata_file = TRAIN_RESULTS / f"{name}_metadata.json"

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        if not metadata_file.is_file():
            raise FileNotFoundError(metadata_file)

        with open(metadata_file) as file:
            metadata = json.load(file)

        if metadata["experiment"] != name:
            raise ValueError(f"Metadata mismatch: {name}")

        if metadata["attention"] != attention:
            raise ValueError(f"Attention mismatch: {name}")

        if metadata["placement"] != placement:
            raise ValueError(f"Placement mismatch: {name}")

        if int(metadata["seed"]) != seed:
            raise ValueError(f"Seed mismatch: {name}")

        experiments.append({
            "name": name,
            "configuration": config,
            "attention": attention,
            "placement": placement,
            "seed": seed,
            "checkpoint": checkpoint,
        })

assert len(experiments) == 21

print("Verified all 21 checkpoints and metadata files.")

# ==========================================
# 3. DEVICE AND TEST DATA
# ==========================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

_, _, test_loader = create_dataloaders(
    batch_size=16,
    augment=False
)

print("Test images:", len(test_loader.dataset))

if len(test_loader.dataset) != 1584:
    raise ValueError("Unexpected test dataset size.")

# Confirm class-index order before labeling results.
dataset_classes = getattr(
    test_loader.dataset, "classes", None
)
if dataset_classes is not None:
    if list(dataset_classes) != CLASS_NAMES:
        raise ValueError("Unexpected class order.")

TEST_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. CSV HELPER
# ==========================================

def save_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

# ==========================================
# 5. TEST EVALUATION FUNCTION
# ==========================================

def evaluate_model(model):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        true_labels, predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2, 3]
    )

    return accuracy, macro_f1, recalls, matrix

# ==========================================
# 6. EVALUATE ALL CHECKPOINTS
# ==========================================

results = []
confusion_matrices = {}

for index, experiment in enumerate(experiments, start=1):

    name = experiment["name"]

    model = AttentionResNet18(
        attention_type=experiment["attention"],
        placement=experiment["placement"],
        num_classes=4,
        pretrained=False
    ).to(device)

    # Load already selected best validation checkpoint.
    state_dict = torch.load(
        experiment["checkpoint"],
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)

    accuracy, macro_f1, recalls, matrix = (
        evaluate_model(model)
    )

    results.append({
        "configuration": experiment["configuration"],
        "seed": experiment["seed"],
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "recall_glioma": float(recalls[0]),
        "recall_meningioma": float(recalls[1]),
        "recall_notumor": float(recalls[2]),
        "recall_pituitary": float(recalls[3]),
    })

    confusion_matrices[name] = matrix

    print(
        f"[{index:02d}/21] {name:24s} "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro-F1: {macro_f1:.4f}",
        flush=True
    )

    del model
    del state_dict

# ==========================================
# 7. SAVE PER-RUN RESULTS
# ==========================================

save_csv(
    TEST_DIR / "test_results_all_runs.csv",
    results
)

# ==========================================
# 8. MEAN AND STANDARD DEVIATION
# ==========================================

summary = []

for config, _, _ in CONFIGURATIONS:
    selected = [
        r for r in results
        if r["configuration"] == config
    ]

    f1_scores = [
        r["test_macro_f1"] for r in selected
    ]

    accuracies = [
        r["test_accuracy"] for r in selected
    ]

    row = {
        "configuration": config,
        "mean_test_macro_f1": statistics.mean(f1_scores),
        "std_test_macro_f1": statistics.stdev(f1_scores),
        "mean_test_accuracy": statistics.mean(accuracies),
        "std_test_accuracy": statistics.stdev(accuracies),
    }

    for class_name in CLASS_NAMES:
        scores = [
            r[f"recall_{class_name}"]
            for r in selected
        ]

        row[f"mean_recall_{class_name}"] = (
            statistics.mean(scores)
        )

    summary.append(row)

save_csv(
    TEST_DIR / "test_configuration_summary.csv",
    summary
)

# ==========================================
# 9. CONFUSION MATRICES
# ==========================================

# Save one combined confusion matrix per configuration.
# Each matrix pools predictions across its three runs.
# Because the same test images are evaluated three times,
# totals represent model-image predictions, not unique images.

for config, _, _ in CONFIGURATIONS:

    combined = np.zeros((4, 4), dtype=int)

    for seed in SEEDS:
        name = f"{config}_seed{seed}"
        combined += confusion_matrices[name]

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=combined,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{config} — Pooled Confusion Matrix (3 Seeds)"
    )

    fig.tight_layout()

    fig.savefig(
        TEST_DIR / f"{config}_confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n=== TEST RESULTS: MEAN ± SD ===")

for row in summary:
    print(
        f"{row['configuration']:14s} "
        f"Macro-F1: "
        f"{row['mean_test_macro_f1']:.4f} "
        f"± {row['std_test_macro_f1']:.4f} | "
        f"Accuracy: "
        f"{row['mean_test_accuracy']:.4f}"
    )

print("\nTest evaluation completed.")
print("Results saved to:", TEST_DIR)
