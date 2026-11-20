
from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation"))
sys.path.insert(0, str(ROOT / "dataset"))

from attention_resnet import AttentionResNet18
from dataloader import create_dataloaders

RESULTS_DIR = ROOT / "training" / "results"
CHECKPOINT_DIR = ROOT / "training" / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. COMMAND-LINE CONFIGURATION
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--attention",
    choices=["none", "se", "cbam"],
    required=True
)
parser.add_argument(
    "--placement",
    choices=["early", "middle", "late"],
    default=None
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-epochs", type=int, default=5)
parser.add_argument("--patience", type=int, default=4)

args = parser.parse_args()

if args.attention == "none" and args.placement is not None:
    parser.error("Baseline must not have a placement.")

if args.attention != "none" and args.placement is None:
    parser.error("SE and CBAM require a placement.")

if args.max_epochs < 1 or args.patience < 1:
    parser.error("max-epochs and patience must be positive.")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4

model_name = (
    "baseline"
    if args.attention == "none"
    else f"{args.attention}_{args.placement}"
)

experiment_name = f"{model_name}_seed{args.seed}"

print("Experiment:", experiment_name, flush=True)


# ==========================================
# 3. REPRODUCIBILITY
# ==========================================

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ==========================================
# 4. DEVICE
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device, flush=True)


# ==========================================
# 5. DATA
# ==========================================

train_loader, val_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE,
    augment=False
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))


# ==========================================
# 6. MODEL AND OPTIMIZER
# ==========================================

model = AttentionResNet18(
    attention_type=args.attention,
    placement=args.placement,
    num_classes=4,
    pretrained=True
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 7. TRAINING
# ==========================================

def train_one_epoch():
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_loader.dataset)


# ==========================================
# 8. VALIDATION
# ==========================================

def validate():
    model.eval()

    total_loss = 0.0
    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )
            true_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(true_labels, predictions)
    val_macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    return val_loss, val_accuracy, val_macro_f1


# ==========================================
# 9. EARLY STOPPING AND CHECKPOINTING
# ==========================================

history = []
best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0
stopped_early = False

checkpoint_path = (
    CHECKPOINT_DIR / f"{experiment_name}_best.pth"
)

print("\nStarting training...\n", flush=True)

for epoch in range(args.max_epochs):
    start = time.perf_counter()

    train_loss = train_one_epoch()
    val_loss, val_accuracy, val_f1 = validate()

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_f1,
        "seconds": elapsed
    })

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1}/{args.max_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro-F1: {val_f1:.4f} | "
        f"Time: {elapsed:.1f}s",
        flush=True
    )

    if epochs_without_improvement >= args.patience:
        stopped_early = True
        print("Early stopping triggered.", flush=True)
        break


# ==========================================
# 10. SAVE RESULTS
# ==========================================

csv_path = RESULTS_DIR / f"{experiment_name}_history.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )
    writer.writeheader()
    writer.writerows(history)

metadata = {
    "experiment": experiment_name,
    "attention": args.attention,
    "placement": args.placement,
    "seed": args.seed,
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW",
    "augmentation": False,
    "checkpoint_metric": "validation_macro_f1",
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_f1,
    "epochs_completed": len(history),
    "stopped_early": stopped_early,
    "total_seconds": sum(r["seconds"] for r in history)
}

metadata_path = RESULTS_DIR / f"{experiment_name}_metadata.json"

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=2)


# ==========================================
# 11. GENERATE FIGURES
# ==========================================

epochs = [r["epoch"] for r in history]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    epochs,
    [r["train_loss"] for r in history],
    marker="o", label="Training Loss"
)
ax.plot(
    epochs,
    [r["val_loss"] for r in history],
    marker="o", label="Validation Loss"
)
ax.axvline(
    best_epoch, linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title(f"{experiment_name} — Loss")
ax.set_xticks(epochs)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / f"{experiment_name}_loss.png",
    dpi=300
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    epochs,
    [r["val_accuracy"] for r in history],
    marker="o", label="Validation Accuracy"
)
ax.plot(
    epochs,
    [r["val_macro_f1"] for r in history],
    marker="o", label="Validation Macro-F1"
)
ax.axvline(
    best_epoch, linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(epochs)
ax.set_title(f"{experiment_name} — Validation Metrics")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / f"{experiment_name}_metrics.png",
    dpi=300
)
plt.close(fig)


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\nTraining completed!")
print("Best epoch:", best_epoch)
print("Best validation Macro-F1:", round(best_f1, 4))
print("Checkpoint:", checkpoint_path)
print("CSV:", csv_path)
print("Metadata:", metadata_path)
print("Figures saved to:", RESULTS_DIR)
