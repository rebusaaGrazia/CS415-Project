
from pathlib import Path
import sys
import random
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "implementation"))
sys.path.insert(0, str(ROOT / "dataset"))

from attention_resnet import AttentionResNet18
from dataloader import create_dataloaders


# ==========================================
# 2. EXPERIMENT SETTINGS
# ==========================================

SEED = 42
MAX_EPOCHS = 5
PATIENCE = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

EXPERIMENT_NAME = "baseline_5epoch"

RESULTS_DIR = ROOT / "training" / "results"
CHECKPOINT_DIR = ROOT / "training" / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
# 5. DATASET AND DATALOADERS
# ==========================================

train_loader, val_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE,
    augment=False
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))


# ==========================================
# 6. MODEL, LOSS AND OPTIMIZER
# ==========================================

model = AttentionResNet18(
    attention_type="none",
    placement=None,
    num_classes=4,
    pretrained=True
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 7. TRAIN ONE EPOCH
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
    predictions = []
    true_labels = []

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

    val_accuracy = accuracy_score(
        true_labels, predictions
    )

    val_macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    return val_loss, val_accuracy, val_macro_f1


# ==========================================
# 9. TRAINING WITH EARLY STOPPING
# ==========================================

history = []

best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

checkpoint_path = (
    CHECKPOINT_DIR / f"{EXPERIMENT_NAME}_best.pth"
)

print("\nStarting baseline training...\n", flush=True)

for epoch in range(MAX_EPOCHS):

    start = time.perf_counter()

    train_loss = train_one_epoch()

    val_loss, val_accuracy, val_f1 = validate()

    # Finish pending GPU operations before timing.
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

    # Early stopping monitors validation Macro-F1.
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            checkpoint_path
        )
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro-F1: {val_f1:.4f} | "
        f"Time: {elapsed:.1f}s",
        flush=True
    )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping triggered at "
            f"epoch {epoch + 1}.",
            flush=True
        )
        break


# ==========================================
# 10. TRAINING SUMMARY
# ==========================================

print("\nTraining completed!")
print("Epochs completed:", len(history))
print("Best epoch:", best_epoch)
print("Best validation Macro-F1:", round(best_f1, 4))
print("Best checkpoint:", checkpoint_path)


# ==========================================
# 11. SAVE RESULTS AS CSV
# ==========================================

csv_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_history.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )
    writer.writeheader()
    writer.writerows(history)

print("CSV saved:", csv_path)


# ==========================================
# 12. GENERATE REPORT FIGURES
# ==========================================

epochs = [row["epoch"] for row in history]

train_losses = [row["train_loss"] for row in history]
val_losses = [row["val_loss"] for row in history]

val_accuracies = [row["val_accuracy"] for row in history]
val_f1s = [row["val_macro_f1"] for row in history]


# Figure 1 — Training vs validation loss

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    epochs, train_losses,
    marker="o",
    label="Training Loss"
)

ax.plot(
    epochs, val_losses,
    marker="o",
    label="Validation Loss"
)

ax.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("ResNet18 Baseline — Training and Validation Loss")
ax.set_xticks(epochs)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / f"{EXPERIMENT_NAME}_loss.png",
    dpi=300
)
plt.close(fig)


# Figure 2 — Validation performance

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(
    epochs, val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

ax.plot(
    epochs, val_f1s,
    marker="o",
    label="Validation Macro-F1"
)

ax.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticks(epochs)
ax.set_title("ResNet18 Baseline — Validation Performance")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / f"{EXPERIMENT_NAME}_metrics.png",
    dpi=300
)
plt.close(fig)

print("\nCSV and figures saved successfully!")
