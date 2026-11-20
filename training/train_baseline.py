
from pathlib import Path
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementation"))
sys.path.insert(0, str(ROOT / "dataset"))

from attention_resnet import AttentionResNet18
from dataloader import create_dataloaders


# ----- Experiment settings -----
SEED = 42
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

train_loader, val_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE,
    augment=False
)

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


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader):
    model.eval()

    total_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )
            true_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        true_labels, predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    return avg_loss, accuracy, macro_f1


# ----- Training -----
history = []

checkpoint_dir = ROOT / "training" / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_f1 = -1.0

for epoch in range(EPOCHS):
    start = time.perf_counter()

    train_loss = train_one_epoch(model, train_loader)

    val_loss, val_accuracy, val_f1 = validate(
        model, val_loader
    )

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

        torch.save(
            model.state_dict(),
            checkpoint_dir / "baseline_best.pth"
        )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro-F1: {val_f1:.4f} | "
        f"Time: {elapsed:.1f}s",
        flush=True
    )

print("\nBest validation Macro-F1:", round(best_f1, 4))
print("Training test completed!")





RESULTS_DIR = ROOT / "training" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save numerical results.
csv_path = RESULTS_DIR / "baseline_history.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )
    writer.writeheader()
    writer.writerows(history)

# Extract values for plotting.
epochs = [row["epoch"] for row in history]
train_losses = [row["train_loss"] for row in history]
val_losses = [row["val_loss"] for row in history]
val_accuracies = [row["val_accuracy"] for row in history]
val_f1s = [row["val_macro_f1"] for row in history]

# Figure 1: Training and validation loss.
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(epochs, train_losses, marker="o", label="Training loss")
ax.plot(epochs, val_losses, marker="o", label="Validation loss")

ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("ResNet18 Baseline — Training and Validation Loss")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "baseline_loss.png", dpi=300)
plt.close(fig)

# Figure 2: Validation accuracy and macro-F1.
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(epochs, val_accuracies, marker="o", label="Accuracy")
ax.plot(epochs, val_f1s, marker="o", label="Macro-F1")

ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("ResNet18 Baseline — Validation Metrics")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "baseline_metrics.png", dpi=300)
plt.close(fig)

print("Results saved to:", RESULTS_DIR)
