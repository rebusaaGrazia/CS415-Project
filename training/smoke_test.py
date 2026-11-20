
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

# Import from our existing project folders.
sys.path.insert(0, str(ROOT / "implementation"))
sys.path.insert(0, str(ROOT / "dataset"))

from attention_resnet import AttentionResNet18
from dataloader import create_dataloaders


# Select available hardware.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

# Load real MRI data.
train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=8,
    augment=False
)

images, labels = next(iter(train_loader))

print("Image batch:", images.shape)
print("Label batch:", labels.shape)

# Create the baseline with ImageNet weights.
model = AttentionResNet18(
    attention_type="none",
    placement=None,
    num_classes=4,
    pretrained=True
).to(device)

images = images.to(device)
labels = labels.to(device)

model.train()

# Forward pass.
outputs = model(images)

assert outputs.shape == (8, 4)

# Compute classification loss.
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

print("Output shape:", outputs.shape)
print("CrossEntropyLoss:", loss.item())

# Backward pass.
model.zero_grad()
loss.backward()

assert model.backbone.fc.weight.grad is not None
assert torch.isfinite(loss).item()

print("Real MRI forward/backward test passed!")
