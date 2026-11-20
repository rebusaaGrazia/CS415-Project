
import torch

from dataloader import create_dataloaders, CLASSES


train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=16,
    augment=False
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))
print("Testing images:", len(test_loader.dataset))

for name, loader in [
    ("Train", train_loader),
    ("Validation", val_loader),
    ("Test", test_loader)
]:
    images, labels = next(iter(loader))

    assert images.shape[1:] == (3, 224, 224)
    assert images.dtype == torch.float32
    assert torch.isfinite(images).all()
    assert labels.min() >= 0
    assert labels.max() < 4

    print(
        f"{name}: "
        f"images={tuple(images.shape)}, "
        f"labels={tuple(labels.shape)}"
    )

print("Class mapping:", dict(enumerate(CLASSES)))
print("All DataLoader tests passed!")
