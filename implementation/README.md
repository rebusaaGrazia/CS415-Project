# CS415 — Attention Implementation

This folder contains the SE and CBAM attention modules integrated into ResNet18 for four-class brain tumor MRI classification.

## Files

- `se.py` — Squeeze-and-Excitation (SE) attention module.
- `cbam.py` — Convolutional Block Attention Module (CBAM).
- `attention_resnet.py` — ResNet18 with configurable attention type and placement.
- `test_attention.py` — Structural, gradient, pretrained-weight and reproducibility tests.

## Model Configurations

Seven configurations are supported:

- ResNet18 baseline (no attention)
- SE — Early, Middle, Late
- CBAM — Early, Middle, Late

Attention is inserted once, immediately after the selected ResNet18 stage:

| Placement | ResNet18 stage |
|---|---|
| Early | `layer1` |
| Middle | `layer2` |
| Late | `layer4` |

## Usage

```python
from attention_resnet import AttentionResNet18

model = AttentionResNet18(
    attention_type="se",
    placement="middle",
    num_classes=4,
    pretrained=True
)
```

To create the baseline:

```python
model = AttentionResNet18(
    attention_type="none",
    placement=None,
    num_classes=4,
    pretrained=True
)
```

The models expect three-channel image tensors with shape `(batch_size, 3, 224, 224)` and return four classification logits per image.

## Testing

Install PyTorch and torchvision, then run from the implementation directory:

```bash
python3 test_attention.py
```

All seven configurations have passed structural and gradient tests, including attention placement, pretrained-weight loading and reproducibility checks.

**Note:** These tests do not establish classification accuracy or clinical reliability. Real-data training and evaluation are handled separately.

## Experiment Design

The planned evaluation uses seven configurations with three random seeds each, for a total of 21 training runs. Placement rankings will be compared using mean macro-F1.

The training pipeline should use consistent dataset splits, preprocessing and training settings across configurations.