import torch
from se import SEBlock
from cbam import CBAMBlock
from attention_resnet import AttentionResNet18

model = SEBlock(channels=64)

x = torch.randn(2, 64, 56, 56)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

loss = output.sum()
loss.backward()

print("SE forward and backward test passed!")



cbam = CBAMBlock(channels=64)

x = torch.randn(2, 64, 56, 56)
output = cbam(x)

print("CBAM input shape:", x.shape)
print("CBAM output shape:", output.shape)

assert output.shape == x.shape

output.sum().backward()

assert all(
    p.grad is not None for p in cbam.parameters()
)

print("CBAM forward and backward test passed!")


configurations = [
    ("none", None),
    ("se", "early"),
    ("se", "middle"),
    ("se", "late"),
    ("cbam", "early"),
    ("cbam", "middle"),
    ("cbam", "late"),
]

print("\nTesting all ResNet18 configurations...")

for attention_type, placement in configurations:
    # No pretrained download needed for structural tests.
    model = AttentionResNet18(
        attention_type=attention_type,
        placement=placement,
        num_classes=4,
        pretrained=False
    )

    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 4)

    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"{attention_type:5s} | "
        f"{str(placement):6s} | "
        f"output={tuple(output.shape)} | "
        f"parameters={total_params:,}"
    )

print("All seven configurations passed!")
