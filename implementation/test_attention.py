import torch
from se import SEBlock
from cbam import CBAMBlock
from attention_resnet import AttentionResNet18
from torchvision.models import resnet18, ResNet18_Weights

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


print("\nTesting attention integration and gradients...")

expected_channels = {
    "early": 64,
    "middle": 128,
    "late": 512
}

for attention_type in ("se", "cbam"):
    for placement in ("early", "middle", "late"):

        model = AttentionResNet18(
            attention_type=attention_type,
            placement=placement,
            pretrained=False
        )

        model.eval()

        # Record the tensor shape entering attention.
        recorded_shapes = []

        def record_input(module, inputs):
            recorded_shapes.append(tuple(inputs[0].shape))

        hook = model.attention.register_forward_pre_hook(
            record_input
        )

        x = torch.randn(1, 3, 224, 224)

        output = model(x)
        loss = output.square().mean()
        loss.backward()

        hook.remove()

        # Attention should be called exactly once.
        assert len(recorded_shapes) == 1

        # Verify expected channel count.
        assert recorded_shapes[0][1] == expected_channels[placement]

        # Verify attention parameters receive gradients.
        attention_params = list(model.attention.parameters())

        assert len(attention_params) > 0
        assert all(p.grad is not None for p in attention_params)

        print(
            f"{attention_type.upper()} {placement}: "
            f"shape={recorded_shapes[0]}, "
            f"gradients=OK"
        )

print("All attention integration tests passed!")


print("\nTesting whether attention changes model outputs...")

torch.manual_seed(42)

baseline = AttentionResNet18(
    attention_type="none",
    placement=None,
    pretrained=False
)

torch.manual_seed(42)

se_model = AttentionResNet18(
    attention_type="se",
    placement="early",
    pretrained=False
)

# Ensure both models have identical backbone weights.
baseline_weights = baseline.backbone.state_dict()
se_weights = se_model.backbone.state_dict()

assert all(
    torch.equal(baseline_weights[key], se_weights[key])
    for key in baseline_weights
)

baseline.eval()
se_model.eval()

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    baseline_output = baseline(x)
    se_output = se_model(x)

assert baseline_output.shape == (2, 4)
assert se_output.shape == (2, 4)

difference = torch.max(
    torch.abs(baseline_output - se_output)
).item()

print("Maximum output difference:", difference)

assert difference > 1e-6, "Attention did not change outputs!"

print("Attention functional validation passed!")



print("\nTesting pretrained ResNet18 weights...")

# Original ImageNet-pretrained ResNet18
reference = resnet18(weights=ResNet18_Weights.DEFAULT)

# Our model with SE attention
model = AttentionResNet18(
    attention_type="se",
    placement="early",
    num_classes=4,
    pretrained=True
)

# Compare the original backbone's convolution weights.
assert torch.equal(
    reference.conv1.weight,
    model.backbone.conv1.weight
)

# Compare weights in the final residual stage.
assert torch.equal(
    reference.layer4[1].conv2.weight,
    model.backbone.layer4[1].conv2.weight
)

# Verify that our output layer has 4 classes.
assert model.backbone.fc.out_features == 4

model.eval()

with torch.no_grad():
    output = model(torch.randn(1, 3, 224, 224))

assert output.shape == (1, 4)

print("Pretrained weights loaded correctly!")
print("Four-class classification head verified!")


print("\nTesting invalid configurations...")

invalid_configs = [
    {"attention_type": "invalid", "placement": "early"},
    {"attention_type": "se", "placement": None},
    {"attention_type": "cbam", "placement": "layer3"},
    {"attention_type": "none", "placement": "early"},
]

for config in invalid_configs:
    try:
        model = AttentionResNet18(
            **config,
            pretrained=False
        )
    except ValueError:
        print(f"Correctly rejected: {config}")
    else:
        raise AssertionError(
            f"Invalid configuration was accepted: {config}"
        )

print("All invalid configuration tests passed!")


print("\nTesting reproducibility...")

def create_model_with_seed(seed):
    torch.manual_seed(seed)

    model = AttentionResNet18(
        attention_type="cbam",
        placement="middle",
        num_classes=4,
        pretrained=False
    )

    model.eval()
    return model


model_a = create_model_with_seed(42)
model_b = create_model_with_seed(42)
model_c = create_model_with_seed(123)

# Same seed should produce identical initial weights.
for key in model_a.state_dict():
    assert torch.equal(
        model_a.state_dict()[key],
        model_b.state_dict()[key]
    )

print("Same seed produces identical weights: PASSED")

# Different seeds should produce different initial weights.
assert not torch.equal(
    model_a.backbone.fc.weight,
    model_c.backbone.fc.weight
)

print("Different seeds produce different weights: PASSED")

# Same input and weights should produce identical outputs.
torch.manual_seed(99)
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output_a = model_a(x)
    output_b = model_b(x)

assert torch.allclose(output_a, output_b, atol=1e-6)

print("Repeated model outputs match: PASSED")
print("All reproducibility tests passed!")
