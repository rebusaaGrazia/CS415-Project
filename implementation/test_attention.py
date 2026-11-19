import torch
from se import SEBlock
from cbam import CBAMBlock

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
