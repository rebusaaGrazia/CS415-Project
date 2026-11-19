import torch
from se import SEBlock

model = SEBlock(channels=64)

x = torch.randn(2, 64, 56, 56)
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

loss = output.sum()
loss.backward()

print("SE forward and backward test passed!")
