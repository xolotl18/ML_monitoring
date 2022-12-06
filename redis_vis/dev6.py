import torch

script = torch.jit.load('resample.pt')

input_signal = torch.randn(1, 100)
print(script(input_signal))