import torch
import torch.nn as nn

x = torch.linspace(-5,5,10)

sigmoid = nn.Sigmoid()
relu = nn.ReLU()
gelu = nn.GELU()

sigmoid_output = sigmoid(x)
relu_output = relu(x)
gelu_output = gelu(x)

print("Input:")
print(x)

print("\nSigmoid Output:")
print(sigmoid_output)

print("\nReLU Output:")
print(relu_output)

print("\nGELU Output:")
print(gelu_output)