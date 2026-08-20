import torch
import torch.nn as nn

output_a = torch.tensor([[0.1,0.2,3.0]])
output_b = torch.tensor([[2.0,1.0,0.5]])

label = torch.tensor([2])

loss_function = nn.CrossEntropyLoss()

loss_a = loss_function(output_a,label)
loss_b = loss_function(output_b,label)

probability_a = torch.softmax(output_a,dim=1)
probability_b = torch.softmax(output_b,dim=1)

print("Probability A:", probability_a)
print("Probability B:", probability_b)
print("Loss A:", loss_a)
print("Loss B:", loss_b)