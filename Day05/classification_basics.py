import torch

logits = torch.tensor([
    [2.0,1.0,0.1],
    [0.5,2.5,1.0]
])

labels = torch.tensor([0,2])

predictions = torch.argmax(logits, dim=1)

correct = (predictions == labels).sum()

accuracy = correct / 2

print("Predictions:", predictions)
print("Labels:", labels)
print("Correct:", correct)
print("Accuracy:", accuracy)