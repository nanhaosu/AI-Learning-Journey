import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset  = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images,labels = next(iter(train_loader))

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Pixel minimum:", images.min())
print("Pixel maximum:", images.max())
print("First 10 labels:", labels[:10])