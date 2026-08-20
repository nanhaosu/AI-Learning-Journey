import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


full_train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# 固定随机种子，使每次划分得到相同结果
generator = torch.Generator().manual_seed(42)

# TODO：将60000个训练样本拆成50000和10000
train_dataset, validation_dataset = random_split(
    full_train_dataset,
    # 补全这里
    [50000,10000],
    generator=generator
)

print("Full training dataset:", len(full_train_dataset))
print("Training dataset:", len(train_dataset))
print("Validation dataset:", len(validation_dataset))
print("Test dataset:", len(test_dataset))