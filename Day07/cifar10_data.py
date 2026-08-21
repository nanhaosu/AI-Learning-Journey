from torch.utils.data import DataLoader
from torchvision import datasets, transforms


mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# TODO：从train_loader取出一个batch
images, labels = next(iter(train_loader))

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Pixel minimum:", images.min())
print("Pixel maximum:", images.max())

print("\nFirst 10 labels:")
for label in labels[:10]:
    index = label.item()
    print(index, class_names[index])