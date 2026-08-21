from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.classifier(x)


def create_data_loaders(data_root):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # These two dataset objects point to the same training images but apply
    # different transforms. Validation must not use random augmentation.
    augmented_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    evaluation_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=evaluation_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=evaluation_transform,
    )

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(augmented_dataset), generator=generator)
    train_indices = indices[:45000].tolist()
    validation_indices = indices[45000:].tolist()

    train_dataset = Subset(augmented_dataset, train_indices)
    validation_dataset = Subset(evaluation_dataset, validation_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, validation_loader, test_loader


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        correct += (predictions == labels).sum().item()
        total += len(labels)

    return total_loss / len(data_loader), correct / total


def evaluate(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / len(data_loader), correct / total


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(__file__).resolve().parents[1] / "data"
    checkpoint_path = Path(__file__).with_name("best_cnn.pth")

    train_loader, validation_loader, test_loader = create_data_loaders(data_root)
    model = CNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_validation_accuracy = 0.0

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model: {best_validation_accuracy:.2%}")

    best_state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    print("\nFinal test result")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
