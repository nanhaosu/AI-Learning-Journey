from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from resnet_model import PlainCNN, SmallResNet


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_data_loaders(data_root):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

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

    split_generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(augmented_dataset), generator=split_generator)
    train_indices = indices[:45000].tolist()
    validation_indices = indices[45000:].tolist()

    train_dataset = Subset(augmented_dataset, train_indices)
    validation_dataset = Subset(evaluation_dataset, validation_indices)
    shuffle_generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def run_epoch(model, data_loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def train_model(model_name, model_class, data_root, device, epochs=5):
    set_seed(42)
    train_loader, validation_loader, test_loader = create_data_loaders(data_root)
    model = model_class().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_validation_accuracy = 0.0
    best_state = None

    print(f"\nModel: {model_name}")
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(epochs):
        # The same seed gives both models the same random augmentations per epoch.
        set_seed(1000 + epoch)
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_epoch(
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
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    test_loss, test_accuracy = run_epoch(
        model,
        test_loader,
        loss_function,
        device,
    )
    print(
        f"Best Validation Accuracy: {best_validation_accuracy:.2%} | "
        f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2%}"
    )

    return {
        "name": model_name,
        "parameters": parameter_count,
        "validation_accuracy": best_validation_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(__file__).resolve().parents[1] / "data"
    print(f"Device: {device}")

    plain_result = train_model(
        "PlainCNN",
        PlainCNN,
        data_root,
        device,
    )
    resnet_result = train_model(
        "SmallResNet",
        SmallResNet,
        data_root,
        device,
    )

    print("\nFinal comparison")
    for result in (plain_result, resnet_result):
        print(
            f"{result['name']:11s} | "
            f"Parameters: {result['parameters']:,} | "
            f"Best Validation Accuracy: {result['validation_accuracy']:.2%} | "
            f"Test Accuracy: {result['test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
