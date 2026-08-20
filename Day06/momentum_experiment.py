from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)


def train_one_epoch(model, optimizer, data_loader, loss_function, device):
    """Train the model on every batch in data_loader once."""
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
    """Measure loss and accuracy without updating model parameters."""
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


def create_data_loaders(data_root):
    """Create a fixed 50k/10k train-validation split."""
    full_train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    split_generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [50000, 10000],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )
    return train_loader, validation_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(__file__).resolve().parents[1] / "data"
    train_loader, validation_loader = create_data_loaders(data_root)
    loss_function = nn.CrossEntropyLoss()

    learning_rates = [0.1]
    epochs = 3
    final_results = []

    print(f"Device: {device}")

    for learning_rate in learning_rates:
        # Every experiment starts from the same initialization and shuffle seed.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model = MLP().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )

        print(f"\nLearning Rate: {learning_rate}")

        for epoch in range(epochs):
            train_loss, train_accuracy = train_one_epoch(
                model,
                optimizer,
                train_loader,
                loss_function,
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

        final_results.append(
            (learning_rate, validation_loss, validation_accuracy)
        )

    print("\nFinal validation results")
    for learning_rate, validation_loss, validation_accuracy in final_results:
        print(
            f"lr={learning_rate:<5} | "
            f"loss={validation_loss:.4f} | "
            f"accuracy={validation_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
