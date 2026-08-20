import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = datasets.FashionMNIST(
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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


model = MLP().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_one_epoch():
    # TODO 1：切换到训练模式
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # TODO 2：把数据移动到device
        images = images.to(device)
        labels = labels.to(device)

        # TODO 3～7：清空梯度、前向传播、计算损失、
        #             反向传播、更新参数
        optimizer.zero_grad()

        logits = model(images)

        loss=loss_function(logits,labels)

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        correct += (predictions == labels).sum().item()
        total += len(labels)

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return average_loss, accuracy


def evaluate():
    # TODO 8：切换到评估模式
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    # TODO 9：评估时关闭梯度计算
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # TODO 10：前向传播并计算loss
            logits = model(images)
            loss = loss_function(logits,labels)

            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    average_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return average_loss, accuracy

best_accuracy = 0.0
checkpoint_path = Path(__file__).with_name("best_mlp.pth")

epochs = 5

for epoch in range(epochs):
    train_loss, train_accuracy = train_one_epoch()
    test_loss, test_accuracy = evaluate()

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2%} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: {test_accuracy:.2%}"
    )

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(f"Saved best model: {best_accuracy:.2%}")

print("\nLoading best model...")

best_state = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(best_state)

best_loss, best_accuracy = evaluate()

print(f"Loaded Model Loss: {best_loss:.4f}")
print(f"Loaded Model Accuracy: {best_accuracy:.2%}")
