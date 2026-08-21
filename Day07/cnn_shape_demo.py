import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # 两次池化后：32×32 → 16×16 → 8×8
        # TODO 1：补全Linear的输入特征数量
        self.classifier = nn.Linear(
            2048,
            10
        )

    def forward(self, x):
        print("Input:       ", x.shape)

        x = self.conv1(x)
        print("After conv1: ", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("After pool1: ", x.shape)

        x = self.conv2(x)
        print("After conv2: ", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("After pool2: ", x.shape)

        x = self.flatten(x)
        print("After flatten:", x.shape)

        logits = self.classifier(x)
        print("Logits:      ", logits.shape)

        return logits


model = CNN()

images = torch.randn(64, 3, 32, 32)
logits = model(images)

# TODO 2：计算每张图片的预测类别
predictions = torch.argmax(logits, dim=1)

print("Predictions:  ", predictions.shape)