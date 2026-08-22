import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
   def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
        # TODO：1×1卷积，使用同一个stride
                nn.Conv2d(
                     in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=stride,
                     bias=False,
                ),
                nn.BatchNorm2d(out_channels)
        # TODO：BatchNorm2d
            )
        else:
            self.shortcut = nn.Identity()

   def forward(self,x):
            identity = self.shortcut(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)


            out = self.conv2(out)
            out = self.bn2(out)

            out = out + identity
            out = self.relu(out)

            return out
   
class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.layer1 = ResidualBlock(
            in_channels=16,
            out_channels=16,
            stride=1,
        )

        self.layer2 = ResidualBlock(
            in_channels=16,
            out_channels=32,
            stride=2,
        )

        self.layer3 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64,10)
        

    def forward(self, x):
        x = self.stem(x)
        print("After stem:  ", x.shape)

        x = self.layer1(x)
        print("After layer1:", x.shape)

        x = self.layer2(x)
        print("After layer2:", x.shape)

        x =self.layer3(x)
        print("After layer3:", x.shape)

        x = self.avg_pool(x)
        print("After pooling:", x.shape)

        x = self.flatten(x)
        print("After flatten:", x.shape)

        logits = self.classifier(x)
        print("Logits:    ",logits.shape)

        return logits


model = SmallResNet()
images = torch.randn(64, 3, 32, 32)
logits = model(images)

print("Final output:", logits.shape)