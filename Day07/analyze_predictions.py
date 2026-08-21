from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train_cifar10 import CNN


CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def predict_batch(model, images):
    model.eval()
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
    return probabilities, predictions, confidence


def denormalize(images):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    day_directory = Path(__file__).resolve().parent
    data_root = day_directory.parent / "data"
    checkpoint_path = day_directory / "best_cnn.pth"
    output_path = day_directory / "prediction_examples.png"

    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=evaluation_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = CNN().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    images, labels = next(iter(test_loader))
    probabilities, predictions, confidence = predict_batch(
        model,
        images.to(device),
    )

    print(f"Device: {device}")
    for index in range(len(labels)):
        result = "correct" if predictions[index].cpu() == labels[index] else "wrong"
        print(
            f"Image {index + 1:2d} | "
            f"True: {CLASS_NAMES[labels[index]]:10s} | "
            f"Predicted: {CLASS_NAMES[predictions[index]]:10s} | "
            f"Confidence: {confidence[index]:.2%} | {result}"
        )

    display_images = denormalize(images)
    save_image(display_images, output_path, nrow=5, padding=2)
    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
