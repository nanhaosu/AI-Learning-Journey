import sys
from pathlib import Path

import torch
import torch.nn as nn


day09_directory = (
    Path(__file__).resolve().parent.parent / "Day09"
)

if str(day09_directory) not in sys.path:
    sys.path.insert(
        0,
        str(day09_directory),
    )

from train_text_classifiers import (
    compute_binary_metrics,
    create_data_loaders,
    prepare_datasets,
)

from transformer_classifier import (
    TransformerTextClassifier,
)


def run_epoch(
    model,
    data_loader,
    loss_function,
    device,
    optimizer=None,
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_examples = 0

    all_predictions = []
    all_labels = []

    with torch.set_grad_enabled(training):
        for token_ids, _, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits, _ = model(token_ids)

            loss = loss_function(
                logits,
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_examples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels
                .detach()
                .cpu()
                .tolist()
            )

    average_loss = (
        total_loss / total_examples
    )

    metrics = compute_binary_metrics(
        all_predictions,
        all_labels,
    )

    return average_loss, metrics

def train_model(
    vocabulary_size,
    datasets,
    class_weights,
    device,
    epochs=8,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(
        datasets,
        seed=42,
    )

    model = TransformerTextClassifier(
        vocabulary_size=vocabulary_size,
        num_classes=2,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        max_length=50,
        padding_idx=0,
        dropout=0.1,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    best_validation_f1 = -1.0
    best_state = None

    print("\nModel: Transformer")
    print(
        "Trainable parameters: "
        f"{parameter_count:,}"
    )

    for epoch in range(epochs):
        train_loss, train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_metrics = (
            run_epoch(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
                device=device,
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_metrics['accuracy']:.2%} | "
            f"Validation Loss: "
            f"{validation_loss:.4f} | "
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.2%} | "
            f"Spam Precision: "
            f"{validation_metrics['precision']:.2%} | "
            f"Spam Recall: "
            f"{validation_metrics['recall']:.2%} | "
            f"Spam F1: "
            f"{validation_metrics['f1']:.2%}"
        )

        if (
            validation_metrics["f1"]
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_metrics["f1"]
            )

            best_state = {
                name: value
                .detach()
                .cpu()
                .clone()
                for name, value
                in model.state_dict().items()
            }

            print(
                "Saved best model: "
                f"{best_validation_f1:.2%}"
            )

    model.load_state_dict(best_state)

    test_loss, test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print("\nFinal test result")
    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.2%} | "
        f"Spam Precision: "
        f"{test_metrics['precision']:.2%} | "
        f"Spam Recall: "
        f"{test_metrics['recall']:.2%} | "
        f"Spam F1: "
        f"{test_metrics['f1']:.2%}"
    )

    print(
        "Confusion counts | "
        f"TP: {test_metrics['tp']} | "
        f"FP: {test_metrics['fp']} | "
        f"FN: {test_metrics['fn']} | "
        f"TN: {test_metrics['tn']}"
    )

    return {
        "parameters": parameter_count,
        "validation_f1": best_validation_f1,
        "test_loss": test_loss,
        **test_metrics,
    }

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_root = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sms_spam"
    )

    (
        datasets,
        vocabulary,
        train_label_counts,
    ) = prepare_datasets(data_root)

    total_training_examples = sum(
        train_label_counts.values()
    )

    class_weights = torch.tensor(
        [
            total_training_examples
            / (
                2
                * train_label_counts[0]
            ),
            total_training_examples
            / (
                2
                * train_label_counts[1]
            ),
        ],
        dtype=torch.float32,
        device=device,
    )

    majority_accuracy = (
        train_label_counts[0]
        / total_training_examples
    )

    print(f"Device: {device}")
    print(
        "Vocabulary size: "
        f"{len(vocabulary)}"
    )
    print(
        "Training label counts: "
        f"{dict(train_label_counts)}"
    )
    print(
        "Class weights: "
        f"{class_weights.cpu().tolist()}"
    )
    print(
        "Always-ham training baseline: "
        f"{majority_accuracy:.2%}"
    )

    transformer_result = train_model(
        vocabulary_size=len(vocabulary),
        datasets=datasets,
        class_weights=class_weights,
        device=device,
        epochs=8,
    )

    print("\nComparison with Day09")
    print(
        "MeanEmbedding | "
        "Parameters: 243,074 | "
        "Validation F1: 84.47% | "
        "Test F1: 83.33%"
    )
    print(
        "RNN           | "
        "Parameters: 251,394 | "
        "Validation F1: 89.66% | "
        "Test F1: 84.81%"
    )
    print(
        "Transformer   | "
        f"Parameters: "
        f"{transformer_result['parameters']:,} | "
        f"Validation F1: "
        f"{transformer_result['validation_f1']:.2%} | "
        f"Test F1: "
        f"{transformer_result['f1']:.2%}"
    )


if __name__ == "__main__":
    main()