from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from text_data import (
    SmsDataset,
    build_vocabulary,
    collate_batch,
    download_sms_dataset,
    load_labeled_messages,
    stratified_split,
)
from text_models import MeanEmbeddingClassifier, RNNTextClassifier


def compute_binary_metrics(predictions, labels):
    tp = sum(
        prediction == 1 and label == 1
        for prediction, label in zip(predictions, labels)
    )
    fp = sum(
        prediction == 1 and label == 0
        for prediction, label in zip(predictions, labels)
    )
    fn = sum(
        prediction == 0 and label == 1
        for prediction, label in zip(predictions, labels)
    )
    tn = sum(
        prediction == 0 and label == 0
        for prediction, label in zip(predictions, labels)
    )

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_epoch(model, data_loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.set_grad_enabled(training):
        for token_ids, lengths, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(token_ids, lengths)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            total += batch_size
            all_predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_binary_metrics(all_predictions, all_labels)
    return total_loss / total, metrics


def prepare_datasets(data_root):
    data_path = download_sms_dataset(data_root)
    examples = load_labeled_messages(data_path)
    train_examples, validation_examples, test_examples = stratified_split(
        examples,
        seed=42,
    )
    vocabulary = build_vocabulary(
        [text for text, _ in train_examples],
        min_frequency=2,
        max_size=10000,
    )
    datasets = (
        SmsDataset(train_examples, vocabulary, max_length=50),
        SmsDataset(validation_examples, vocabulary, max_length=50),
        SmsDataset(test_examples, vocabulary, max_length=50),
    )
    train_label_counts = Counter(label for _, label in train_examples)
    return datasets, vocabulary, train_label_counts


def create_data_loaders(datasets, seed):
    train_dataset, validation_dataset, test_dataset = datasets
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return train_loader, validation_loader, test_loader


def train_model(
    model_name,
    model_factory,
    datasets,
    class_weights,
    device,
    epochs=8,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    train_loader, validation_loader, test_loader = create_data_loaders(
        datasets,
        seed=42,
    )
    model = model_factory().to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_validation_f1 = -1.0
    best_state = None

    print(f"\nModel: {model_name}")
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(epochs):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_loss, validation_metrics = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_metrics['accuracy']:.2%} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_metrics['accuracy']:.2%} | "
            f"Spam Precision: {validation_metrics['precision']:.2%} | "
            f"Spam Recall: {validation_metrics['recall']:.2%} | "
            f"Spam F1: {validation_metrics['f1']:.2%}"
        )

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        loss_function,
        device,
    )
    print(
        f"Best Validation F1: {best_validation_f1:.2%} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: {test_metrics['accuracy']:.2%} | "
        f"Spam Precision: {test_metrics['precision']:.2%} | "
        f"Spam Recall: {test_metrics['recall']:.2%} | "
        f"Spam F1: {test_metrics['f1']:.2%}"
    )
    print(
        f"Confusion counts | TP: {test_metrics['tp']} | "
        f"FP: {test_metrics['fp']} | FN: {test_metrics['fn']} | "
        f"TN: {test_metrics['tn']}"
    )
    return {
        "name": model_name,
        "parameters": parameter_count,
        "validation_f1": best_validation_f1,
        "test_loss": test_loss,
        **test_metrics,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(__file__).resolve().parents[1] / "data" / "sms_spam"
    datasets, vocabulary, train_label_counts = prepare_datasets(data_root)

    total_training_examples = sum(train_label_counts.values())
    class_weights = torch.tensor(
        [
            total_training_examples / (2 * train_label_counts[0]),
            total_training_examples / (2 * train_label_counts[1]),
        ],
        dtype=torch.float32,
        device=device,
    )
    majority_accuracy = train_label_counts[0] / total_training_examples

    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Training label counts: {dict(train_label_counts)}")
    print(f"Class weights: {class_weights.cpu().tolist()}")
    print(f"Always-ham training baseline: {majority_accuracy:.2%}")

    embedding_dim = 64
    mean_result = train_model(
        "MeanEmbedding",
        lambda: MeanEmbeddingClassifier(
            len(vocabulary),
            embedding_dim,
            2,
        ),
        datasets,
        class_weights,
        device,
    )
    rnn_result = train_model(
        "RNN",
        lambda: RNNTextClassifier(
            len(vocabulary),
            embedding_dim,
            hidden_size=64,
            num_classes=2,
        ),
        datasets,
        class_weights,
        device,
    )

    print("\nFinal comparison")
    for result in (mean_result, rnn_result):
        print(
            f"{result['name']:13s} | "
            f"Parameters: {result['parameters']:,} | "
            f"Validation F1: {result['validation_f1']:.2%} | "
            f"Test Accuracy: {result['accuracy']:.2%} | "
            f"Spam Precision: {result['precision']:.2%} | "
            f"Spam Recall: {result['recall']:.2%} | "
            f"Spam F1: {result['f1']:.2%}"
        )


if __name__ == "__main__":
    main()
