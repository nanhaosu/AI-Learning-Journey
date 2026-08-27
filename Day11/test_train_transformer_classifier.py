import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

day09_directory = (
    Path(__file__).resolve().parent.parent / "Day09"
)

if str(day09_directory) not in sys.path:
    sys.path.insert(
        0,
        str(day09_directory),
    )

from text_data import SmsDataset, collate_batch
from transformer_classifier import (
    TransformerTextClassifier,
)


class TrainTransformerClassifierTest(unittest.TestCase):
    def test_training_epoch_updates_parameters(self):
        try:
            from train_transformer_classifier import (
                run_epoch,
            )
        except (ModuleNotFoundError, ImportError):
            self.fail(
                "Transformer training loop "
                "has not been implemented yet"
            )

        vocabulary = {
            "<PAD>": 0,
            "<UNK>": 1,
            "free": 2,
            "home": 3,
            "prize": 4,
            "see": 5,
        }

        dataset = SmsDataset(
            [
                ("free prize", 1),
                ("see you home", 0),
                ("free free prize", 1),
                ("home home", 0),
            ],
            vocabulary,
            max_length=4,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_batch,
        )

        model = TransformerTextClassifier(
            vocabulary_size=len(vocabulary),
            num_classes=2,
            d_model=8,
            num_heads=2,
            ff_dim=16,
            num_layers=1,
            max_length=4,
            padding_idx=0,
            dropout=0.0,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

        weight_before = (
            model.classifier.weight
            .detach()
            .clone()
        )

        run_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            optimizer=optimizer,
        )

        weight_after = (
            model.classifier.weight
            .detach()
        )

        self.assertFalse(
            torch.equal(
                weight_before,
                weight_after,
            )
        )

    def test_evaluation_returns_dataset_metrics(self):
        from train_transformer_classifier import (
            run_epoch,
        )

        vocabulary = {
            "<PAD>": 0,
            "<UNK>": 1,
            "free": 2,
            "home": 3,
            "prize": 4,
            "see": 5,
        }

        dataset = SmsDataset(
            [
                ("free prize", 1),
                ("see home", 0),
                ("free free", 1),
                ("home home", 0),
            ],
            vocabulary,
            max_length=4,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_batch,
        )

        model = TransformerTextClassifier(
            vocabulary_size=len(vocabulary),
            num_classes=2,
            d_model=8,
            num_heads=2,
            ff_dim=16,
            num_layers=1,
            max_length=4,
            padding_idx=0,
            dropout=0.0,
        )

        average_loss, metrics = run_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )

        self.assertIsInstance(
            average_loss,
            float,
        )

        self.assertEqual(
            (
                metrics["tp"]
                + metrics["fp"]
                + metrics["fn"]
                + metrics["tn"]
            ),
            4,
        )

        for metric_name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            self.assertGreaterEqual(
                metrics[metric_name],
                0.0,
            )
            self.assertLessEqual(
                metrics[metric_name],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()