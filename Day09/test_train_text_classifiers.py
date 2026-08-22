import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_data import SmsDataset, collate_batch
from text_models import MeanEmbeddingClassifier
from train_text_classifiers import compute_binary_metrics, run_epoch


class TrainTextClassifiersTest(unittest.TestCase):
    def test_binary_metrics_count_all_confusion_cases(self):
        metrics = compute_binary_metrics(
            predictions=[1, 0, 1, 0],
            labels=[1, 1, 0, 0],
        )

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_training_epoch_updates_model_parameters(self):
        vocabulary = {
            "<PAD>": 0,
            "<UNK>": 1,
            "free": 2,
            "home": 3,
        }
        dataset = SmsDataset(
            [
                ("free free", 1),
                ("home home", 0),
                ("free home", 1),
                ("home", 0),
            ],
            vocabulary,
            max_length=4,
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_batch,
        )
        model = MeanEmbeddingClassifier(4, 8, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        weight_before = model.classifier.weight.detach().clone()

        run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer=optimizer,
        )

        self.assertFalse(
            torch.equal(weight_before, model.classifier.weight.detach())
        )


if __name__ == "__main__":
    unittest.main()
