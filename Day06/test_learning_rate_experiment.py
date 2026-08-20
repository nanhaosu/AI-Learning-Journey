import unittest
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_rate_experiment import evaluate, train_one_epoch


class LearningRateExperimentTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        images = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        labels = torch.tensor([0, 0, 1, 1])
        self.loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        self.loss_function = nn.CrossEntropyLoss()

    def test_train_one_epoch_updates_model_parameters(self):
        torch.manual_seed(42)
        model = nn.Linear(1, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss, accuracy = train_one_epoch(
            model, optimizer, self.loader, self.loss_function, self.device
        )

        after = list(model.parameters())
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, after)))
        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_evaluate_does_not_update_model_parameters(self):
        model = nn.Linear(1, 2)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss, accuracy = evaluate(
            model, self.loader, self.loss_function, self.device
        )

        after = list(model.parameters())
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, after)))
        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
