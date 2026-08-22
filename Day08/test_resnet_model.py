import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resnet_model import PlainCNN, ResidualBlock, SmallResNet
from compare_plain_and_resnet import run_epoch


class ResNetModelTest(unittest.TestCase):
    def test_residual_blocks_produce_expected_shapes(self):
        images = torch.randn(4, 16, 32, 32)

        same_shape = ResidualBlock(16, 16, stride=1)(images)
        downsampled = ResidualBlock(16, 32, stride=2)(images)

        self.assertEqual(same_shape.shape, torch.Size([4, 16, 32, 32]))
        self.assertEqual(downsampled.shape, torch.Size([4, 32, 16, 16]))

    def test_small_resnet_outputs_logits_without_debug_prints(self):
        model = SmallResNet()
        images = torch.randn(4, 3, 32, 32)
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            logits = model(images)

        self.assertEqual(logits.shape, torch.Size([4, 10]))
        self.assertEqual(captured_output.getvalue(), "")

    def test_plain_cnn_outputs_ten_logits(self):
        model = PlainCNN()
        images = torch.randn(4, 3, 32, 32)

        logits = model(images)

        self.assertEqual(logits.shape, torch.Size([4, 10]))

    def test_training_epoch_updates_parameters(self):
        model = SmallResNet()
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 3, 32, 32),
                torch.tensor([0, 1, 2, 3]),
            ),
            batch_size=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        parameters_before = model.classifier.weight.detach().clone()

        run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer=optimizer,
        )

        self.assertFalse(
            torch.equal(parameters_before, model.classifier.weight.detach())
        )

    def test_evaluation_epoch_does_not_update_parameters(self):
        model = SmallResNet()
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 3, 32, 32),
                torch.tensor([0, 1, 2, 3]),
            ),
            batch_size=4,
        )
        parameters_before = model.classifier.weight.detach().clone()

        run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )

        self.assertTrue(
            torch.equal(parameters_before, model.classifier.weight.detach())
        )


if __name__ == "__main__":
    unittest.main()
