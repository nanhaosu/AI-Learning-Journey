import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_cifar10 import CNN
from analyze_predictions import predict_batch


class Cifar10CnnTest(unittest.TestCase):
    def test_cnn_outputs_ten_logits_per_image(self):
        model = CNN()
        images = torch.randn(4, 3, 32, 32)

        logits = model(images)

        self.assertEqual(logits.shape, torch.Size([4, 10]))

    def test_predict_batch_returns_probabilities_and_confidence(self):
        model = CNN()
        images = torch.randn(4, 3, 32, 32)

        probabilities, predictions, confidence = predict_batch(model, images)

        self.assertEqual(probabilities.shape, torch.Size([4, 10]))
        self.assertEqual(predictions.shape, torch.Size([4]))
        self.assertEqual(confidence.shape, torch.Size([4]))
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
