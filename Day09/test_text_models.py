import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_models import MeanEmbeddingClassifier, RNNTextClassifier


class TextModelsTest(unittest.TestCase):
    def test_both_models_output_two_logits_per_message(self):
        token_ids = torch.tensor(
            [
                [2, 3, 4],
                [5, 6, 0],
            ]
        )
        lengths = torch.tensor([3, 2])

        mean_logits = MeanEmbeddingClassifier(10, 8, 2)(token_ids, lengths)
        rnn_logits = RNNTextClassifier(10, 8, 12, 2)(token_ids, lengths)

        self.assertEqual(mean_logits.shape, torch.Size([2, 2]))
        self.assertEqual(rnn_logits.shape, torch.Size([2, 2]))

    def test_mean_embedding_is_invariant_to_token_order(self):
        model = MeanEmbeddingClassifier(10, 8, 2)
        token_ids = torch.tensor(
            [
                [2, 3, 4],
                [4, 3, 2],
            ]
        )
        lengths = torch.tensor([3, 3])

        logits = model(token_ids, lengths)

        self.assertTrue(torch.allclose(logits[0], logits[1], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
