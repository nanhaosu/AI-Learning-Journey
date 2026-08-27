import unittest

import torch

class TransformerEncoderBlockTest(unittest.TestCase):
    def test_encoder_block_preserves_model_shape(self):
        try:
            from transformer_encoder import (
                TransformerEncoderBlock,
            )
        except (ModuleNotFoundError,ImportError):
            self.fail(
                "TransformerEncoderBlock "
                "has not been implemented yet"
            )

        block = TransformerEncoderBlock(
            d_model=32,
            num_heads=4,
            ff_dim=128,
            dropout=0.0,
        )

        x = torch.randn(2,5,32)

        output, attention_weights = block(x)

        self.assertEqual(
            output.shape,
            torch.Size([2,5,32])
        )

        self.assertEqual(
            attention_weights.shape,
            torch.Size([2,4,5,5])
        )

if __name__ == "__main__":
    unittest.main()