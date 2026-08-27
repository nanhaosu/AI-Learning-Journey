import unittest

import torch

class TransformerClassifierTest(unittest.TestCase):
    def test_masked_mean_ignores_padding_position(self):
        try:
            from transformer_classifier import (
                masked_mean_pool,
            )
        except (ModuleNotFoundError,ImportError):
            self.fail(
                "masked_mean_pool "
                "has not been implemented yet"
            )

        token_features = torch.tensor([
            [
                [1.0,2.0],
                [3.0,4.0],
                [100.0,100.0],
            ],
            [
                [2.0,4.0],
                [60.0,80.0],
                [10.0,12.0],
            ],
        ])

        key_valid_mask = torch.tensor([
            [True,True,False],
            [True,False,False],
        ])

        pooled = masked_mean_pool(
            token_features,
            key_valid_mask,
        )

        expected = torch.tensor([
            [2.0,3.0],
            [2.0,4.0],
        ])

        self.assertTrue(
            torch.allclose(
                pooled,
                expected,
                atol=1e-6,
            )
        )

    def test_classifier_returns_expected_shapes(self):
        from transformer_classifier import (
            TransformerTextClassifier,
        )

        model = TransformerTextClassifier(
            vocabulary_size=20,
            num_classes=3,
            d_model=16,
            num_heads=4,
            ff_dim=64,
            num_layers=2,
            max_length=8,
            padding_idx=0,
            dropout=0.0,
        )

        token_ids = torch.tensor([
            [2,3,4,0,0],
            [5,6,7,8,9],
        ])

        logits, layer_attention_weights = model(
            token_ids
        )

        self.assertEqual(
            logits.shape,
            torch.Size([2,3]),
        )

        self.assertEqual(
            len(layer_attention_weights),
            2,
        )

        self.assertEqual(
            layer_attention_weights[0].shape,
            torch.Size([2,4,5,5]),
        )

        self.assertEqual(
            layer_attention_weights[1].shape,
            torch.Size([2,4,5,5]),
        )

    def test_classifier_masks_padding_keys(self):
        from transformer_classifier import (
            TransformerTextClassifier,
        )

        model = TransformerTextClassifier(
            vocabulary_size=20,
            num_classes=2,
            d_model=16,
            num_heads=4,
            ff_dim=64,
            num_layers=1,
            max_length=8,
            padding_idx=0,
            dropout=0.0,
        )

        token_ids = torch.tensor([
            [2, 3, 0, 0],
        ])

        _, layer_attention_weights = model(
            token_ids
        )

        first_layer_weights = (
            layer_attention_weights[0]
        )

        padding_weights = first_layer_weights[
            0,
            :,
            :,
            2:,
        ]

        self.assertTrue(
            torch.all(padding_weights == 0)
        )

    def test_classifier_rejects_sequence_too_long(self):
        from transformer_classifier import (
            TransformerTextClassifier,
        )

        model = TransformerTextClassifier(
            vocabulary_size=20,
            num_classes=2,
            d_model=16,
            num_heads=4,
            ff_dim=64,
            num_layers=1,
            max_length=4,
            padding_idx=0,
            dropout=0.0,
        )

        token_ids = torch.tensor([
            [2, 3, 4, 5, 6],
        ])

        with self.assertRaisesRegex(
            ValueError,
            "sequence length",
        ):
            model(token_ids)

    def test_classifier_rejects_all_padding_sequence(self):
        from transformer_classifier import (
            TransformerTextClassifier,
        )

        model = TransformerTextClassifier(
            vocabulary_size=20,
            num_classes=2,
            d_model=16,
            num_heads=4,
            ff_dim=64,
            num_layers=1,
            max_length=4,
            padding_idx=0,
            dropout=0.0,
        )

        token_ids = torch.tensor([
            [0, 0, 0, 0],
        ])

        with self.assertRaisesRegex(
            ValueError,
            "at least one non-padding token",
        ):
            model(token_ids)

    def test_classifier_accepts_exact_max_length(self):
        from transformer_classifier import (
            TransformerTextClassifier,
        )

        model = TransformerTextClassifier(
            vocabulary_size=20,
            num_classes=2,
            d_model=16,
            num_heads=4,
            ff_dim=64,
            num_layers=1,
            max_length=4,
            padding_idx=0,
            dropout=0.0,
        )

        token_ids = torch.tensor([
            [2, 3, 4, 5],
        ])

        logits, layer_attention_weights = model(
            token_ids
        )

        self.assertEqual(
            logits.shape,
            torch.Size([1, 2]),
        )

        self.assertEqual(
            layer_attention_weights[0].shape,
            torch.Size([1, 4, 4, 4]),
        )


if __name__ == "__main__":
    unittest.main()