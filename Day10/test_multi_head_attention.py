import unittest

import torch

class MultiHeadAttentionTest(unittest.TestCase):
    def test_forward_returns_expected_shapes(self):
        try:
            from multi_head_attention import MultiHeadSelfAttention
        except (ModuleNotFoundError,ImportError):
            self.fail(
                "MultiHeadSelfAttention has not been implemented yet"
            )

        model = MultiHeadSelfAttention(
            d_model=64,
            num_heads=8,
        )

        x =torch.randn(2,5,64)

        output, attention_weights = model(x)

        self.assertEqual(
            output.shape,
            torch.Size([2,5,64]),
        )
        self.assertEqual(
            attention_weights.shape,
            torch.Size([2,8,5,5])
        )

    def test_padding_mask_blocks_invalid_keys(self):
        from multi_head_attention import MultiHeadSelfAttention

        model = MultiHeadSelfAttention(
            d_model=64,
            num_heads=8,
        )

        x = torch.randn(2,4,64)

        key_valid_mask = torch.tensor([
            [True,True,True,False],
            [True,True,False,False]
        ])

        _, attention_weights = model(
            x,
            key_valid_mask=key_valid_mask,
        )

        self.assertTrue(
            torch.all(attention_weights[0,:,:,3]== 0)
        )

        self.assertTrue(
            torch.all(attention_weights[1,:,:,2:]== 0)
        )

        row_sums = attention_weights.sum(dim=-1)

        self.assertTrue(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-6,
            )
        )

    def test_causal_mask_blocks_future_keys(self):
        from multi_head_attention import MultiHeadSelfAttention

        model = MultiHeadSelfAttention(
            d_model=32,
            num_heads=4,
        )

        x =torch.randn(1,4,32)

        _, attention_weights = model(
            x,
            causal=True,
        )

        for query_index in range(4):
            future_weights = attention_weights[
                0,
                :,
                query_index,
                query_index +1:,
            ]

            self.assertTrue(
                torch.all(future_weights == 0)
            )

        row_sums = attention_weights.sum(dim=-1)

        self.assertTrue(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol = 1e-6,
            )
        )


if __name__=="__main__":
    unittest.main()