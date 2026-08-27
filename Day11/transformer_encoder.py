import sys
from pathlib import Path

import torch.nn as nn

day10_directory = (
    Path(__file__).resolve().parent.parent / "Day10"
)

if str(day10_directory) not in sys.path:
    sys.path.insert(
        0,
        str(day10_directory),
    )

from multi_head_attention import MultiHeadSelfAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            ff_dim,
            dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.attention_output_dropout = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim,d_model),
        )

        self.feed_forward_output_dropout = nn.Dropout(
            dropout
        )

    def forward(
            self,
            x,
            key_valid_mask=None,
    ):
        normalized_x = self.norm1(x)

        attention_output, attention_weights =(
            self.attention(
                normalized_x,
                key_valid_mask=key_valid_mask,
            )
        )

        x = (
            x
            + self.attention_output_dropout(
                attention_output
            )
        )

        normalized_x = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        x = (
            x
            + self.feed_forward_output_dropout(
                feed_forward_output
            )
        )

        return x , attention_weights