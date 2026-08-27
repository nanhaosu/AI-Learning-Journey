import torch
import torch.nn as nn

from transformer_encoder import(
    TransformerEncoderBlock,
)

def masked_mean_pool(
        token_features,
        key_valid_mask,
):
    expanded_mask = key_valid_mask.unsqueeze(-1)

    expanded_mask = expanded_mask.to(
        device=token_features.device,
        dtype=token_features.dtype,
    )

    valid_feature_sum = (
        token_features * expanded_mask
    ).sum(dim=1)

    valid_token_count = expanded_mask.sum(
        dim=1
    )

    pooled = (
        valid_feature_sum / valid_token_count
    )

    return pooled

class TransformerTextClassifier(nn.Module):
    def __init__(
            self,
            vocabulary_size,
            num_classes,
            d_model,
            num_heads,
            ff_dim,
            num_layers,
            max_length,
            padding_idx=0,
            dropout=0.0,
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.max_length = max_length

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            d_model,
        )

        self.input_dropout = nn.Dropout(
            dropout
        )

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            d_model
        )

        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(self, token_ids):
        sequence_length = token_ids.size(1)

        if sequence_length > self.max_length:
            raise ValueError(
                "sequence length "
                f"{sequence_length} exceeds "
                f"max_length {self.max_length}"
            )

        key_valid_mask = (
            token_ids != self.padding_idx
        )

        has_valid_token = key_valid_mask.any(
            dim=1
        )

        if not has_valid_token.all():
            raise ValueError(
                "each sequence must contain "
                "at least one non-padding token"
            )

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(
            token_ids
        )

        position_vectors = self.position_embedding(
            position_ids
        )

        x = token_vectors + position_vectors

        x = self.input_dropout(x)

        layer_attention_weights = []

        for block in self.encoder_blocks:
            x, attention_weights = block(
                x,
                key_valid_mask=key_valid_mask,
            )

            layer_attention_weights.append(
                attention_weights
            )

        x = self.final_norm(x)

        pooled = masked_mean_pool(
            x,
            key_valid_mask,
        )

        logits = self.classifier(pooled)

        return logits, layer_attention_weights


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