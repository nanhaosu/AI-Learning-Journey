import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        if d_model % num_heads !=0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.key_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.value_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
        )

    def split_heads(self,tensor):
        batch_size = tensor.size(0)
        sequence_length = tensor.size(1)

        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        tensor = tensor.transpose(1,2)

        return tensor

    def merge_heads(self,tensor):
        batch_size = tensor.size(0)
        sequence_length = tensor.size(2)

        tensor = tensor.transpose(1,2)

        tensor = tensor.contiguous().reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )

        return tensor

    def forward(
        self,
        x,
        key_valid_mask=None,
        causal=False,
    ):
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attention_scores = (
            query @ key.transpose(-2, -1)
        )

        scaled_scores = (
            attention_scores / (self.head_dim ** 0.5)
        )

        if key_valid_mask is not None:
            key_valid_mask = key_valid_mask.to(
                device=scaled_scores.device,
                dtype=torch.bool,
            )

            expanded_mask = key_valid_mask[:, None, None, :]

            scaled_scores = scaled_scores.masked_fill(
                ~expanded_mask,
                float("-inf"),
            )

        if causal:
            sequence_length = x.size(1)

            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=scaled_scores.device,
                    dtype=torch.bool,
                )
            )

            expanded_causal_mask = causal_mask[
                None,
                None,
                :,
                :,
            ]

            scaled_scores = scaled_scores.masked_fill(
                ~expanded_causal_mask,
                float("-inf"),
            )

        attention_weights = torch.softmax(
            scaled_scores,
            dim=-1,
        )

        context = attention_weights @ value

        context = self.merge_heads(context)

        output = self.output_projection(context)

        return output, attention_weights