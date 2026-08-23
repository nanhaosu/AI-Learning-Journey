import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    query,
    key,
    value,
    key_valid_mask=None,
):
    d_k = query.size(-1)

    scores = query @ key.transpose(-2,-1)

    scaled_scores = scores / math.sqrt(d_k)

    if key_valid_mask is not None:
        expanded_mask = key_valid_mask.unsqueeze(1)

        scaled_scores = scaled_scores.masked_fill(
            ~expanded_mask,
            float("-inf"),
    )

    attention_weights = F.softmax(scaled_scores,dim=-1)

    output = attention_weights@value

    return output, attention_weights

torch.manual_seed(42)

query = torch.randn(2,4,8)
key = torch.randn(2,4,8)
value = torch.randn(2,4,6)
key_valid_mask = torch.tensor(
    [
        [True, True, True, False],
        [True, True, False, False],
    ]
)
output, attention_weights = scaled_dot_product_attention(
    query,
    key,
    value,
    key_valid_mask,
)

print("Query:            ", query.shape)
print("Key:              ", key.shape)
print("Value:            ", value.shape)
print("Attention scores: ", attention_weights.shape)
print("Output:           ", output.shape)

print("\nSum of each attention row:")
print(attention_weights.sum(dim=-1))
print("\nAttention weights:")
print(attention_weights)

print("\nMasked columns:")
print("Batch 0, key 3:", attention_weights[0, :, 3])
print("Batch 1, key 2:", attention_weights[1, :, 2])
print("Batch 1, key 3:", attention_weights[1, :, 3])
