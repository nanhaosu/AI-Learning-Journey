import torch
import torch.nn as nn

torch.manual_seed(42)

token_ids = torch.tensor([
    [2,3,2,4],
    [2,2,3,4],
])

vocabulary_size = 10
d_model = 6
max_length = 8

token_embedding = nn.Embedding(
    vocabulary_size,
    d_model,
)

position_embedding = nn.Embedding(
    max_length,
    d_model,
)

token_vectors = token_embedding(token_ids)

sequence_length = token_ids.size(1)

position_ids = torch.arange(sequence_length)

position_vectors = position_embedding(position_ids)

combined_vectors = (
    token_vectors + position_vectors
)

print("Token IDs:", token_ids.shape)
print("Token vectors:", token_vectors.shape)
print("Position IDs:", position_ids.shape)
print("Position vectors:", position_vectors.shape)
print("Combined vectors:", combined_vectors.shape)

print(
    "Same token vectors:",
    torch.allclose(
        token_vectors[0, 0],
        token_vectors[0, 2],
    ),
)

print(
    "Same combined vectors:",
    torch.allclose(
        combined_vectors[0, 0],
        combined_vectors[0, 2],
    ),
)