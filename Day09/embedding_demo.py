import torch
import torch.nn as nn

token_ids = torch.tensor(
    [
        [2,3,4,5],
        [3,5,0,0],
        [2,4,6,0],
    ]
)

embedding = nn.Embedding(
    num_embeddings=7,
    embedding_dim=4,
    padding_idx=0,
)

embedded = embedding(token_ids)

print("Token IDs shape:", token_ids.shape)
print("Embedded shape:", embedded.shape)

print("\nPAD vector:")
print(embedding.weight[0])

print("\nVector of token 2 in sentence A:")
print(embedded[0, 0])

print("\nVector of token 2 in sentence C:")
print(embedded[2, 0])

print("\nAre the two token-2 vectors equal?")
print(torch.allclose(embedded[0, 0], embedded[2, 0]))