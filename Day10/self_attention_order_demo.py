import torch

from multi_head_attention import MultiHeadSelfAttention

torch.manual_seed(42)

model = MultiHeadSelfAttention(
    d_model=16,
    num_heads=4,
)

model.eval()

x = torch.randn(1,4,16)

permutation = torch.tensor([1,3,0,2])

with torch.no_grad():
    original_output, _ = model(x)

    permuted_input = x[:, permutation, :]

    permuted_output, _ = model(permuted_input)

    expected_output = original_output[:, permutation,:]


print("Original output shape:", original_output.shape)
print("Permuted output shape:", permuted_output.shape)

print(
    "Maximum difference:",
    (permuted_output - expected_output).abs().max().item(),
)

print(
    "Outputs follow the same permutation:",
    torch.allclose(
        permuted_output,
        expected_output,
        atol=1e-6,
    ),
)