# Day10: Scaled Dot-Product and Multi-Head Self-Attention

## Goal

Day10 built attention from tensor operations instead of treating it as a black-box layer. The work covered:

- the roles of Query, Key, and Value;
- dot-product similarity, scaling, and softmax;
- attention tensor shapes and matrix multiplication;
- padding masks and causal masks;
- splitting and merging multiple attention heads;
- a reusable multi-head self-attention module;
- test-driven implementation of output shapes and masking behavior;
- the lack of position information in self-attention without positional encoding.

## Query, Key, and Value

For each token, the model learns three representations:

```text
Query: what information is this position looking for?
Key:   what information can this position match?
Value: what content will this position provide?
```

Query and Key determine the attention weights. The weights are then used to combine the Value vectors.

For one query and several keys:

```text
scores  = Q @ K^T
weights = softmax(scores / sqrt(d_k))
output  = weights @ V
```

The operation with `V` is matrix multiplication rather than element-wise multiplication because each Query combines information from every available Value.

## Why Scale the Scores?

When the Key dimension `d_k` is large, unscaled dot products can also become large. Softmax may then produce probabilities that are extremely close to zero or one, causing small gradients and unstable optimization.

Scaled dot-product attention uses:

```text
scaled_scores = scores / sqrt(d_k)
```

This controls the magnitude of the logits before softmax.

## Tensor Shapes

For:

```text
batch size:       B
sequence length:  T
model dimension:  D
number of heads:  H
head dimension:   D_h = D / H
```

the main shapes are:

```text
input x:             [B,T,D]
projected Q/K/V:     [B,T,D]
split Q/K/V:         [B,H,T,D_h]
attention scores:    [B,H,T,T]
attention weights:   [B,H,T,T]
per-head context:    [B,H,T,D_h]
merged context:      [B,T,D]
final output:        [B,T,D]
```

Within one head:

```text
Q:   [T,D_h]
K^T: [D_h,T]

[T,D_h] @ [D_h,T] -> [T,T]
```

The final two attention dimensions mean `Query position x Key position`. Each row describes how one Query distributes attention across all Keys.

## Multi-Head Attention

The implementation first applies three learned linear projections and then splits their feature dimension into heads:

```text
[B,T,D]
-> reshape [B,T,H,D_h]
-> transpose [B,H,T,D_h]
```

Each head computes attention independently and can learn relationships in a different feature subspace. After attention, the heads are merged:

```text
[B,H,T,D_h]
-> transpose [B,T,H,D_h]
-> reshape [B,T,D]
```

The final output projection is another learned linear layer. It mixes information from the concatenated heads; it does not change the `[B,T,D]` shape.

## Masks

Masks are applied to the scaled scores before softmax. A blocked score is replaced by negative infinity:

```text
softmax(-inf) = 0
```

Applying the mask before softmax also lets the remaining valid positions be normalized so that each attention row still sums to one.

### Padding Mask

A padding mask prevents attention from using positions that were added only to equalize sequence lengths.

The input mask has shape:

```text
[B,T]
```

It is expanded to:

```text
[B,1,1,T]
```

and broadcast across all heads and all Query positions. The final dimension corresponds to Key positions, so invalid Key columns receive zero attention weight.

### Causal Mask

A causal mask prevents a Query from accessing future Keys during autoregressive generation:

```text
[[T, F, F, F],
 [T, T, F, F],
 [T, T, T, F],
 [T, T, T, T]]
```

Its shape is expanded from `[T,T]` to `[1,1,T,T]`, allowing the same causal rule to broadcast across batches and heads.

Padding masks answer "which tokens are real?" Causal masks answer "which positions are currently visible?"

## Position Information

Self-attention without positional information is permutation equivariant. If the input tokens are reordered, the outputs follow the same reordering:

```text
input:  [A,B,C,D]
output: [oA,oB,oC,oD]

input:  [B,D,A,C]
output: [oB,oD,oA,oC]
```

The experiment measured a maximum difference of approximately `1.19e-7` between the permuted output and the correspondingly permuted original output, which is normal floating-point error.

This behavior shows that attention can model content relationships but does not know absolute token positions by itself. Positional information will be added when constructing a Transformer.

## Test-Driven Implementation

The reusable module was built through small red-green cycles:

1. A shape test failed because the module did not exist.
2. The class and projections were created.
3. Q, K, and V projection, head splitting, attention, head merging, and output projection were implemented.
4. The shape test passed.
5. A padding-mask test failed because the interface did not yet support a mask.
6. Padding masking was implemented and the test passed.
7. A causal-mask test failed because the interface did not yet support causal attention.
8. Causal masking was implemented and all tests passed.

The tests verify:

- output shape `[B,T,D]`;
- attention-weight shape `[B,H,T,T]`;
- zero weights for padding Keys;
- zero weights for future Keys under causal attention;
- attention rows summing to one after masking.

## Errors and Lessons

1. Applying softmax to `Key` instead of the scaled attention scores uses the wrong tensor and cannot produce Query-to-Key weights.
2. Attention weights must be multiplied with `Value` using matrix multiplication, not element-wise multiplication.
3. A padding mask applies to Key columns, even though it is shared across all Query rows.
4. A mask of `[B,T]` cannot be used directly with scores of `[B,H,T,T]`; it must be reshaped to align batch and Key dimensions correctly.
5. `num_head` and `num_heads` are different Python keyword names.
6. A trailing comma after a function call can wrap a returned pair inside a one-element tuple and break unpacking.
7. Correct shapes alone do not prove correct masking or normalization, so behavioral tests are necessary.
8. Attention without positional information follows token permutations rather than recognizing absolute order.

## Files

- `scaled_dot_product_demo.py`: scaled dot-product attention, padding masking, row normalization, and masked-column inspection;
- `multi_head_shape_demo.py`: learned Q/K/V projections and head-splitting shapes;
- `multi_head_attention.py`: reusable multi-head self-attention with optional padding and causal masks;
- `test_multi_head_attention.py`: shape, padding-mask, causal-mask, and normalization tests;
- `self_attention_order_demo.py`: permutation-equivariance experiment without positional information.

## Run

From the `Day10` directory:

```bash
python scaled_dot_product_demo.py
python multi_head_shape_demo.py
python self_attention_order_demo.py
python -m unittest test_multi_head_attention.py
```

The verified test result was:

```text
Ran 3 tests
OK
```
