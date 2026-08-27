# Day11: Building a Minimal Transformer Encoder

## Goal

Day11 assembled the attention mechanism from Day10 into a minimal Transformer encoder and applied it to the SMS spam-classification task from Day09. The work covered:

- separate learned token and position embeddings;
- LayerNorm and position-wise feed-forward networks;
- Pre-LN Transformer encoder blocks with residual connections;
- stacked encoder blocks and complete tensor-shape tracking;
- padding masks inside attention and masked mean pooling after the encoder;
- explicit sequence-length and all-padding validation;
- class-weighted Transformer training with validation-F1 checkpoint selection;
- a comparison with the Day09 Mean Embedding and RNN classifiers;
- the difference between bidirectional encoder attention and causal decoder attention.

## From Token IDs to Positioned Vectors

A token ID identifies a row in the token Embedding table. Repeated occurrences of the same token therefore retrieve the same content vector. Token Embedding alone does not identify where the token occurs.

Day11 uses two independent learned tables:

```text
Token Embedding weights:    [vocabulary_size,d_model]
Position Embedding weights: [max_length,d_model]
```

For a batch of token IDs:

```text
token_ids:             [B,T]
token_vectors:         [B,T,D]
position_ids:          [T]
position_vectors:      [T,D]
combined input:        [B,T,D]
```

PyTorch broadcasts the same `[T,D]` position vectors across the batch:

```text
[B,T,D] + [T,D] -> [B,T,D]
```

The content and position dimensions must both equal `d_model` because the vectors are combined by element-wise addition rather than concatenation.

The position experiment confirmed:

```text
same token at two positions -> identical token vectors
same token at two positions -> different combined vectors
```

## LayerNorm

For an input of `[B,T,D]`, `nn.LayerNorm(D)` independently normalizes the `D` features of each token vector:

```text
input:  [B,T,D]
output: [B,T,D]
```

For `[32,20,64]`, there are `32 x 20 = 640` separate token vectors, and each vector computes statistics across its own 64 features.

Unlike BatchNorm, LayerNorm does not aggregate statistics across the batch and does not maintain training-time running means or variances. This makes it suitable for variable-length sequence models and small or changing batch sizes.

## Position-Wise Feed-Forward Network

The feed-forward network processes every token independently with shared parameters:

```text
Linear(D,FF)
-> GELU
-> Dropout
-> Linear(FF,D)
```

Its shape path is:

```text
[B,T,D] -> [B,T,FF] -> [B,T,D]
```

Attention exchanges information between tokens. The FFN instead performs a nonlinear transformation within each token representation. The final projection returns to `d_model` so that the result can be added to the residual branch.

Without GELU or another nonlinear activation, two Linear layers can be algebraically combined into one Linear layer and do not provide the intended nonlinear capacity.

## Pre-LN Transformer Encoder Block

Day11 implements a Pre-LN encoder block:

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

The first sublayer lets tokens exchange information. The second processes each updated token independently. Both sublayers return `[B,T,D]`, which is required for residual addition.

### Complete Shape Path

For:

```text
x:         [B,T,D]
heads:     H
head_dim:  D_h = D / H
FFN size:  FF
```

the encoder block uses:

```text
x                               [B,T,D]
LayerNorm(x)                    [B,T,D]

projected Q/K/V                 [B,T,D]
reshape Q/K/V                   [B,T,H,D_h]
transpose Q/K/V                 [B,H,T,D_h]
attention weights               [B,H,T,T]
per-head context                [B,H,T,D_h]
merged attention output         [B,T,D]
first residual output           [B,T,D]

LayerNorm                       [B,T,D]
FFN expanded features           [B,T,FF]
FFN projected features          [B,T,D]
second residual/final output    [B,T,D]
```

The attention-weight dimensions mean:

```text
[batch,head,Query position,Key position]
```

For `[16,30,128]` with eight heads:

```text
split Q/K/V:       [16,8,30,16]
attention weights: [16,8,30,30]
encoder output:    [16,30,128]
```

## Transformer Text Classifier

The classifier stacks two encoder blocks:

```text
token IDs                         [B,T]
-> Token Embedding                [B,T,64]
-> add Position Embedding         [B,T,64]
-> input Dropout                  [B,T,64]
-> Encoder Block 1                [B,T,64]
-> Encoder Block 2                [B,T,64]
-> final LayerNorm                [B,T,64]
-> masked mean pooling            [B,64]
-> Linear(64,2)                   [B,2]
```

The `[B,2]` output contains logits for ham and spam. Softmax is not placed in the model because `CrossEntropyLoss` expects raw logits.

## Why Padding Is Handled Twice

The padding ID is zero, so the classifier derives:

```python
key_valid_mask = token_ids != 0
```

For:

```text
token IDs: [5,8,2,0,0]
mask:      [T,T,T,F,F]
```

the attention mask blocks the final two Key columns for every Query and every head. This prevents real tokens from reading padding as content.

Padding positions may still produce nonzero Query outputs, especially after position embeddings are added. Therefore a second mask is required during classification:

```text
masked mean = sum(valid token features) / number of valid tokens
```

The mask changes from `[B,T]` to `[B,T,1]` and broadcasts across all feature dimensions:

```text
features: [B,T,D]
mask:     [B,T,1]
product:  [B,T,D]
sum:      [B,D]
count:    [B,1]
pooled:   [B,D]
```

The model explicitly rejects an all-padding sequence because it would provide no valid Keys and make the pooling denominator zero.

## Encoder and Decoder Attention

The implemented classifier is encoder-only. It uses a padding mask but no causal mask, so each real Query may attend to all real Keys on both sides:

```text
Encoder Query 2 in a length-6 sequence:
visible Keys = 0,1,2,3,4,5
```

A GPT-style decoder uses causal self-attention:

```text
Decoder Query 2:
visible Keys = 0,1,2
```

Bidirectional encoder attention is appropriate for classification and representation learning. Causal decoder attention prevents future-token leakage during next-token prediction.

The original encoder-decoder Transformer also includes cross-attention: decoder Queries read Keys and Values produced by the encoder.

## Dataset and Training Setup

The experiment reused Day09's SMS pipeline and identical data split:

```text
training:   4,458
validation:   556
test:         560
vocabulary: 3,796
```

Only training texts contributed vocabulary statistics.

Model and optimization settings:

```text
d_model:            64
attention heads:     4
head dimension:     16
FFN dimension:     128
encoder blocks:      2
maximum length:     50
dropout:           0.1
optimizer:       AdamW
learning rate:   0.001
epochs:              8
loss: weighted cross-entropy
checkpoint metric: validation spam F1
trainable parameters: 313,346
```

The class weights were:

```text
ham:  0.5773
spam: 3.7337
```

An always-ham predictor would reach 86.61% training accuracy while having zero spam recall and F1, so accuracy alone is not sufficient.

## Training Results

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Spam Precision | Spam Recall | Spam F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4047 | 79.54% | 0.2356 | 90.47% | 58.97% | 93.24% | 72.25% |
| 2 | 0.1996 | 91.99% | 0.1612 | 97.30% | 86.42% | 94.59% | 90.32% |
| 3 | 0.1383 | 94.86% | 0.1614 | 93.17% | 67.31% | 94.59% | 78.65% |
| 4 | 0.0854 | 97.08% | 0.1574 | 98.74% | 95.89% | 94.59% | **95.24%** |
| 5 | 0.0785 | 97.47% | 0.1468 | 97.12% | 85.37% | 94.59% | 89.74% |
| 6 | 0.0490 | 98.43% | 0.1560 | 95.86% | 78.65% | 94.59% | 85.89% |
| 7 | 0.0358 | 98.86% | 0.1664 | 97.84% | 89.74% | 94.59% | 92.11% |
| 8 | 0.0257 | 99.17% | 0.1711 | 97.48% | 87.50% | 94.59% | 90.91% |

The epoch-4 state was selected because validation spam F1, not training loss or validation loss, was the checkpoint criterion.

Final test result:

```text
Test Loss:      0.2556
Test Accuracy:  96.43%
Spam Precision: 86.84%
Spam Recall:    86.84%
Spam F1:        86.84%
TP=66, FP=10, FN=10, TN=474
```

## Comparison with Day09

| Model | Parameters | Best Validation F1 | Test F1 |
| --- | ---: | ---: | ---: |
| Mean Embedding | 243,074 | 84.47% | 83.33% |
| RNN | 251,394 | 89.66% | 84.81% |
| Transformer | 313,346 | 95.24% | 86.84% |

The Transformer test F1 was 2.03 percentage points above the RNN result. This single experiment does not prove that Transformers are universally better:

- the Transformer has about 62,000 more parameters;
- only one random seed was tested;
- hyperparameters were not tuned equally for every model;
- the validation set contains only 556 messages;
- selecting the best of eight checkpoints introduces validation-selection variance.

The gap between 95.24% validation F1 and 86.84% test F1 demonstrates why the test set must remain untouched until model selection is complete.

Training loss continued to fall after epoch 4 while validation F1 fluctuated and validation loss later rose, indicating growing overfitting. The epoch-8 model was therefore not selected despite having the lowest training loss and highest training accuracy.

## Test-Set Leakage

Evaluating the test set after every epoch and choosing the highest test-F1 checkpoint would indirectly tune the experiment to the test set. No test gradients are required for this leakage: using test feedback for model selection already turns the test set into another validation set and makes the reported result optimistic.

Correct responsibilities remain:

```text
training set   -> update parameters
validation set -> choose checkpoints and hyperparameters
test set       -> one final evaluation after all choices
```

## Tests

The Day11 suite checks:

- encoder input/output and attention-weight shapes;
- classifier logits and per-layer attention shapes;
- zero attention weight for padding Keys;
- masked mean pooling with hand-derived expected values;
- rejection beyond the maximum position length;
- acceptance exactly at `max_length`;
- rejection of all-padding sequences;
- training parameter updates after backward and optimizer step;
- epoch-level loss and confusion-metric aggregation.

## Errors and Lessons

1. `text_transformer_encoder.py` was initially used instead of the intended test filename `test_transformer_encoder.py`, so unittest could not find the module.
2. Running unittest from the wrong directory also prevents Python from discovering a local test module.
3. The first position demo mistakenly reused one token Embedding table for both token IDs and position IDs. Plausible output did not prove correct logic; separate tables were required.
4. Token and position vectors must share `d_model` because they are added element by element.
5. Splitting heads first produces the intermediate `[B,T,H,D_h]`, but attention uses the transposed `[B,H,T,D_h]` representation.
6. Attention weights have shape `[B,H,T,T]`, not `[B,H,T,D_h]`; their last dimensions are Query and Key positions.
7. Padding must be removed both from attention Keys and from final pooling.
8. Input validation produces clearer errors than allowing an internal Embedding index failure or a NaN from an all-padding sequence.
9. Shape tests alone are insufficient: masking, pooling, boundaries, and actual parameter updates require behavioral tests.

## Files

- `position_embedding_demo.py`: separate token and position Embedding experiment;
- `transformer_encoder.py`: reusable Pre-LN Transformer encoder block;
- `transformer_classifier.py`: stacked encoder text classifier and masked mean pooling;
- `train_transformer_classifier.py`: SMS data reuse, weighted training, validation-F1 checkpointing, testing, and Day09 comparison;
- `test_transformer_encoder.py`: encoder and attention shape test;
- `test_transformer_classifier.py`: pooling, shape, padding-mask, length-boundary, and all-padding tests;
- `test_train_transformer_classifier.py`: parameter-update and epoch-metric tests.

## Run

From the `Day11` directory:

```bash
python position_embedding_demo.py
python -m unittest test_transformer_encoder.py test_transformer_classifier.py test_train_transformer_classifier.py
python train_transformer_classifier.py
```
