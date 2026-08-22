# Day09: Text Embeddings, RNNs, and SMS Classification

## Goal

Day09 moved from fixed-size image tensors to variable-length text sequences. The work covered the complete path from raw strings to a trained sequence classifier:

- tokenization, vocabularies, token IDs, unknown tokens, and padding;
- train-only vocabulary construction and preprocessing leakage;
- learned word embeddings;
- variable-length batching and valid-sequence lengths;
- recurrent hidden states and order-sensitive processing;
- a controlled comparison between mean-pooled embeddings and an RNN;
- class imbalance, weighted loss, precision, recall, F1, and confusion counts.

## Dataset

The main experiment used the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms), a CC BY 4.0 dataset of 5,574 labeled SMS messages.

```text
ham:  4,827
spam:   747
total: 5,574
```

The data was split independently within each class:

```text
training:   4,458
validation:   556
test:         560
```

Only training texts were used to count tokens and build the 3,796-token vocabulary. This prevents validation and test vocabulary statistics from leaking into preprocessing.

## Text Preprocessing

The text pipeline is:

```text
raw text
→ lowercase tokenization
→ vocabulary lookup
→ token IDs
→ truncation to at most 50 tokens
→ dynamic batch padding
→ Embedding vectors
```

Two special vocabulary entries are reserved:

```text
<PAD> → 0
<UNK> → 1
```

`<PAD>` fills unused batch positions, while `<UNK>` represents tokens that are absent from the training vocabulary.

### Token IDs Are Indices

A token ID identifies one row of the Embedding table. Its numerical value does not represent magnitude or importance. For example, token ID 9 is not inherently more meaningful than token ID 3.

### Dynamic Padding

Messages in a batch have different lengths. The custom `collate_batch` pads only to the longest sequence in the current batch:

```text
lengths:   [5, 41, 25, 6, 18, 12, 8, 29]
token IDs: [8, 41]
```

The global maximum length is 50, but this batch needs only 41 positions. Dynamic padding avoids unnecessary work compared with padding every batch to 50.

## Embeddings

An Embedding layer stores one learned vector for each vocabulary item:

```python
nn.Embedding(
    num_embeddings=vocabulary_size,
    embedding_dim=64,
    padding_idx=0,
)
```

Its shape transformation is:

```text
token IDs: [batch, sequence_length]
embedded:  [batch, sequence_length, embedding_dim]
```

For example:

```text
[32,50] → Embedding(dim=64) → [32,50,64]
```

Embedding is a learned lookup table, not a direct interpretation of the integer IDs as continuous values. Repeated occurrences of the same token select the same row. The padding row stays zero and is not updated during training.

## RNN Fundamentals

An RNN updates a hidden state at every sequence position:

```text
h_t = tanh(x_t W_x + h_(t-1) W_h + b)
```

- `x_t` is the current token embedding;
- `h_(t-1)` is the previous hidden state;
- `h_t` summarizes the sequence up to the current position;
- the same recurrent parameters are shared across all positions.

With `batch_first=True`:

```text
embedded: [batch, sequence_length, embedding_dim]
output:   [batch, sequence_length, hidden_size]
h_n:      [num_layers, batch, hidden_size]
```

For an input of `[3,4,4]`, hidden size 6, and one recurrent layer:

```text
output: [3,4,6]
h_n:    [1,3,6]
```

### Selecting the Last Valid State

Padding embeddings are zero, but an RNN can still update its hidden state through the previous state, recurrent weights, and bias. Therefore, the final padded position is not necessarily the last meaningful state.

For real lengths `[4,2,3]`, the final valid indices are:

```text
lengths - 1 = [3,1,2]
```

The classifier selects:

```python
last_hidden = output[
    batch_indices,
    lengths - 1,
]
```

This produces one valid final representation per message.

## Models

### Mean Embedding Classifier

```text
token IDs
→ Embedding
→ sum valid token vectors and divide by true length
→ Linear(64,2)
→ logits
```

This model can learn which words are associated with spam, but vector addition is order-independent. Two messages containing the same tokens in different orders receive identical mean representations.

### RNN Classifier

```text
token IDs
→ Embedding
→ RNN
→ last valid hidden state
→ Linear(64,2)
→ logits
```

The recurrent state depends on the sequence of earlier updates, so changing token order can change the final representation.

### Parameter Counts

```text
MeanEmbedding: 243,074
RNN:           251,394
Difference:      8,320
```

The models share the same 3,796-by-64 Embedding table. The RNN adds recurrent input weights, recurrent hidden weights, and two bias vectors. Therefore, the comparison does not isolate word order perfectly: the RNN also has additional parameters and nonlinear computation.

## Class Imbalance

The training labels were:

```text
ham:  3,861
spam:   597
```

An always-ham classifier would achieve 86.61% training accuracy while having zero spam recall and zero spam F1. Accuracy alone is therefore misleading.

Weighted cross-entropy used:

```text
ham weight:  0.5773
spam weight: 3.7337
```

A spam classification error therefore contributed about 6.47 times as much loss as a ham error, matching the approximate class-frequency ratio.

### Metrics

Treating spam as the positive class:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

- Precision measures how many predicted spam messages are truly spam.
- Recall measures how many real spam messages were found.
- FP represents a normal message incorrectly flagged as spam.
- FN represents a spam message missed as ham.

The best model checkpoint was selected by validation spam F1 rather than accuracy.

## Experiment Setup

- Vocabulary: training tokens with frequency at least 2, maximum size 10,000
- Maximum sequence length: 50
- Embedding dimension: 64
- RNN hidden size: 64
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 8
- Loss: class-weighted cross-entropy
- Device: CUDA

## Results

### Mean Embedding

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Spam Precision | Spam Recall | Spam F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6734 | 72.66% | 0.6130 | 79.14% | 36.18% | 74.32% | 48.67% |
| 2 | 0.5612 | 84.28% | 0.4870 | 84.89% | 46.67% | 94.59% | 62.50% |
| 3 | 0.4196 | 90.26% | 0.3479 | 90.29% | 58.06% | 97.30% | 72.73% |
| 4 | 0.2919 | 94.32% | 0.2501 | 93.35% | 67.62% | 95.95% | 79.33% |
| 5 | 0.2082 | 95.58% | 0.1930 | 94.42% | 72.63% | 93.24% | 81.66% |
| 6 | 0.1604 | 96.39% | 0.1599 | 94.78% | 74.73% | 91.89% | 82.42% |
| 7 | 0.1319 | 97.04% | 0.1392 | 94.96% | 75.56% | 91.89% | 82.93% |
| 8 | 0.1113 | 97.31% | 0.1253 | 95.50% | 78.16% | 91.89% | **84.47%** |

```text
Test Accuracy:  95.00%
Spam Precision: 76.09%
Spam Recall:    92.11%
Spam F1:        83.33%
TP=70, FP=22, FN=6, TN=462
```

### RNN

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Spam Precision | Spam Recall | Spam F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5732 | 64.22% | 0.3833 | 77.52% | 35.91% | 87.84% | 50.98% |
| 2 | 0.2804 | 87.86% | 0.2569 | 91.73% | 64.00% | 86.49% | 73.56% |
| 3 | 0.1626 | 94.39% | 0.2566 | 93.71% | 71.91% | 86.49% | 78.53% |
| 4 | 0.1110 | 96.46% | 0.2378 | 95.14% | 79.01% | 86.49% | 82.58% |
| 5 | 0.0831 | 97.06% | 0.2700 | 95.86% | 84.00% | 85.14% | 84.56% |
| 6 | 0.0570 | 98.34% | 0.2088 | 94.60% | 74.44% | 90.54% | 81.71% |
| 7 | 0.0455 | 98.50% | 0.2328 | 96.94% | 89.04% | 87.84% | 88.44% |
| 8 | 0.0305 | 99.13% | 0.2459 | 97.30% | 91.55% | 87.84% | **89.66%** |

```text
Test Accuracy:  95.71%
Spam Precision: 81.71%
Spam Recall:    88.16%
Spam F1:        84.81%
TP=67, FP=15, FN=9, TN=469
```

## Interpretation

Both learned models substantially exceeded the always-ham baseline by detecting real spam messages. Mean Embedding had higher test recall and found 70 of 76 spam messages, but it falsely flagged 22 normal messages. RNN had higher precision and F1, falsely flagging only 15 normal messages, but it missed three more spam messages.

The RNN result supports the prediction that sequence processing can help, but the test F1 advantage was only 1.48 percentage points. A single run cannot prove that word order caused the improvement or that RNN is universally better. Repeated random seeds, targeted order-sensitive examples, and hyperparameter tuning would be required for a stronger conclusion.

RNN training accuracy reached 99.13% while validation loss fluctuated in later epochs, showing early signs of overfitting or overconfident probability estimates. Its test F1 was higher, but its test cross-entropy loss was also higher than Mean Embedding's loss.

The preferred model depends on deployment cost:

- a warning system may prefer Mean Embedding's higher recall;
- an automatic blocking system may prefer RNN's higher precision, although 81.71% precision is still too low for irreversible deletion without safeguards.

## RNN Limitations

Information and gradients in a vanilla RNN must pass through every intermediate time step. Repeated multiplication can make gradients shrink toward zero or grow uncontrollably:

- vanishing gradients make long-range information difficult to learn;
- exploding gradients make optimization unstable;
- gradient clipping helps explosion but does not solve vanishing gradients.

LSTM and GRU introduce gates to control memory. Attention, studied next, allows positions to access other positions more directly instead of relying only on a sequential hidden-state chain.

## Errors and Lessons

1. Token values were initially at risk of being confused with position indices. Token IDs identify vocabulary rows, while sequence positions use zero-based indexing.
2. The final valid index is `length - 1`, not the token ID value and not the final padded position.
3. A missing `Path` import was caught by the file-loading unit test before the full data pipeline ran.
4. A real spam predicted as ham is a false negative and reduces recall; a real ham predicted as spam is a false positive and reduces precision.
5. Output shape alone cannot prove that a model used the correct data path, so model behavior and invariance tests were also added.

## Files

- `embedding_demo.py`: Embedding lookup, padding row, and repeated-token vectors;
- `tokenization_demo.py`: tokenization, training vocabulary, and unknown-token encoding;
- `rnn_shape_demo.py`: RNN output, hidden-state shapes, and final valid indexing;
- `rnn_classifier_shape_demo.py`: learner-built RNN text-classifier shape path;
- `inspect_sms_data.py`: real dataset inspection and dynamic batch padding;
- `text_data.py`: download, parsing, stratified splitting, vocabulary, encoding, Dataset, and collation;
- `text_models.py`: reusable Mean Embedding and RNN classifiers;
- `train_text_classifiers.py`: class-weighted training, validation-F1 selection, testing, and model comparison;
- `test_text_data.py`: text-pipeline unit tests;
- `test_text_models.py`: model shape and order-invariance tests;
- `test_train_text_classifiers.py`: metric and parameter-update tests.
