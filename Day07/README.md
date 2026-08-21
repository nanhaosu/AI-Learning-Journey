# Day07: CNN Fundamentals and CIFAR-10 Classification

## Goal

Day07 introduced convolutional neural networks through a complete CIFAR-10 image-classification experiment. The main topics were:

- convolution, padding, stride, and pooling;
- tensor-shape changes through a CNN;
- local connections and parameter sharing;
- input channels, output channels, and convolutional parameters;
- image normalization and training-only data augmentation;
- CNN training with separate training, validation, and test sets;
- checkpoint selection, inference, confidence, and error analysis.

## Dataset

CIFAR-10 is an image-classification dataset with ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

```text
Training images: 50,000
Test images:     10,000
Image shape:     [3, 32, 32]
```

The original training set was split into 45,000 training images and 5,000 validation images. The official test set was reserved for one final evaluation.

## Data Processing

The training transform applies:

1. random cropping with padding;
2. random horizontal flipping;
3. conversion to a tensor;
4. channel-wise normalization.

Random augmentation is used only for training. Validation and test images use deterministic conversion and the same normalization statistics, ensuring that model comparisons remain repeatable and that all inputs share the same numerical scale.

## CNN Architecture

The model processes a batch of 64 images as follows:

```text
Input:         [64, 3, 32, 32]
Conv1:         [64, 16, 32, 32]
Pool1:         [64, 16, 16, 16]
Conv2:         [64, 32, 16, 16]
Pool2:         [64, 32, 8, 8]
Flatten:       [64, 2048]
Logits:        [64, 10]
Predictions:   [64]
```

The architecture is:

```text
image
  → convolution → ReLU → max pooling
  → convolution → ReLU → max pooling
  → flatten → linear classifier → logits
```

Flattening is performed only after the convolutional layers have extracted spatial features.

## Convolution Fundamentals

### Local Connections

A convolutional kernel processes a small local region instead of connecting every input pixel to every output value. Early layers can learn edges, colors, and textures, while later layers combine them into more complex features.

### Parameter Sharing

The same kernel parameters are reused at every spatial location. A kernel that detects an edge can therefore detect that edge in different parts of an image without learning a separate parameter set for each position.

### Channels and Weight Shape

For the layer:

```python
nn.Conv2d(16, 64, kernel_size=5, padding=2)
```

every output kernel observes all 16 input channels. PyTorch stores its weights in this order:

```text
[out_channels, in_channels, kernel_height, kernel_width]
[64, 16, 5, 5]
```

Its parameter count is:

```text
weights: 64 × 16 × 5 × 5 = 25,600
biases:  64
total:   25,664
```

The batch size never appears in the weight shape and does not change the number of model parameters.

### Pooling

Pooling reduces spatial size, computation, and sensitivity to small positional changes. It also allows later features to summarize larger areas of the original image. Its cost is the loss of spatial detail. Repeatedly applying `2 × 2` pooling to a `32 × 32` feature map gives:

```text
32 × 32 → 16 × 16 → 8 × 8 → 4 × 4 → 2 × 2 → 1 × 1
```

At `1 × 1`, each channel retains feature strength but almost no spatial-location information.

## Model Parameters

The implemented CNN has 25,578 trainable parameters:

```text
conv1: 3 × 16 × 3 × 3 + 16  =    448
conv2: 16 × 32 × 3 × 3 + 32 =  4,640
linear: 2048 × 10 + 10       = 20,490
total:                            25,578
```

Convolution keeps this parameter count far below that of a dense layer connecting every image pixel to every spatial output.

## Experiment Setup

- Model: two-layer CNN
- Activation: ReLU
- Pooling: `2 × 2` max pooling
- Loss function: cross-entropy loss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 5
- Device: CUDA

## Results

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5896 | 42.96% | 1.2987 | 54.24% |
| 2 | 1.3342 | 52.45% | 1.1801 | 58.32% |
| 3 | 1.2294 | 56.55% | 1.0950 | 61.48% |
| 4 | 1.1658 | 58.93% | 1.0130 | 64.66% |
| 5 | **1.1161** | **60.90%** | **0.9696** | **66.02%** |

The best checkpoint came from epoch 5. Validation performance improved throughout all five epochs, so there was no clear evidence of overfitting within this training budget.

```text
Final Test Loss:     0.9648
Final Test Accuracy: 66.53%
```

Validation accuracy was higher than training accuracy because the training images used random augmentation and were therefore harder, while validation images were evaluated without random transformations.

## Inference and Confidence

Inference loads the best checkpoint and uses:

```python
model.eval()
with torch.no_grad():
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
```

In the first ten test images, the model classified nine correctly. One airplane was incorrectly predicted as a ship with 54.65% confidence. A correctly classified frog received only 31.23% confidence, showing that a correct prediction can still be uncertain.

Confidence must not be treated as proof that the model learned the correct features. A high-confidence error may indicate overconfidence or reliance on a shortcut, while one low-confidence sample may simply be ambiguous. Claims about model behavior require patterns across many samples rather than a single example.

![Prediction examples](prediction_examples.png)

## Main Lessons

1. CNNs exploit the local structure of images through local connections and shared parameters.
2. Different output kernels can learn different features, and each kernel observes every input channel.
3. Convolutional weight shape and output-tensor shape describe different objects.
4. Pooling trades spatial detail for lower computation and broader feature summaries.
5. Data augmentation belongs in training, while validation and test evaluation should remain deterministic.
6. Validation data selects the checkpoint; repeated test-set selection would leak test information.
7. Accuracy and confidence measure different aspects of model behavior.

## Files

- `cnn_shape_demo.py`: traces tensor shapes through a two-layer CNN;
- `cifar10_data.py`: loads CIFAR-10 and explores labels, shapes, and transforms;
- `one_batch_cnn_training.py`: performs one CNN parameter update;
- `train_cifar10.py`: runs the full train-validation-test pipeline and saves the best checkpoint;
- `analyze_predictions.py`: loads the checkpoint and reports predictions and confidence;
- `prediction_examples.png`: visualization of ten test images;
- `test_train_cifar10.py`: verifies CNN output and inference probability shapes.
