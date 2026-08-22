# Day08: Residual Networks and Controlled CNN Comparison

## Goal

Day08 extended the basic CIFAR-10 CNN into a small residual network. The main goals were to understand why deeper networks can be difficult to optimize, implement residual blocks from first principles, and compare a plain CNN with a structurally similar ResNet under controlled training conditions.

The main topics were:

- network degradation and residual learning;
- identity and projection shortcuts;
- shape changes across residual stages;
- Batch Normalization in training and evaluation modes;
- global average pooling;
- the structure and naming of ResNet-18;
- controlled architecture comparison;
- the difference between accuracy, cross-entropy loss, and confidence.

## Residual Learning

A plain convolutional block learns a complete mapping:

```text
y = F(x)
```

A residual block adds the original input through a shortcut:

```text
y = F(x) + x
```

If the existing representation is already useful, a plain block must learn the full identity mapping `F(x) = x` to preserve it. A residual block only needs its residual branch to approach zero:

```text
F(x) ≈ 0  →  y ≈ x
```

The shortcut also gives information and gradients a more direct path through a deep network. Removing `out = out + identity` leaves a plain convolutional network even if all tensor shapes remain unchanged.

## Residual Block

The same-shape residual block uses:

```text
x ───────────────────────────────┐
│                                │
└→ Conv3×3 → BN → ReLU           │
            → Conv3×3 → BN ──────+→ ReLU → output
```

The second convolution is followed by Batch Normalization, but the final ReLU is applied only after adding the shortcut. Applying ReLU to the shortcut before addition would remove its negative values and prevent it from transmitting information as directly as possible.

### Projection Shortcut

Tensor addition requires identical shapes. When the main branch changes:

```text
[64,16,32,32] → [64,32,16,16]
```

the shortcut uses a learnable projection:

```python
nn.Sequential(
    nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
    nn.BatchNorm2d(32),
)
```

The `1 × 1` convolution changes `16 → 32` channels, and `stride=2` changes `32 × 32 → 16 × 16`. The two branches can then be added.

## Batch Normalization

`BatchNorm2d(C)` normalizes each of the `C` channels separately. During training, it uses statistics from the current batch and updates running estimates. During evaluation, it uses the accumulated running mean and variance so that predictions remain stable and do not depend on the other images in the current batch.

This is why validation and inference require:

```python
model.eval()
```

Each channel has two learned BatchNorm parameters:

```text
gamma: learned scale
beta:  learned shift
```

Therefore, `BatchNorm2d(64)` has `64 + 64 = 128` trainable parameters. Its `running_mean` and `running_var` are saved state, but they are not updated through gradients.

A convolution directly followed by BatchNorm commonly uses `bias=False`, because BatchNorm's learned beta already provides a shift.

## SmallResNet Architecture

The educational SmallResNet processes CIFAR-10 images as follows:

```text
Input:               [64, 3, 32, 32]
Stem:                [64,16,32,32]
Residual stage 1:    [64,16,32,32]
Residual stage 2:    [64,32,16,16]
Residual stage 3:    [64,64, 8, 8]
Global average pool: [64,64, 1, 1]
Flatten:             [64,64]
Classifier:          [64,10]
```

As spatial resolution decreases, the network increases the number of channels. Lower resolution reduces the cost of spatial processing, while additional channels allow later layers to represent more kinds of complex features. The parameter count still increases with channel width; it is the spatial computation that is kept more manageable.

### Global Average Pooling

`AdaptiveAvgPool2d((1,1))` averages every spatial feature map into one value:

```text
[64,64,8,8] → [64,64,1,1] → flatten → [64,64]
```

The classifier can therefore use `Linear(64,10)` instead of connecting all `64 × 8 × 8` spatial values to the output. This greatly reduces classifier parameters but removes precise final-stage location information.

## Connection to ResNet-18

The SmallResNet is a teaching model, not an implementation of the official ResNet-18. ResNet-18 contains:

```text
1 stem convolution
4 stages × 2 residual blocks × 2 convolutions
1 final fully connected layer
= 18 learned main-path layers
```

The first block of a new stage increases channels and usually downsamples with `stride=2`. Later blocks in that stage keep the same channels and use `stride=1`.

For small `32 × 32` CIFAR-10 images, a `3 × 3`, stride-1 stem preserves more early spatial detail than the larger, aggressively downsampling stem commonly used for `224 × 224` ImageNet images.

## Controlled Experiment

The experiment compared:

- `PlainCNN`: the same three two-convolution main blocks without shortcuts;
- `SmallResNet`: the same main structure with identity or projection shortcuts.

The following conditions were held constant:

- 45,000 training and 5,000 validation images;
- the official 10,000-image test set;
- data augmentation and normalization;
- data split, shuffle, and augmentation seeds;
- Adam optimizer with learning rate `0.001`;
- batch size `64`;
- five training epochs.

Model selection used validation accuracy. The test set was evaluated only after restoring each model's best validation state.

### Parameter Counts

```text
PlainCNN:    75,290
SmallResNet: 78,042
Difference:   2,752
```

The difference comes exactly from the two projection shortcuts:

```text
16→32 shortcut: 16×32 + 2×32 =   576
32→64 shortcut: 32×64 + 2×64 = 2,176
Total:                             2,752
```

## Results

### PlainCNN

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5377 | 43.44% | 1.3694 | 50.10% |
| 2 | 1.1504 | 58.88% | 1.1236 | 58.68% |
| 3 | 1.0179 | 63.86% | 1.0087 | 64.20% |
| 4 | 0.9348 | 66.67% | 0.9037 | 67.68% |
| 5 | 0.8770 | 68.97% | 0.8693 | **68.72%** |

```text
Test Loss:     0.8846
Test Accuracy: 68.66%
```

### SmallResNet

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5398 | 43.50% | 1.3348 | 51.28% |
| 2 | 1.1585 | 58.56% | 1.0879 | 61.00% |
| 3 | 1.0095 | 64.27% | 0.9491 | 66.58% |
| 4 | 0.9232 | 67.17% | 0.8962 | 67.12% |
| 5 | 0.8607 | 69.60% | 0.8842 | **69.12%** |

```text
Test Loss:     0.8949
Test Accuracy: 68.71%
```

## Interpretation

SmallResNet had higher validation accuracy during the first three epochs, which provides limited evidence of faster early convergence. Its best validation accuracy was 0.40 percentage points higher. However, its test accuracy was only 0.05 percentage points higher, corresponding to approximately five additional correct predictions among 10,000 test images.

This single five-epoch run does not establish that SmallResNet generalizes better. The network is shallow, the difference is small, and no repeated-seed experiment was performed. A stronger claim would require deeper networks, multiple seeds, longer training, and reporting the mean and variability.

PlainCNN achieved slightly lower test loss despite slightly lower accuracy. Accuracy only checks the highest-scoring class, while cross-entropy also measures the probability assigned to the true class. SmallResNet may have made a few more correct predictions while being less confident on some correct samples or more confident on some errors.

## Errors and Lessons

1. The first residual-block attempt called two convolutions but omitted BatchNorm, ReLU, and shortcut addition. Its output shape was still correct, demonstrating that shape checks alone cannot verify computation logic.
2. `forward()` was temporarily indented outside the class, causing PyTorch to report that the module had no forward function.
3. A second block inside the same stage must use `stride=1`; using `stride=2` would perform an unintended second downsampling.
4. Debug prints are useful in a shape demo but should not remain in the reusable model used for every training batch.
5. Controlled experiments answer a specific question only when non-target conditions are kept fixed.

## Files

- `residual_block_demo.py`: same-shape and projection residual-block experiments;
- `small_resnet_shape_demo.py`: step-by-step SmallResNet tensor-shape trace;
- `resnet_model.py`: clean reusable implementations of PlainCNN and SmallResNet;
- `compare_plain_and_resnet.py`: controlled CIFAR-10 training and evaluation experiment;
- `test_resnet_model.py`: model-shape, silent-forward, training-update, and evaluation-safety tests.
