# Day06: Optimization and Generalization

## Goal

Day06 focused on how optimization settings affect training and how validation data is used to measure generalization. The main topics were:

- the role of the learning rate;
- convergence, oscillation, and divergence;
- the differences between SGD, SGD with Momentum, and Adam;
- training, validation, and test splits;
- underfitting and overfitting;
- Dropout and weight decay as regularization methods.

## Learning Rate

For basic SGD, a parameter update can be summarized as:

```text
new parameter = old parameter - learning rate × gradient
```

A learning rate that is too small produces slow parameter updates and may require many additional training epochs. A learning rate that is too large can repeatedly overshoot a minimum, causing the loss to oscillate or diverge.

The one-parameter experiment used:

```text
loss = (w - 1)²
```

Starting from `w = 3` produced four different behaviors:

- `lr=0.1`: stable but slow convergence;
- `lr=0.5`: reached the minimum in one update for this specific function;
- `lr=1.0`: oscillated between two points without reducing the loss;
- `lr=1.1`: oscillated with increasing amplitude and diverged.

## Train, Validation, and Test Sets

The original 60,000 Fashion-MNIST training images were split into:

```text
50,000 training images
10,000 validation images
```

The official 10,000-image test set remained separate.

- The training set is used to calculate gradients and update model parameters.
- The validation set is used to compare hyperparameters and select a model.
- The test set is used only for the final evaluation after all choices have been made.

Repeatedly selecting configurations from test-set results would leak information from the test set into model development, making the final evaluation less independent.

## Learning-Rate Experiment

Four learning rates were compared with the same MLP, SGD optimizer, data split, random seed, and three-epoch training budget.

| Learning Rate | Final Validation Loss | Final Validation Accuracy | Interpretation |
| ---: | ---: | ---: | --- |
| 0.001 | 1.2941 | 64.93% | Stable but too slow for the training budget |
| 0.01 | 0.5963 | 79.60% | Stable, but not yet fully converged |
| 0.1 | **0.4620** | **83.15%** | Best result among the tested SGD rates |
| 1.0 | 2.3169 | 9.79% | Training failed because the rate was too large |

The result only establishes that `0.1` was the best of these four SGD learning rates under this specific three-epoch setup. It does not show that `0.1` is universally optimal.

## Optimizer Comparison

### SGD

Basic SGD mainly follows the gradient calculated from the current batch. Noisy batch gradients can cause the update direction to change repeatedly.

### SGD with Momentum

Momentum preserves part of the previous update direction. This can reduce side-to-side oscillation and maintain progress when individual batches produce noisy gradients.

### Adam

Adam tracks both gradient direction and gradient scale, adapting the effective update size for different parameters. Its usual learning-rate scale differs from SGD, so the numerical learning rates should not be compared directly.

The optimizer configurations produced:

| Optimizer | Learning Rate | Final Train Accuracy | Final Validation Loss | Final Validation Accuracy |
| --- | ---: | ---: | ---: | ---: |
| SGD | 0.1 | 85.32% | 0.4620 | 83.15% |
| SGD with Momentum | 0.1 | 83.49% | 0.4512 | 84.84% |
| Adam | 0.001 | **86.53%** | **0.3639** | **86.88%** |

Adam performed best in this three-epoch Fashion-MNIST experiment. This does not imply that Adam is always better than SGD for every model, dataset, or training budget.

## Underfitting and Overfitting

### Underfitting

When both training and validation performance are poor, the model has not learned the training data sufficiently. Possible causes include insufficient model capacity, too few epochs, an unsuitable learning rate, or an optimization problem.

### Overfitting

Overfitting is indicated when training performance continues to improve while validation performance stops improving or becomes worse. A large gap between strong training performance and weak validation performance is another warning sign.

Good generalization requires both strong absolute validation performance and a reasonable gap between training and validation results.

## Regularization

### Dropout

Dropout randomly removes part of a layer's output during training, reducing the model's ability to depend on a small set of activations. With `p=0.5`, PyTorch scales the retained values by `1 / (1 - p) = 2` during training to preserve their expected magnitude.

```text
Training mode:   random values are removed and retained values are scaled
Evaluation mode: no random removal is applied
```

This is why a model containing Dropout must use `model.eval()` during validation and testing.

### Weight Decay

Weight decay discourages excessively large parameter values. It can reduce overfitting, but like Dropout, it is not automatically helpful when a model is already underfitting.

## Main Lessons

1. Larger learning rates do not always produce faster learning.
2. Hyperparameter comparisons must keep other experimental conditions fixed.
3. Optimizers use different learning-rate scales.
4. Model selection should use validation performance, not training accuracy or repeated test-set checks.
5. Regularization should address observed overfitting rather than being added automatically.

## Files

- `learning_rate_demo.py`: one-parameter convergence and divergence experiment;
- `validation_split_demo.py`: reproducible training-validation split;
- `learning_rate_experiment.py`: controlled learning-rate comparison;
- `momentum_experiment.py`: SGD with Momentum experiment;
- `adam_experiment.py`: Adam experiment;
- `dropout_mode_demo.py`: Dropout behavior in training and evaluation modes;
- `test_learning_rate_experiment.py`: checks that training updates parameters and evaluation does not.
