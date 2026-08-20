# Day05: Fashion-MNIST Image Classification

## Goal

The goal of Day05 was to understand and implement a complete training pipeline for an MLP classifier. The main tasks included:

- exploring the Fashion-MNIST dataset;
- understanding the shapes of images, labels, logits, and predictions;
- learning how Softmax and cross-entropy loss work;
- training a model on a GPU;
- distinguishing between training and evaluation modes;
- saving and restoring the best-performing model parameters.

## Dataset

Fashion-MNIST is an image classification dataset, not a model. It contains ten categories of clothing images:

- 60,000 training images;
- 10,000 test images;
- one `28 × 28` grayscale channel per image;
- one class label from `0` to `9` per image.

This project uses an MLP to classify the Fashion-MNIST images.

## Tensor Shapes

With a batch size of 64, the tensors move through the model as follows:

```text
Original images:  [64, 1, 28, 28]
Flattened images: [64, 784]
Logits:           [64, 10]
Labels:           [64]
Predictions:      [64]
```

The logits tensor has shape `[64, 10]` because the batch contains 64 images and the model produces ten class scores for each image.

## Training One Batch

Each training iteration follows these steps:

1. Load a batch of images and labels from the `DataLoader`.
2. Move the images and labels to the GPU.
3. Clear gradients from the previous iteration with `optimizer.zero_grad()`.
4. Perform a forward pass to obtain the logits.
5. Calculate the cross-entropy loss from the logits and labels.
6. Run `loss.backward()` to calculate gradients.
7. Run `optimizer.step()` to update the model parameters.
8. Record the loss and accuracy for monitoring.

## Key Concepts

### Forward Pass

```python
logits = model(images)
```

During the forward pass, the images move through the layers of the MLP. The model uses its current parameters to calculate one class score for every possible category. A forward pass calculates predictions but does not update the parameters.

### Logits

Logits are the raw class scores produced by the model. They are not probabilities. In this ten-class task, the model produces ten logits for each image, and the index of the largest logit is used as the predicted class.

### Softmax

Softmax converts logits into class probabilities whose sum is 1. It helps show how confident the model is about each possible class.

Softmax should not be applied manually before PyTorch's `CrossEntropyLoss`, because the loss function already performs the required internal calculation.

### Cross-Entropy Loss

Cross-entropy measures how much probability the model assigns to the correct class. A higher probability for the correct class produces a lower loss, while a lower probability produces a higher loss.

### Backward Pass

```python
loss.backward()
```

The backward pass uses the loss to calculate a gradient for every trainable parameter. These gradients are stored in each parameter's `.grad` attribute. The backward pass calculates gradients; `optimizer.step()` performs the actual parameter update.

### Clearing Gradients

PyTorch accumulates gradients by default. Without clearing the previous gradients before the next backward pass, gradients from different batches would be added together and the ordinary training procedure would no longer behave as intended.

### Training and Evaluation Modes

- `model.train()` switches the model to training mode.
- `model.eval()` makes layers such as Dropout and Batch Normalization use their evaluation behavior.
- `torch.no_grad()` disables computation-graph and gradient recording, reducing memory use and computation during evaluation.

`model.eval()` does not disable gradient recording, so evaluation commonly uses both `model.eval()` and `torch.no_grad()`.

### Checkpoints

`model.state_dict()` contains the model parameters. During training, `best_mlp.pth` is updated only when the evaluation accuracy improves. Loading this checkpoint after training restores the best-performing model instead of automatically using the final epoch.

## Experiment Setup

- Model: MLP with dimensions `784 → 128 → 10`
- Activation: ReLU
- Loss function: cross-entropy loss
- Optimizer: SGD
- Learning rate: 0.1
- Epochs: 5
- Device: CUDA

## Result

The best result occurred at epoch 4:

```text
Test Loss:     0.4098
Test Accuracy: 85.04%
```

At epoch 5, the training accuracy continued to improve while the test accuracy decreased. This suggests the beginning of overfitting. Restoring the saved checkpoint reproduced the result from epoch 4.

> This introductory experiment monitored the test set after every epoch. A more rigorous project should use separate training, validation, and test sets: the training set updates parameters, the validation set selects models and hyperparameters, and the test set is used only for final evaluation.

## Errors and Lessons

1. The accuracy was initially divided by the number of classes instead of the number of samples.
2. Softmax was initially applied along `dim=0`; class probabilities needed to be calculated along `dim=1`.
3. A missing comma inside `nn.Sequential` caused a syntax error.
4. Writing `loss.backward` without parentheses referenced the method but did not execute backpropagation.
5. Writing `torch.no_grad` without parentheses failed to create the context manager required by `with`.
6. The best accuracy was initially defined inside a function, so the outer training loop could not access or preserve it across epochs.
7. The checkpoint message was initially outside the `if` block, causing it to report a save even when the checkpoint was not updated.

## Files

- `classification_basics.py`: class indices, `argmax`, and accuracy;
- `cross_entropy_demo.py`: Softmax and cross-entropy loss;
- `fashion_mnist_data.py`: Fashion-MNIST and `DataLoader` usage;
- `mlp_shape_demo.py`: tensor-shape changes through the MLP;
- `one_batch_training.py`: one parameter update on a single batch;
- `train_fashion_mnist.py`: full training, evaluation, checkpoint saving, and checkpoint loading.
