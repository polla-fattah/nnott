# AlexNet (2012)

## Historical Context

AlexNet, created by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, won the 2012 ImageNet competition and sparked the modern deep-learning wave in computer vision. It proved that deep CNNs paired with GPUs could outperform traditional methods by a wide margin.

## Architecture Highlights

1. **Five convolutional layers** with large early kernels (e.g., 11×11) and overlapping max pooling.
2. **ReLU activations** throughout—faster to train than tanh/sigmoid.
3. **Dropout** in the fully connected layers to combat overfitting.
4. **Data augmentation** (random crops, flips) in the original paper; in this sandbox, shift augmentation covers similar ground.

## Implementation Notes

- **File:** `convolutional/architectures/alexnet.py`
- Mimics the original filter sizes/channel counts but adapted to 28×28 MNIST inputs (so spatial dimensions are smaller).
- Uses the shared Conv2D/BatchNorm/ReLU modules plus `Dense` layers with dropout in the head.

## Teaching Angles

- Introduces deeper conv stacks and the idea of aggressively increasing feature channels.
- Shows how dropout became essential when networks grew larger.
- Good case study for GPU vs CPU timing—AlexNet has enough parameters to reveal meaningful differences.

## Suggested Experiments

- Train for a few epochs on GPU and compare accuracy/time to LeNet.
- Toggle dropout or change batch size to study optimization stability.

## References

- [AlexNet on Wikipedia](https://en.wikipedia.org/wiki/AlexNet)
