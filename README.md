# Convolutional Neural Network from Scratch

A fully vectorized implementation of a Convolutional Neural Network (CNN) built entirely from scratch using NumPy, with no deep learning frameworks like PyTorch or TensorFlow.

<img width="640" height="480" alt="convnet_final" src="https://github.com/user-attachments/assets/8a849796-a00a-4b5a-b58c-5be43dbc7730" />

## Overview

This project demonstrates the implementation of a complete CNN architecture using only NumPy, linear algebra, and calculus. The implementation is fully vectorized for efficient training and inference, achieving **99% accuracy** on the MNIST dataset.

## Features

- **Pure NumPy Implementation**: No reliance on PyTorch, TensorFlow, or other deep learning frameworks
- **Fully Vectorized Operations**: Efficient implementation using `np.einsum` and stride tricks for optimal performance
- **Complete CNN Architecture**: Includes convolutional layers, max pooling, and fully connected layers
- **Adam Optimizer**: Implements the Adam optimization algorithm with learning rate scheduling
- **Memory Efficient**: Custom memory pooling system to minimize allocations during training
- **Flexible Architecture**: Easily configurable network structure through parameter dictionaries

## Network Architecture

The default configuration implements the following architecture:

```
Input (28x28x1)
    ↓
Conv Layer 1 (5x5 kernel, 32 filters, stride=1, padding=2) + ReLU
    ↓
Max Pooling (2x2, stride=2)
    ↓
Conv Layer 2 (5x5 kernel, 64 filters, stride=1, padding=2) + ReLU
    ↓
Max Pooling (2x2, stride=2)
    ↓
Flatten
    ↓
Dense Layer (1024 units) + ReLU
    ↓
Output Layer (10 units) + Softmax
```

## Key Components

### Forward Propagation
- **Vectorized Convolution**: Uses `np.lib.stride_tricks` and `np.einsum` for efficient sliding window operations
- **Max Pooling**: Vectorized implementation with proper handling of pooling windows
- **Dense Layers**: Standard fully connected layers with ReLU activation
- **Softmax Output**: Categorical cross-entropy loss for multi-class classification

### Backpropagation
- **Convolutional Backprop**: Efficient gradient computation through conv layers using einsum
- **Max Pool Backprop**: Gradient routing through max pooling using argmax masks
- **Dense Backprop**: Standard backpropagation through fully connected layers

### Optimization
- **Adam Optimizer**: Adaptive learning rate with momentum and RMSprop
- **Learning Rate Scheduling**: Decay schedule for improved convergence
- **Batch Training**: Mini-batch gradient descent with shuffling

### Memory Management
- **Buffer Pooling**: Reuses pre-allocated buffers to reduce memory overhead
- **Batched Prediction**: Processes large datasets in batches with periodic cleanup

## Results

Training on the full MNIST dataset (60,000 training images, 10,000 test images):

- **Training Accuracy**: 100.0%
- **Test Accuracy**: 99%
- **Training Time**: ~25 epochs with batch size of 128

## Installation

```bash
# Clone the repository
git clone https://github.com/MaximusThomas/ML-Projects.git
cd ML-Projects

# Install dependencies
pip install numpy matplotlib tqdm scikit-learn tensorflow
```

Note: TensorFlow is only used for loading the MNIST dataset, not for model implementation.

## Usage

```python
# Load and preprocess data
from tensorflow.keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Reshape and normalize
train_X = train_X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
test_X = test_X.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

# One-hot encode labels
train_y = one_hot_encode(train_y)
test_y = one_hot_encode(test_y)

# Define architecture
conv_dims = {
    'conv': {'n_f': 5, 'n_s': 1, 'n_p': 2, 'n_filters': 32},
    'max_pool': {'pool_size': 2, 'n_s': 2},
    'conv2': {'n_f': 5, 'n_s': 1, 'n_p': 2, 'n_filters': 64},
    'max_pool2': {'pool_size': 2, 'n_s': 2}
}
dense_dims = [1024, 10]

# Train the model
params, cost_history = train(
    train_X, train_y, test_X, test_y,
    dims=(conv_dims, dense_dims),
    epochs=25,
    batch_size=128
)

# Make predictions
predictions = batched_predict_with_cleanup(test_X, test_y, params, dims)
```

## Implementation Details

### Vectorization Techniques

The implementation uses advanced NumPy techniques for efficiency:

1. **Stride Tricks**: Creates memory-efficient sliding windows for convolution and pooling
2. **Einstein Summation (einsum)**: Computes convolutions and gradients with optimal performance
3. **Broadcasting**: Leverages NumPy broadcasting for batch operations

### Parameter Initialization

- **He Initialization**: Weights initialized using He initialization for ReLU activation
- **Zero Bias**: Biases initialized to zero

### Training Optimizations

- **Learning Rate Decay**: Reduces learning rate after epochs 10 and 20
- **Batch Normalization**: Data normalized to [0, 1] range
- **Data Shuffling**: Random permutation of training data each epoch

## Architecture Flexibility

The network architecture is highly configurable. You can easily modify:

- Number of convolutional layers
- Filter sizes and counts
- Pooling strategies
- Dense layer dimensions
- Activation functions

Simply adjust the `conv_dims` and `dense_dims` dictionaries to experiment with different architectures.

## Performance Considerations

- **Memory Pooling**: Reduces allocation overhead by reusing buffers
- **Vectorized Operations**: All operations are fully vectorized for NumPy efficiency
- **Batched Processing**: Handles large datasets through mini-batch processing
- **Gradient Computation**: Efficient einsum-based gradient calculations

## Educational Value

This project is ideal for:

- Understanding the mathematics behind CNNs
- Learning how convolution and backpropagation work at a low level
- Exploring optimization techniques
- Studying vectorization and efficient NumPy programming

## Future Improvements

Potential enhancements include:

- [ ] Batch normalization layers
- [ ] Dropout for regularization
- [ ] Additional activation functions (Leaky ReLU, ELU)
- [ ] Different optimizers (SGD with momentum, RMSprop)
- [ ] Data augmentation
- [ ] Model checkpointing and loading
- [ ] Support for different dataset formats

## Dependencies

- NumPy
- Matplotlib (for visualization)
- tqdm (for progress bars)
- scikit-learn (for metrics)
- TensorFlow (only for MNIST data loading)

## License

This project is open source and available for educational purposes.

## Author

Maximus Thomas

## Acknowledgments

Built as a learning project to understand the fundamentals of convolutional neural networks and deep learning from first principles.
