# JaxMix

A repository for exploring Mixture Density Networks (MDNs) in scientific machine learning applications. This package demonstrates how MDNs provide a principled, data-efficient alternative to diffusion and flow-based models for capturing multimodal conditional uncertainty in scientific problems.

This work supports the paper **"Multimodal Scientific Learning Beyond Diffusions and Flows"**.

## Abstract

Scientific machine learning (SciML) increasingly requires models that capture multimodal conditional uncertainty arising from ill-posed inverse problems, multistability, and chaotic dynamics. While recent work has favored highly expressive implicit generative models such as diffusion and flow-based methods, these approaches are often data-hungry, computationally costly, and misaligned with the structured solution spaces frequently found in scientific problems. We demonstrate that Mixture Density Networks (MDNs) provide a principled yet largely overlooked alternative for multimodal uncertainty quantification in SciML. As explicit parametric density estimators, MDNs impose an inductive bias tailored to low-dimensional, multimodal physics, enabling direct global allocation of probability mass across distinct solution branches. This structure delivers strong data efficiency, allowing reliable recovery of separated modes in regimes where scientific data is scarce. We formalize these insights through a unified probabilistic framework contrasting explicit and implicit distribution networks, and demonstrate empirically that MDNs achieve superior generalization, interpretability, and sample efficiency across a range of inverse, multistable, and chaotic scientific regression tasks.

## Examples

This repository contains four examples demonstrating MDN capabilities across different types of multimodal scientific problems:

### 1. Sine Inverse Problem (`examples/sine_inverse_problem/`)

A classic multimodal inverse problem where multiple inputs map to the same output. This example demonstrates:
- **Multimodal inverse problem solving**: Learning the one-to-many mapping from observations back to inputs
- **Data efficiency comparisons**: Shows how MDNs perform in low-data regimes typical of scientific applications
- **Mixture component ablations**: Explores the impact of the number of mixture components on model performance

This toy problem provides an accessible entry point for understanding MDN behavior and serves as a baseline for comparing against other generative approaches.

### 2. Gravitational Attractor System (`examples/attractor_system/`)

A dynamical system with multiple stable attractors, where initial conditions determine which basin of attraction the system evolves into. This example demonstrates:
- **Multiple attractor basins**: Capturing distinct solution branches corresponding to different long-term behaviors
- **Multistability**: Learning the mapping from initial conditions to multiple possible steady states
- **Mode separation**: MDN's ability to explicitly allocate probability mass to separated regions of solution space

### 3. Bifurcation System (`examples/bifurcation_system/`)

Systems exhibiting saddle-node bifurcations, where qualitative behavior changes dramatically at critical parameter values. This example demonstrates:
- **Saddle-node bifurcation examples**: Capturing abrupt transitions between stability regimes
- **Critical transitions**: Modeling how solutions jump between branches at bifurcation points

### 4. Lorenz System (`examples/lorenz_system/`)

The Lorenz attractor is a canonical example of chaotic dynamics arising from a simple 3D ODE system. This example demonstrates:
- **Chaotic attractor dynamics**: Capturing the sensitive dependence on initial conditions
- **Uncertainty quantification in chaos**: MDN's ability to represent the multimodal distributions that emerge from chaotic evolution

## Motivation

We encourage computational scientists to explore MDNs for their own multimodal scientific problems. The examples in this repository demonstrate that MDNs offer several key advantages for scientific applications:

- **Data efficiency**: Strong performance even with limited training data, common in experimental and simulation-based science
- **Interpretability**: Direct access to mixture components, means, variances, and weights enables physical interpretation
- **Computational efficiency**: Faster training and inference compared to diffusion/flow models
- **Structured inductive bias**: Natural alignment with low-dimensional multimodal structure in scientific problems

Whether you're working with inverse problems, multistable systems, bifurcations, or chaotic dynamics, MDNs provide a principled framework for uncertainty quantification that balances expressiveness with the constraints of scientific data.

## Implementation

This package is implemented in **JAX** with **Flax** for neural network architectures, enabling:
- Fast automatic differentiation for gradient-based optimization
- Easy integration with scientific computing workflows
- GPU acceleration for efficient training

The core MDN implementation and training utilities are provided in the `jaxmix/` module, with self-contained example notebooks demonstrating application to each system.

## Usage

The `jaxmix` package provides a simple API for building, training, and sampling from Mixture Density Networks. Here's a basic example:

### 1. Building an MDN Model

```python
import jax.random as random
from jaxmix.archs import MLP, MDN
from jaxmix.trainers import MDNTrainer
from jaxmix.data_loaders import BatchedDataset

# Set random seed
key = random.PRNGKey(42)

# Define model architecture
input_dim = 1
output_dim = 1
num_mixtures = 5

# Create a backbone network (MLP)
backbone = MLP(features=[64, 64, 64])

# Create the MDN architecture
mdn_arch = MDN(
    num_mixtures=num_mixtures,
    num_output_dims=output_dim,
    backbone=backbone
)
```

### 2. Preparing Data

```python
from jaxmix.utils import split_data

# Prepare your data (inputs, outputs, weights)
# inputs: (n_samples, input_dim)
# outputs: (n_samples, output_dim)
# weights: (n_samples, 1) - typically all ones

# Split into train/test sets
key, subkey = random.split(key)
train_data, test_data = split_data(
    subkey, inputs, outputs, 
    n_train=5000, n_val=1000
)

# Create data loaders
key, subkey = random.split(key)
train_loader = BatchedDataset(
    train_data, 
    key=subkey, 
    batch_size=256
)
```

### 3. Training the Model

```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Initialize trainer
model = MDNTrainer(
    arch=mdn_arch,
    init_batch=train_data,
    optimizer=optimizer,
    key=key
)

# Train the model
model.train(
    train_loader=train_loader,
    nIter=10_000,
)
```

For complete examples demonstrating these techniques on real scientific problems, see the notebooks in the `examples/` directory.
