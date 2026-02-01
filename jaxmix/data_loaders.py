# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad

import torch.utils.data as torch_data
import pandas as pd
from functools import partial
import scipy.io
from typing import Tuple, Optional, Any

# Dataset loader
class BatchedDataset(torch_data.Dataset):
  """A dataset loader that returns random batches of data.

  This dataset takes raw data (inputs, targets, weights) and returns random batches
  of specified size. If batch_size is None or larger than dataset size, returns full dataset.

  Args:
      raw_data (tuple): Tuple of (inputs, targets, weights) arrays
      key (jax.random.PRNGKey): Random key for batch sampling
      batch_size (int, optional): Size of batches to return. Defaults to None (full dataset).
  """
  def __init__(
      self,
      raw_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      key: jax.Array,
      batch_size: Optional[int] = None
    ) -> None:
    super().__init__()
    self.inputs = raw_data[0]
    self.targets = raw_data[1]
    self.weights = raw_data[2]
    assert len(self.inputs) == len(self.targets), f'inputs and targets must have the same length, but got {len(self.inputs)} and {len(self.targets)}'
    assert len(self.inputs) == len(self.weights), f'inputs and weights must have the same length, but got {len(self.inputs)} and {len(self.weights)}'
    self.size = len(self.weights)
    self.key = key
    if batch_size is None: # Will use full batch
      self.batch_size = self.size
    else:
      if batch_size > self.size:
        print(f'WARNING: batch_size is greater than the dataset size, will use full batch instead.')
        self.batch_size = self.size
      else:
        self.batch_size = batch_size
    
  def __len__(self) -> int:
    return self.size
  
  def __getitem__(self, idx: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get a random batch of data.

    Args:
        idx: Unused, required by Dataset interface

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays
    """
    self.key, subkey = random.split(self.key)
    batch_inputs, batch_targets, batched_weights = self.__select_batch(subkey)
    return batch_inputs, batch_targets, batched_weights

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select a random batch using the given key.

    Args:
        key (jax.random.PRNGKey): Random key for batch selection

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays
    """
    idx = random.choice(key, self.size, (self.batch_size,), replace=False)
    batch_inputs = self.inputs[idx]
    batch_targets = self.targets[idx]
    batched_weights = self.weights[idx]
    return batch_inputs, batch_targets, batched_weights


# Dataset loader for Flow Matching
class FlowMatchingDataset(torch_data.Dataset):
  """A dataset loader for conditional flow matching training.

  This dataset takes conditional variables, samples, and weights, and returns batches
  suitable for training conditional flow matching models. Each batch consists of:
  - net_inputs: concatenation of [conditional_variables, y_t, t] where y_t is interpolated
  - target_velocity_field: the conditional velocity field u_t(y_t | x, y_1)
  - weights: sample weights

  Args:
      raw_data (tuple): Tuple of (conditional_variables, samples, weights) arrays
      key (jax.random.PRNGKey): Random key for batch sampling and time sampling
      batch_size (int, optional): Size of batches to return. Defaults to None (full dataset).
      sigma (float, optional): Standard deviation for conditional flow matching. Defaults to 0.0 (OT-CFM).
  """
  def __init__(
      self,
      raw_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      key: jax.Array,
      batch_size: Optional[int] = None,
      sigma: float = 0.0,
      noise_mean: Optional[jnp.ndarray] = None,
      noise_std: Optional[jnp.ndarray] = None,
    ) -> None:
    super().__init__()
    self.conditional_variables = raw_data[0]
    self.samples = raw_data[1]
    self.weights = raw_data[2]
    assert len(self.conditional_variables) == len(self.samples), f'conditional_variables and samples must have the same length, but got {len(self.conditional_variables)} and {len(self.samples)}'
    assert len(self.conditional_variables) == len(self.weights), f'conditional_variables and weights must have the same length, but got {len(self.conditional_variables)} and {len(self.weights)}'
    if noise_mean is None:
      self.noise_mean = jnp.zeros((self.samples.shape[-1],))
    else:
      self.noise_mean = noise_mean
    if noise_std is None:
      self.noise_std = jnp.ones((self.samples.shape[-1],))
    else:
      self.noise_std = noise_std
    self.size = len(self.weights)
    self.key = key
    self.sigma = sigma
    if batch_size is None: # Will use full batch
      self.batch_size = self.size
    else:
      if batch_size > self.size:
        print(f'WARNING: batch_size is greater than the dataset size, will use full batch instead.')
        self.batch_size = self.size
      else:
        self.batch_size = batch_size
    
  def __len__(self) -> int:
    return self.size
  
  def __getitem__(self, idx: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get a random batch of flow matching training data.

    Args:
        idx: Unused, required by Dataset interface

    Returns:
        tuple: (net_inputs, target_velocity_field, batched_weights) arrays
    """
    self.key, subkey = random.split(self.key)
    net_inputs, target_velocity_field, batched_weights = self.__select_batch(subkey)
    return net_inputs, target_velocity_field, batched_weights

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select a random batch and compute flow matching training data.

    Args:
        key (jax.random.PRNGKey): Random key for batch and time sampling

    Returns:
        tuple: (net_inputs, target_velocity_field, batched_weights) arrays
    """
    # Split keys for different random operations
    key, idx_key, t_key, noise_key = random.split(key, 4)
    
    # Sample batch indices
    idx = random.choice(idx_key, self.size, (self.batch_size,), replace=False)
    batch_conditional = self.conditional_variables[idx]
    batch_samples = self.samples[idx]
    batched_weights = self.weights[idx]
    
    # Sample time uniformly from [0, 1]
    t = random.uniform(t_key, shape=(self.batch_size, 1), minval=0.0, maxval=1.0)
    
    # Sample initial noise y_0 ~ N(0, I)
    y_0 = random.normal(noise_key, shape=batch_samples.shape) * self.noise_std + self.noise_mean
    
    # Compute interpolated state y_t = (1 - (1 - sigma) * t) * y_0 + t * y_1
    # For OT-CFM (sigma=0): y_t = (1 - t) * y_0 + t * y_1
    y_t = (1 - (1 - self.sigma) * t) * y_0 + t * batch_samples
    
    # Compute conditional velocity field u_t = y_1 - (1 - sigma) * y_0
    # For OT-CFM (sigma=0): u_t = y_1 - y_0
    target_velocity_field = batch_samples - (1 - self.sigma) * y_0
    
    # Concatenate [conditional_variables, y_t, t] as network input
    net_inputs = jnp.concatenate([batch_conditional, y_t, t], axis=-1)
    
    return net_inputs, target_velocity_field, batched_weights


# Dataset loader
class BootstrapBatchedDataset(torch_data.Dataset):
  """A dataset loader that returns random batches from bootstrapped data. Meant for ensemble training via bootstrap sampling/bagging.

  This dataset creates multiple bootstrap samples of the input data and returns random batches
  from these bootstrapped samples. Useful for ensemble training.

  Args:
      raw_data (tuple): Tuple of (inputs, targets, weights) arrays
      key (jax.random.PRNGKey): Random key for batch sampling
      num_bootstraps (int): Number of bootstrap samples to create
      bootstrap_proportion (float, optional): Proportion of data to use in each bootstrap. Defaults to 1.
      use_replacement (bool, optional): Whether to sample with replacement. Defaults to True.
      batch_size (int, optional): Size of batches to return. Defaults to None (full dataset).
  """
  def __init__(
      self,
      raw_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      key: jax.Array,
      num_bootstraps: int,
      bootstrap_proportion: float = 1,
      use_replacement: bool = True,
      batch_size: Optional[int] = None
    ) -> None:
    super().__init__()
    self.inputs = raw_data[0]
    self.targets = raw_data[1]
    self.weights = raw_data[2]
    assert len(self.inputs) == len(self.targets), f'inputs and targets must have the same length, but got {len(self.inputs)} and {len(self.targets)}'
    assert len(self.inputs) == len(self.weights), f'inputs and weights must have the same length, but got {len(self.inputs)} and {len(self.weights)}'
    self.size = len(self.weights)
    self.key = key
    if batch_size is None: # Will use full batch
      self.batch_size = self.size
    else:
      if batch_size > self.size:
        print(f'WARNING: batch_size is greater than the dataset size, will use full batch instead.')
        self.batch_size = self.size
      else:
        self.batch_size = batch_size
      
    # Initialize bootstrap information
    self.num_bootstraps = num_bootstraps
    assert self.num_bootstraps > 0, f'num_bootstraps must be greater than 0, but got {self.num_bootstraps}'
    self.bootstrap_proportion = bootstrap_proportion
    self.use_replacement = use_replacement
    self.elements_per_bootstrap = int(self.size * self.bootstrap_proportion)
    assert self.elements_per_bootstrap > 0, f'elements_per_bootstrap must be greater than 0, but got {self.elements_per_bootstrap}'
     
    # Create bootstrap groups
    self.key, subkey = random.split(self.key)
    self.bootstrap_inputs, self.bootstrap_targets, self.bootstrap_weights = bootstrap_dataset(
        key=subkey,
        inputs=self.inputs,
        targets=self.targets,
        weights=self.weights,
        num_bootstraps=self.num_bootstraps,
        elements_per_bootstrap=self.elements_per_bootstrap,
        use_replacement=self.use_replacement
        )

  def __len__(self) -> int:
    return self.size
  
  def __getitem__(self, idx: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get a random batch from bootstrapped data.

    Args:
        idx: Unused, required by Dataset interface

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays from bootstrapped data
    """
    self.key, subkey = random.split(self.key)
    batch_inputs, batch_targets, batched_weights = self.__select_batch(subkey)
    return batch_inputs, batch_targets, batched_weights

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select a random batch from bootstrapped data using the given key.

    Args:
        key (jax.random.PRNGKey): Random key for batch selection

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays from bootstrapped data
    """
    idx = random.choice(key, self.elements_per_bootstrap, (self.batch_size,), replace=False)
    batch_inputs = self.bootstrap_inputs[:,idx]
    batch_targets = self.bootstrap_targets[:,idx]
    batched_weights = self.bootstrap_weights[:,idx]
    return batch_inputs, batch_targets, batched_weights


def bootstrap_dataset(
    key: jax.Array,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
    num_bootstraps: int,
    elements_per_bootstrap: int,
    use_replacement: bool
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create a consistent bootstrap dataset from the given inputs, targets, and weights.
    Args:
        inputs: shape (size, input_dim)
        targets: shape (size, output_dim)
        weights: shape (size,)
        num_bootstraps: int
        elements_per_bootstrap: int
        use_replacement: bool
    Returns:
        bootstrap_inputs: shape (num_bootstraps, elements_per_bootstrap, input_dim)
        bootstrap_targets: shape (num_bootstraps, elements_per_bootstrap, output_dim)
        bootstrap_weights: shape (num_bootstraps, elements_per_bootstrap)
    """
    size = inputs.shape[0]
    input_dim = inputs.shape[1]
    output_dim = targets.shape[1]
    
    # Initialize output arrays
    bootstrap_inputs = jnp.zeros((num_bootstraps, elements_per_bootstrap, input_dim))
    bootstrap_targets = jnp.zeros((num_bootstraps, elements_per_bootstrap, output_dim))
    bootstrap_weights = jnp.zeros((num_bootstraps, elements_per_bootstrap, 1))
    
    # Create bootstrap samples
    for i in range(num_bootstraps):
        # Sample indices with or without replacement
        key, subkey = random.split(key)
        indices = random.choice(subkey, size, (elements_per_bootstrap,), replace=use_replacement)
        
        # Select data using the sampled indices
        bootstrap_inputs = bootstrap_inputs.at[i].set(inputs[indices])
        bootstrap_targets = bootstrap_targets.at[i].set(targets[indices])
        bootstrap_weights = bootstrap_weights.at[i].set(weights[indices])
    
    return bootstrap_inputs, bootstrap_targets, bootstrap_weights