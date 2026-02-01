# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad
from jax.scipy.stats import multivariate_normal

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import optax
from optax._src import linear_algebra

from scipy.stats import pearsonr
import pandas as pd
import numpy as onp

from functools import partial
import itertools
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Optional, Tuple, Iterable, Sequence, Union


# Custom Imports
from jaxmix.trainers import BaseTrainer, mdn_loss_func
from jaxmix.utils import stable_logsumexp, sample_from_gaussian_mixture

empty_frozen_dict = FrozenDict({})
identity = lambda x: x


class RNNMDN(nn.Module):
    """
    Recurrent Neural Network Mixture Density Network (RNN-MDN).
    
    Combines an RNN with an MDN output layer for probabilistic sequence modeling.
    At each time step, the RNN processes the current input and updates its hidden state,
    then the hidden state is passed to an MDN head to predict mixture parameters.
    
    Args:
        num_mixtures: Number of mixture components.
        num_output_dims: Number of output dimensions for each mixture component.
        hidden_size: Size of the RNN hidden state.
        rnn_type: Type of RNN cell to use ('GRU' or 'LSTM').
        num_rnn_layers: Number of RNN layers.
        mean_output_activation: Activation function for mixture means (default: identity).
        variance_output_activation: Activation function for mixture variances (default: softplus).
    """
    num_mixtures: int
    num_output_dims: int
    hidden_size: int
    rnn_type: str = 'GRU'
    num_rnn_layers: int = 1
    mean_output_activation: Callable = identity
    variance_output_activation: Callable = nn.softplus
    
    def setup(self):
        """Initialize RNN cells and MDN output layer."""
        # Create RNN cells based on type
        if self.rnn_type.upper() == 'GRU':
            self.rnn_cells = [nn.GRUCell(self.hidden_size) for _ in range(self.num_rnn_layers)]
        elif self.rnn_type.upper() == 'LSTM':
            self.rnn_cells = [nn.LSTMCell(self.hidden_size) for _ in range(self.num_rnn_layers)]
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}. Choose 'GRU' or 'LSTM'.")
        
        # MDN output layer
        self.mdn_head = nn.DenseGeneral(
            (self.num_mixtures, (1 + 2 * self.num_output_dims))
        )
    
    def initialize_carry(self, batch_size: int) -> Union[Tuple, jnp.ndarray]:
        """
        Initialize the RNN hidden state(s).
        
        Args:
            batch_size: Batch size for initialization.
            
        Returns:
            Initial carry/hidden state(s). For LSTM, returns tuple of (c, h) for each layer.
            For GRU/RNN, returns hidden state for each layer.
        """
        if self.rnn_type.upper() == 'LSTM':
            # LSTM needs (c, h) for each layer
            return tuple([
                (
                    jnp.zeros((batch_size, self.hidden_size)),
                    jnp.zeros((batch_size, self.hidden_size))
                )
                for _ in range(self.num_rnn_layers)
            ])
        else:
            # GRU and RNN just need hidden state for each layer
            return tuple([
                jnp.zeros((batch_size, self.hidden_size))
                for _ in range(self.num_rnn_layers)
            ])
    
    def apply_rnn_step(
        self, 
        carry: Union[Tuple, jnp.ndarray],
        x: jnp.ndarray
    ) -> Tuple[Union[Tuple, jnp.ndarray], jnp.ndarray]:
        """
        Apply one RNN step.
        
        Args:
            carry: Current hidden state(s).
            x: Input at current time step, shape (batch_size, input_dim).
            
        Returns:
            Tuple of (new_carry, hidden_output) where hidden_output is the output
            of the last RNN layer.
        """
        new_carry = []
        current_input = x
        
        for i, cell in enumerate(self.rnn_cells):
            # Flax RNN cells return (new_carry, output)
            new_carry_i, current_input = cell(carry[i], current_input)
            new_carry.append(new_carry_i)
        
        return tuple(new_carry), current_input
    
    def apply_mdn_head(
        self, 
        hidden: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply MDN head to hidden state to get mixture parameters.
        
        Args:
            hidden: Hidden state from RNN, shape (..., hidden_size).
            
        Returns:
            Tuple containing:
                - mixture_logit_weights: (..., num_mixtures, 1)
                - mixture_means: (..., num_mixtures, num_output_dims)
                - mixture_variances: (..., num_mixtures, num_output_dims)
        """
        # Apply MDN head
        flattened_output = self.mdn_head(hidden)  # (..., num_mixtures, 1 + 2*num_output_dims)
        
        # Split into components
        mixture_logit_weights = flattened_output[..., 0:1]  # (..., num_mixtures, 1)
        mixture_means = flattened_output[..., 1:self.num_output_dims + 1]  # (..., num_mixtures, num_output_dims)
        mixture_variances = flattened_output[..., -self.num_output_dims:]  # (..., num_mixtures, num_output_dims)
        
        # Apply activations
        mixture_means = self.mean_output_activation(mixture_means)
        mixture_variances = self.variance_output_activation(mixture_variances)
        
        return mixture_logit_weights, mixture_means, mixture_variances
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        initial_carry: Optional[Union[Tuple, jnp.ndarray]] = None,
        return_carry: bool = False,
        next_step_only: bool = False,
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Union[Tuple, jnp.ndarray]]
    ]:
        """
        Apply the RNN-MDN to a sequence.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim).
            initial_carry: Optional initial hidden state. If None, initialized to zeros.
            return_carry: Whether to return the final hidden state.
            
        Returns:
            If return_carry is False:
                Tuple containing sequence of:
                    - mixture_logit_weights: (batch_size, seq_len, num_mixtures, 1)
                    - mixture_means: (batch_size, seq_len, num_mixtures, num_output_dims)
                    - mixture_variances: (batch_size, seq_len, num_mixtures, num_output_dims)
            If return_carry is True:
                Tuple of (mixture_outputs, final_carry)
        """
        batch_size, seq_len = x.shape[-3:-1]
        
        # Initialize carry if not provided
        if initial_carry is None:
            carry = self.initialize_carry(batch_size)
        else:
            carry = initial_carry
        
        # Process sequence
        mixture_logit_weights_list = []
        mixture_means_list = []
        mixture_variances_list = []
        
        if next_step_only:
            # pass # TODO: Implement next step only
            # Apply RNN step
            carry, hidden = self.apply_rnn_step(carry, x[..., -1, :])
            
            # Apply MDN head
            logit_weights, means, variances = self.apply_mdn_head(hidden)
            
            mixture_logit_weights = logit_weights[..., None, :, :]  # (batch_size, 1, num_mixtures, 1)
            mixture_means = means[..., None, :, :]  # (batch_size, 1, num_mixtures, num_output_dims)
            mixture_variances = variances[..., None, :, :]  # (batch_size, 1, num_mixtures, num_output_dims)
        else:
            for t in range(seq_len):
                # Apply RNN step
                carry, hidden = self.apply_rnn_step(carry, x[..., t, :])
                
                # Apply MDN head
                logit_weights, means, variances = self.apply_mdn_head(hidden)
                
                mixture_logit_weights_list.append(logit_weights)
                mixture_means_list.append(means)
                mixture_variances_list.append(variances)
            
            # Stack along time dimension
            mixture_logit_weights = jnp.stack(mixture_logit_weights_list, axis=-3)  # (batch_size, seq_len, num_mixtures, 1)
            mixture_means = jnp.stack(mixture_means_list, axis=-3)  # (batch_size, seq_len, num_mixtures, num_output_dims)
            mixture_variances = jnp.stack(mixture_variances_list, axis=-3)  # (batch_size, seq_len, num_mixtures, num_output_dims)
        
        if return_carry:
            return (mixture_logit_weights, mixture_means, mixture_variances), carry
        else:
            return mixture_logit_weights, mixture_means, mixture_variances


class RNNMDNTrainer(BaseTrainer):
    """
    Trainer for Recurrent Neural Network Mixture Density Networks (RNN-MDN).
    
    Extends BaseTrainer to handle sequential data with teacher forcing during training
    and autoregressive rollout during prediction.
    """
    
    def __init__(
        self,
        arch: Any,
        init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        optimizer: Optional[Any] = None,
        normalize_outputs: bool = False,
        normalize_inputs: bool = False,
        masked_param_names: Optional[Sequence[str]] = None,
        key: jax.random.PRNGKey = random.PRNGKey(43),
        steps_per_check: int = 100,
    ) -> None:
        """
        Initialize RNNMDNTrainer.
        
        Args:
            arch: RNNMDN architecture
            init_batch: Initialization batch (inputs, outputs, weights)
                - inputs: shape (batch_size, seq_len, input_dim)
                - outputs: shape (batch_size, seq_len, output_dim)
                - weights: shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            optimizer: (Optional) Optax optimizer
            normalize_outputs: Whether to normalize outputs
            normalize_inputs: Whether to normalize inputs
            masked_param_names: (Optional) Names of parameters to mask for optimizer
            key: PRNGKey for parameter initialization
            steps_per_check: Steps per logging/checkpoint
        """
        super().__init__(
            arch, 
            init_batch, 
            optimizer, 
            normalize_outputs, 
            normalize_inputs, 
            masked_param_names, 
            key, 
            steps_per_check,
            initialize_apply_function=False
        )
        self.__set_apply_function(init_batch)
    
    def _apply_mdn_output_normalization(
        self, 
        pred: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        return_carry: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply output normalization to MDN predictions.
        
        Args:
            pred: Tuple of (mixture_logit_weights, mixture_means, mixture_variances)
            
        Returns:
            Normalized predictions
        """
        if return_carry:
            # pred contrains (mixture_logit_weights, mixture_means, mixture_variances), carry
            (mixture_logit_weights, mixture_means, mixture_variances), carry = pred
        else:
            mixture_logit_weights, mixture_means, mixture_variances = pred
        # Scale means by output normalization stats
        mixture_means = mixture_means * (self.output_norm_stats[1] + 1e-6) + self.output_norm_stats[0]
        # Scale variances by output normalization std squared
        mixture_variances = mixture_variances * (self.output_norm_stats[1] + 1e-6)**2
        if return_carry:
            return (mixture_logit_weights, mixture_means, mixture_variances), carry
        else:
            return mixture_logit_weights, mixture_means, mixture_variances
    
    def __set_apply_function(
        self, 
        init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> None:
        """Set up apply function with appropriate normalization."""
        inputs, outputs, _ = init_batch
        
        # Compute normalization statistics across batch and time dimensions
        if self.normalize_outputs and self.normalize_inputs:
            # Flatten batch and time dimensions for computing stats
            flat_outputs = outputs.reshape(-1, outputs.shape[-1])
            flat_inputs = inputs.reshape(-1, inputs.shape[-1])
            mu_y, sig_y = flat_outputs.mean(0, keepdims=True), flat_outputs.std(0, keepdims=True)
            mu_x, sig_x = flat_inputs.mean(0, keepdims=True), flat_inputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = mu_x, sig_x
            self.apply = lambda params, x, carry=None, kwargs=empty_frozen_dict: self._apply_mdn_output_normalization(
                self.arch.apply(params, (x - mu_x) / (sig_x + 1e-6), initial_carry=carry, **kwargs),
                return_carry=kwargs.get('return_carry', False)
            )
            self._apply_raw_outputs = lambda params, x, carry=None, kwargs=empty_frozen_dict: self.arch.apply(
                params, (x - mu_x) / (sig_x + 1e-6), initial_carry=carry, **kwargs
            )
        elif self.normalize_outputs and not self.normalize_inputs:
            flat_outputs = outputs.reshape(-1, outputs.shape[-1])
            mu_y, sig_y = flat_outputs.mean(0, keepdims=True), flat_outputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = None
            self.apply = lambda params, x, carry=None, kwargs=empty_frozen_dict: self._apply_mdn_output_normalization(
                self.arch.apply(params, x, initial_carry=carry, **kwargs),
                return_carry=kwargs.get('return_carry', False)
            )
            self._apply_raw_outputs = lambda params, x, carry=None, kwargs=empty_frozen_dict: self.arch.apply(
                params, x, initial_carry=carry, **kwargs
            )
        elif not self.normalize_outputs and self.normalize_inputs:
            flat_inputs = inputs.reshape(-1, inputs.shape[-1])
            mu_x, sig_x = flat_inputs.mean(0, keepdims=True), flat_inputs.std(0, keepdims=True)
            self.input_norm_stats = mu_x, sig_x
            self.output_norm_stats = None
            self.apply = lambda params, x, carry=None, kwargs=empty_frozen_dict: self.arch.apply(
                params, (x - mu_x) / (sig_x + 1e-6), initial_carry=carry, **kwargs
            )
            self._apply_raw_outputs = self.apply
        else:  # no normalization
            self.input_norm_stats = None
            self.output_norm_stats = None
            self.apply = lambda params, x, carry=None, kwargs=empty_frozen_dict: self.arch.apply(params, x, initial_carry=carry, **kwargs)
            self._apply_raw_outputs = self.apply
        
        # JIT compile apply functions
        self.apply = jit(self.apply, static_argnames=('kwargs',))
        self._apply_raw_outputs = jit(self._apply_raw_outputs, static_argnames=('kwargs',))
    
    def per_element_loss(
        self,
        pred: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        targets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes per-element negative log-likelihood loss for RNN-MDN over sequences.
        
        Args:
            pred: Tuple of (mixture_logit_weights, mixture_means, mixture_variances)
                Each with shape (batch_size, seq_len, num_mixtures, *)
            targets: Ground truth targets, shape (batch_size, seq_len, num_output_dims)
            
        Returns:
            Array of negative log-likelihood loss for each batch element and time step.
            Shape: (batch_size, seq_len, 1)
        """
        mixture_logit_weights, mixture_means, mixture_variances = pred
        batch_size, seq_len, num_output_dims = targets.shape
        
        # Reshape to process all time steps at once
        # Flatten batch and time dimensions
        flat_logit_weights = mixture_logit_weights.reshape(-1, *mixture_logit_weights.shape[2:])
        flat_means = mixture_means.reshape(-1, *mixture_means.shape[2:])
        flat_variances = mixture_variances.reshape(-1, *mixture_variances.shape[2:])
        flat_targets = targets.reshape(-1, num_output_dims)
        
        # Compute loss for all time steps
        flat_loss = mdn_loss_func(flat_logit_weights, flat_means, flat_variances, flat_targets)
        
        # Reshape back to (batch_size, seq_len, 1)
        loss = flat_loss.reshape(batch_size, seq_len, 1)
        
        return loss
    
    @partial(jit, static_argnums=(0,), static_argnames=('restrict_rare_event_rate', 'truncated_normal_std_limit'))
    def sample_from_mixture(
        self,
        key: jax.random.PRNGKey,
        mixture_logit_weights: jnp.ndarray,
        mixture_means: jnp.ndarray,
        mixture_variances: jnp.ndarray,
        restrict_rare_event_rate: Optional[float] = None,
        truncated_normal_std_limit: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Sample from a mixture of Gaussians.
        
        Args:
            key: jax.random.PRNGKey
            mixture_logit_weights: Array of shape (..., num_mixtures, 1)
            mixture_means: Array of shape (..., num_mixtures, num_output_dims)
            mixture_variances: Array of shape (..., num_mixtures, num_output_dims)
            restrict_rare_event_rate: (Optional) Restrict rare event rate
            truncated_normal_std_limit: (Optional) Limit the standard deviation of the truncated normal distribution to the specified value.
                If None, no limit is applied. Otherwise, the standard deviation is limited to the specified value.
        Returns:
            samples: Array of shape (..., num_output_dims)
        """
        return sample_from_gaussian_mixture(
            key,
            mixture_logit_weights,
            mixture_means,
            mixture_variances,
            restrict_rare_event_rate,
            truncated_normal_std_limit
        )
    
    #@partial(jit, static_argnums=(0,), static_argnames=('rollout_steps', 'restrict_rare_event_rate'))
    def rollout(
        self,
        params: Any,
        initial_input: jnp.ndarray,
        key: jax.random.PRNGKey,
        rollout_steps: int,
        initial_carry: Optional[Union[Tuple, jnp.ndarray]] = None,
        restrict_rare_event_rate: Optional[float] = None,
        truncated_normal_std_limit: Optional[float] = None,
        verbose: bool = True,
        difference_prediction: bool = False,
        reset_carry_interval: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Perform autoregressive rollout by sampling from the model at each step.
        
        Args:
            params: Model parameters
            initial_input: Initial input, shape (batch_size, input_dim)
            key: PRNGKey for sampling
            rollout_steps: Number of steps to roll out
            initial_carry: Optional initial hidden state
            restrict_rare_event_rate: Optional rare event rate restriction
            truncated_normal_std_limit: Optional limit on the standard deviation of the truncated normal distribution.
            verbose: Whether display progress bar during rollout computation.
            difference_prediction: Whether the model predicts the difference between the current input and the predicted output.
                If True, the predicted output is the difference between the current input and the predicted output.
                If False, the predicted output is the output itself.
                Default is False.
            reset_carry_interval: Optional interval to reset the carry.
                If None, the carry is not reset.
                If int, the carry is reset every reset_carry_interval steps.
                Default is None.
        Returns:
            Tuple containing:
                - samples: Sampled outputs, shape (batch_size, rollout_steps, num_output_dims)
                - mixture_logit_weights: shape (batch_size, rollout_steps, num_mixtures, 1)
                - mixture_means: shape (batch_size, rollout_steps, num_mixtures, num_output_dims)
                - mixture_variances: shape (batch_size, rollout_steps, num_mixtures, num_output_dims)
        """
        batch_size = initial_input.shape[0]
        
        # Initialize carry if not provided
        if initial_carry is None:
            carry = self.arch.initialize_carry(batch_size)
        else:
            carry = initial_carry
        
        # Lists to collect outputs
        samples_list = []
        logit_weights_list = []
        means_list = []
        variances_list = []
        
        current_input = initial_input
        
        if verbose:
            pbar = tqdm(range(rollout_steps), desc='Rolling out...')
        else:
            pbar = range(rollout_steps)
        for step in pbar:
            # Get mixture parameters for current input (single step)
            # Expand to sequence dimension
            current_input_seq = jnp.expand_dims(current_input, axis=1)  # (batch_size, 1, input_dim)
            
            # Apply model for one step
            # (logit_weights, means, variances), carry = self.arch.apply(
            #     params, 
            #     current_input_seq,
            #     initial_carry=carry,
            #     return_carry=True,
            #     next_step_only=True
            # )
            kwargs = FrozenDict({
                'return_carry': True,
                'next_step_only': True
            })
            (logit_weights, means, variances), carry = self.apply(params, current_input_seq, carry=carry, kwargs=kwargs)

            if reset_carry_interval is not None and step % reset_carry_interval == 0:
                carry = self.arch.initialize_carry(batch_size)
            
            # Remove sequence dimension
            logit_weights = logit_weights[:, 0, :, :]  # (batch_size, num_mixtures, 1)
            means = means[:, 0, :, :]  # (batch_size, num_mixtures, num_output_dims)
            variances = variances[:, 0, :, :]  # (batch_size, num_mixtures, num_output_dims)
            
            # Apply output normalization if needed
            if self.normalize_outputs:
                logit_weights, means, variances = self._apply_mdn_output_normalization(
                    (logit_weights, means, variances)
                )
            
            # Sample from mixture
            key, subkey = random.split(key)
            sample = self.sample_from_mixture(
                subkey, logit_weights, means, variances,
                restrict_rare_event_rate=restrict_rare_event_rate, truncated_normal_std_limit=truncated_normal_std_limit,
            )  # (batch_size, num_mixtures, num_output_dims)
            sample = sample[:, 0, :]  # (batch_size, num_output_dims)
            if difference_prediction:
                # prediction of the network is the difference between the current input and the predicted output
                # so we need to add the current input to the predicted output
                sample = sample + current_input
            
            # Store outputs
            samples_list.append(sample)
            logit_weights_list.append(logit_weights)
            means_list.append(means)
            variances_list.append(variances)
            
            # Use sampled output as next input
            current_input = sample
        
        # Stack along time dimension
        samples = jnp.stack(samples_list, axis=-2)  # (batch_size, rollout_steps, num_output_dims)
        mixture_logit_weights = jnp.stack(logit_weights_list, axis=-2)
        mixture_means = jnp.stack(means_list, axis=-2)
        mixture_variances = jnp.stack(variances_list, axis=-2)
        
        return samples, mixture_logit_weights, mixture_means, mixture_variances




import torch.utils.data as torch_data

# Dataset loader
class RNNBatchedDataset(torch_data.Dataset):
  """A dataset loader that returns random batches of data for RNN.

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
      batch_size: Optional[int] = None,
      subsequence_length: Optional[int] = None,
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
    if subsequence_length is None:
      self.subsequence_length = self.inputs.shape[1]
    else:
      if subsequence_length > self.inputs.shape[1]:
        print(f'WARNING: subsequence_length is greater than the input sequence length, will use full sequence length instead.')
        self.subsequence_length = self.inputs.shape[1]
      else:
        self.subsequence_length = subsequence_length
    # Compute the maximum start index for the subsequence
    self.max_start_idx = self.inputs.shape[1] - self.subsequence_length

  def __len__(self) -> int:
    return self.size
  
  def __getitem__(self, idx: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get a random batch of data.

    Args:
        idx: Unused, required by Dataset interface

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays
    """
    self.key, subkey_1, subkey_2 = random.split(self.key, 3)
    batch_inputs, batch_targets, batched_weights = self.__select_batch(subkey_1, subkey_2)
    return batch_inputs, batch_targets, batched_weights

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, subkey_1: jax.Array, subkey_2: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select a random batch using the given key.

    Args:
        key (jax.random.PRNGKey): Random key for batch selection

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays
    """
    # # Select random trajectories
    # idx = random.choice(subkey_1, self.size, (self.batch_size,), replace=False)
    # batch_inputs = self.inputs[idx] # (batch_size, sequence_length, input_dim)
    # batch_targets = self.targets[idx] # (batch_size, sequence_length, output_dim)
    # batched_weights = self.weights[idx] # (batch_size, sequence_length, 1)
    
    # # Select random subsequence start index for each trajectory
    # max_start_idx = self.inputs.shape[1] - self.subsequence_length
    # start_indices = random.randint(subkey_2, (self.batch_size,), 0, max_start_idx + 1)
    
    # # Extract subsequences for each trajectory using vectorized indexing
    # # Create index arrays for each dimension
    # batch_idx = jnp.arange(self.batch_size)[:, None]  # (batch_size, 1)
    # seq_idx = jnp.arange(self.subsequence_length)[None, :]  # (1, subsequence_length)
    # indices = start_indices[:, None] + seq_idx  # (batch_size, subsequence_length)
    
    # batch_inputs_subseq = batch_inputs[batch_idx, indices]  # (batch_size, subsequence_length, input_dim)
    # batch_targets_subseq = batch_targets[batch_idx, indices]  # (batch_size, subsequence_length, output_dim)
    # batched_weights_subseq = batched_weights[batch_idx, indices]  # (batch_size, subsequence_length, 1)


    # Select random trajectories
    trajectory_idx = random.choice(subkey_1, self.size, (self.batch_size,1), replace=False)

    # Select random subsequence start index for each trajectory
    start_indices = random.randint(subkey_2, (self.batch_size,), 0, self.max_start_idx + 1)

    # Extract subsequences for each trajectory using vectorized indexing
    # Create index arrays for each dimension
    seq_idx = jnp.arange(self.subsequence_length)[None, :]  # (1, subsequence_length)
    indices = start_indices[:, None] + seq_idx  # (batch_size, subsequence_length)
    
    batch_inputs_subseq = self.inputs[trajectory_idx, indices]  # (batch_size, subsequence_length, input_dim)
    batch_targets_subseq = self.targets[trajectory_idx, indices]  # (batch_size, subsequence_length, output_dim)
    batched_weights_subseq = self.weights[trajectory_idx, indices]  # (batch_size, subsequence_length, 1)
   
    return batch_inputs_subseq, batch_targets_subseq, batched_weights_subseq