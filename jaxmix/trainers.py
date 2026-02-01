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
from tqdm import trange
import matplotlib.pyplot as plt


# custom imports
from jaxmix.utils import (
    create_mask_by_name,
    plot_logs,
    apply_output_scaling,
    stable_logsumexp,
    log_normal_pdf,
    get_ensemble_cfm_div,
    sample_from_gaussian_mixture,
    )


from typing import Any, Callable, Dict, Optional, Tuple, Iterable, Sequence, Union
empty_frozen_dict = FrozenDict({})

class BaseTrainer:
    """
    Base class for neural network trainers.

    Handles model initialization, normalization, optimizer creation, application,
    logging, and training loop. Subclasses should override `per_element_loss`.
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
        ensemble_size: Optional[int] = None,
        initialize_apply_function: bool = True,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            arch: Model architecture (Flax module or similar with apply/init function)
            init_batch: Tuple of (inputs, outputs, weights) for param initialization
            optimizer: (Optional) Optax optimizer, or None to use default Adam
            normalize_outputs: Whether to normalize outputs #NOTE: this should be implemented in subclasses.
            normalize_inputs: Whether to normalize inputs
            masked_param_names: (Optional) list of parameter names to mask for optimizer
            key: PRNGKey for parameter initialization
            steps_per_check: Iterations per logging/checkpoint
            ensemble_size: (Optional) Number of ensemble members
            initialize_apply_function: Whether to initialize the apply function (True by default). Meant to be
                set to False for subclasses that implement their own apply function.
        """
        # Define model
        self.arch = arch
        self.key = key
        self.steps_per_check = steps_per_check
        self.ensemble_size = ensemble_size

        # Initialize parameters
        inputs, outputs, _ = init_batch
        assert len(inputs) == len(outputs), f'inputs and outputs must have the same length, but got {len(inputs)} and {len(outputs)}'
        self.params = self.arch.init(self.key, inputs)

        # Tabulate function for checking network architecture
        self.tabulate = lambda : self.arch.tabulate(self.key, inputs, console_kwargs={'width':120})
        
        # Vectorized functions / normalization
        self.normalize_outputs = normalize_outputs
        self.normalize_inputs = normalize_inputs
        if initialize_apply_function:
            self.__set_apply_function(init_batch)
        
        # Optimizer
        self.masked_param_names = masked_param_names
        if optimizer is None:
            lr = optax.exponential_decay(1e-3, transition_steps=1000, decay_rate=0.9, end_value=1e-5)
            self.optimizer = optax.adam(learning_rate=lr)
        else:
            self.optimizer = optimizer
        if self.masked_param_names is not None:
            self.mask = create_mask_by_name(self.params, self.masked_param_names)
            self.optimizer = optax.masked(self.optimizer, self.mask)
        self.opt_state = self.optimizer.init(self.params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log: list = []
        self.grad_norm_log: list = []
        self.test_rel_l2_log: Optional[list] = None

    def __set_apply_function(self, init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
        inputs, outputs, _ = init_batch
        if self.normalize_outputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            raise NotImplementedError('Normalization of outputs is not implemented for BaseTrainer.')
        else:
            self.output_norm_stats = None
            if self.normalize_inputs:
                mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
                self.input_norm_stats = mu_x, sig_x
                self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
            else:
                self.input_norm_stats = None
                self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
        # jits apply function for numerical consistency
        # (sometimes jitted version behaves slightly differently than non-jitted one)
        self.apply = jit(self.apply, static_argnames=('kwargs',))
        self._apply_raw_outputs = self.apply

    def per_element_loss(self, pred: Any, targets: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the loss for each element in the batch.
        Subclasses must override this method.

        Args:
            pred: Model predictions (output structure as returned by model)
            targets: Ground truth targets

        Returns:
            Per-element loss array of shape (batch_size,) or matching batch dimension
            if training a single model, or (batch_size, ensemble_size) if training an ensemble.
        """
        raise NotImplementedError('per_element_loss must be implemented in subclass')

    def loss(
        self,
        params: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        apply_kwargs: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        """
        Computes the mean weighted loss for a batch.

        Args:
            params: Model parameters (PyTree)
            batch: (inputs, targets, weights) tuple
            apply_kwargs: (Optional) Extra kwargs for model apply

        Returns:
            Scalar loss (float array)
        """
        if apply_kwargs is None:
            apply_kwargs = FrozenDict({})
        inputs, targets, weights = batch

        
        if self.normalize_outputs:
            normalized_pred = self._apply_raw_outputs(params, inputs, kwargs=apply_kwargs)
            normalized_targets = (targets - self.output_norm_stats[0]) / (self.output_norm_stats[1] + 1e-6)
            per_element_loss_values = self.per_element_loss(normalized_pred, normalized_targets) # shape (batch_dim,)
        else:
            pred = self.apply(params, inputs, kwargs=apply_kwargs)
            per_element_loss_values = self.per_element_loss(pred, targets) # shape (batch_dim,)
        return jnp.mean(weights * per_element_loss_values) # scalar

    # Define a compiled update step
    @partial(jit, static_argnums=(0,), static_argnames=('apply_kwargs',))
    def step(
        self,
        params: Any,
        opt_state: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        apply_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Performs a gradient step.

        Args:
            params: Model parameters (PyTree)
            opt_state: Optimizer state
            batch: (inputs, targets, weights) tuple
            apply_kwargs: (Optional) Extra kwargs for model apply

        Returns:
            Tuple of (updated_params, updated_opt_state, grads)
        """
        grads = grad(self.loss)(params, batch, apply_kwargs=apply_kwargs)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads

    def train(
        self,
        dataset: Iterable[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        nIter: int = 10000,
        apply_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Runs the main training loop.

        Args:
            dataset: Iterable data loader yielding batches (inputs, targets, weights)
            nIter: Number of training iterations
            apply_kwargs: (Optional) Extra kwargs for model apply

        Returns:
            None
        """
        if apply_kwargs is None:
            apply_kwargs = FrozenDict({})
        else:
            # convert apply_kwargs to FrozenDict so that it is hashable
            apply_kwargs = FrozenDict(apply_kwargs)
        data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(data)
            self.params, self.opt_state, grads = self.step(
                self.params, self.opt_state, batch, apply_kwargs=apply_kwargs)
            # Logger
            if it % self.steps_per_check == 0:
                l = self.loss(self.params, batch, apply_kwargs=apply_kwargs)
                g_norm = linear_algebra.global_norm(grads).squeeze()
                self.loss_log.append(l)
                self.grad_norm_log.append(g_norm)
                pbar.set_postfix({'loss': l, 'grad_norm': jnp.mean(jnp.array(g_norm))})
        pbar.close()
    
    def plot_logs(self, window: Optional[int] = None) -> None:
        """
        Plots training logs.

        Args:
            window: Moving average window for smoothing (optional)

        Returns:
            None
        """
        plot_logs(self.loss_log, self.grad_norm_log, self.test_rel_l2_log,
                  window=window, steps_per_check=self.steps_per_check)


class MSETrainer(BaseTrainer):
    """
    Mean Squared Error (MSE) trainer, extending BaseTrainer.
    Implements per-element loss for MSE.
    """
    def __init__(
        self,
        arch: Any,
        init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        optimizer: Optional[Any]=None,
        normalize_outputs: bool = False,
        normalize_inputs: bool = False,
        masked_param_names: Optional[Sequence[str]] = None,
        key: jax.random.PRNGKey = random.PRNGKey(43),
        steps_per_check: int = 100
    ) -> None:
        """
        Initialize MSETrainer.

        Args:
            arch: Model architecture (must output MSETuple structure)
            init_batch: Initialization batch (inputs, outputs, weights)
            optimizer: (Optional) Optax optimizer
            normalize_outputs: Whether to normalize outputs
            normalize_inputs: Whether to normalize inputs
            masked_param_names: (Optional) Names of parameters to mask for optimizer
            key: PRNGKey for parameter initialization
            steps_per_check: Steps per logging/checkpoint

        Returns:
            None
        """
        super().__init__(arch, init_batch, optimizer, normalize_outputs, normalize_inputs, masked_param_names, key, steps_per_check, initialize_apply_function=False)
        self.__set_apply_function(init_batch)

    def __set_apply_function(self, init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
        inputs, outputs, _ = init_batch
        if self.normalize_outputs and self.normalize_inputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = mu_x, sig_x
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)*(sig_y + 1e-6) + mu_y
            self._apply_raw_outputs = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
        elif self.normalize_outputs and not self.normalize_inputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)*(sig_y + 1e-6) + mu_y
            self._apply_raw_outputs = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
        elif not self.normalize_outputs and self.normalize_inputs:
            mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
            self.input_norm_stats = mu_x, sig_x
            self.output_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
            self._apply_raw_outputs = self.apply
        else: # no normalization
            self.input_norm_stats = None
            self.output_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
            self._apply_raw_outputs = self.apply
        # jits apply function for numerical consistency
        # (sometimes jitted version behaves slightly differently than non-jitted one)
        self.apply = jit(self.apply, static_argnames=('kwargs',))
        self._apply_raw_outputs = jit(self._apply_raw_outputs, static_argnames=('kwargs',))

    def per_element_loss(
        self,
        pred: jnp.ndarray,
        targets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes per-element mean squared error loss.

        Args:
            pred: Model predictions, shape (batch_size, num_output_dims) if single model, or (batch_size, ensemble_size, num_output_dims) if ensemble.
            targets: Ground truth targets, shape (batch_size, num_output_dims)

        Returns:
            Array of mean squared error loss for each batch element. Shape (batch_size, 1) if single model, or (batch_size, ensemble_size, 1) if ensemble.
        """
        return jnp.mean((pred - targets) ** 2, axis=-1, keepdims=True)


class MDNTrainer(BaseTrainer):
    """
    Mixture Density Network (MDN) trainer, extending BaseTrainer.
    Implements per-element loss for MDN and sampling from Gaussian mixture models.
    """

    def __init__(
        self,
        arch: Any,
        init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        optimizer: Optional[Any]=None,
        normalize_outputs: bool = False,
        normalize_inputs: bool = False,
        masked_param_names: Optional[Sequence[str]] = None,
        key: jax.random.PRNGKey = random.PRNGKey(43),
        steps_per_check: int = 100,
    ) -> None:
        """
        Initialize MDNTrainer.

        Args:
            arch: Model architecture (must output MDN tuple structure)
            init_batch: Initialization batch (inputs, outputs, weights)
            optimizer: (Optional) Optax optimizer
            normalize_outputs: Whether to normalize outputs
            normalize_inputs: Whether to normalize inputs
            masked_param_names: (Optional) Names of parameters to mask for optimizer
            key: PRNGKey for parameter initialization
            steps_per_check: Steps per logging/checkpoint

        Returns:
            None
        """
        super().__init__(arch, init_batch, optimizer, normalize_outputs, normalize_inputs, masked_param_names, key, steps_per_check, initialize_apply_function=False)
        self.__set_apply_function(init_batch)

    def _apply_mdn_output_normalization(self, pred):
        # keep logit weights unchanged
        # scale means by output normalization stats
        # scale variances by output normalization std squared
        mixture_logit_weights, mixture_means, mixture_variances = pred
        mixture_means = mixture_means * (self.output_norm_stats[1] + 1e-6) + self.output_norm_stats[0]
        mixture_variances = mixture_variances * (self.output_norm_stats[1] + 1e-6)**2
        return mixture_logit_weights, mixture_means, mixture_variances

    def __set_apply_function(self, init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
        inputs, outputs, _ = init_batch
        if self.normalize_outputs and self.normalize_inputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = mu_x, sig_x
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self._apply_mdn_output_normalization(self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs))
            self._apply_raw_outputs = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
        elif self.normalize_outputs and not self.normalize_inputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self._apply_mdn_output_normalization(self.arch.apply(params, x, **kwargs))
            self._apply_raw_outputs = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
        elif not self.normalize_outputs and self.normalize_inputs:
            mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
            self.input_norm_stats = mu_x, sig_x
            self.output_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
            self._apply_raw_outputs = self.apply
        else: # no normalization
            self.input_norm_stats = None
            self.output_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
            self._apply_raw_outputs = self.apply
        # jits apply function for numerical consistency
        # (sometimes jitted version behaves slightly differently than non-jitted one)
        self.apply = jit(self.apply, static_argnames=('kwargs',))
        self._apply_raw_outputs = jit(self._apply_raw_outputs, static_argnames=('kwargs',))

    def per_element_loss(
        self,
        pred: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        targets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes per-element negative log-likelihood loss for MDN.

        Args:
            pred: Tuple of (mixture_logit_weights, mixture_means, mixture_variances)
            targets: Ground truth targets, shape (batch_size, num_output_dims)

        Returns:
            Array of negative log-likelihood loss for each batch element.
        """
        mixture_logit_weights, mixture_means, mixture_variances = pred
        assert mixture_means.shape == mixture_variances.shape, (
            f'mixture means and variances must have the same shape, '
            f'but got {mixture_means.shape} and {mixture_variances.shape}'
        )
        return mdn_loss_func(mixture_logit_weights, mixture_means, mixture_variances, targets)

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


class ConditionalFlowMatchingTrainer(BaseTrainer):
    """
    Conditional Flow Matching (CFM) trainer, extending BaseTrainer.
    Implements per-element loss for flow matching and sample generation via ODE integration.
    """

    def __init__(
        self,
        arch: Any,
        init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        optimizer: Optional[Any]=None,
        normalize_outputs: bool = False,
        normalize_inputs: bool = False,
        masked_param_names: Optional[Sequence[str]] = None,
        key: jax.random.PRNGKey = random.PRNGKey(43),
        steps_per_check: int = 100,
        sigma: float = 0.0,
        ensemble_size: Optional[int] = None,
    ) -> None:
        """
        Initialize ConditionalFlowMatchingTrainer.

        Args:
            arch: Model architecture (velocity field network)
            init_batch: Initialization batch (inputs, outputs, weights)
            optimizer: (Optional) Optax optimizer
            normalize_outputs: Whether to normalize outputs
            normalize_inputs: Whether to normalize inputs
            masked_param_names: (Optional) Names of parameters to mask for optimizer
            key: PRNGKey for parameter initialization
            steps_per_check: Steps per logging/checkpoint
            sigma: Standard deviation for conditional flow matching (default 0.0 for OT-CFM)

        Returns:
            None
        """
        super().__init__(arch, init_batch, optimizer, normalize_outputs, normalize_inputs, masked_param_names, key, steps_per_check, ensemble_size=ensemble_size, initialize_apply_function=False)
        self.sigma = sigma
        self.__set_apply_function(init_batch)

    def __set_apply_function(self, init_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
        inputs, outputs, _ = init_batch
        if self.normalize_outputs and self.normalize_inputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = mu_x, sig_x
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)*(sig_y + 1e-6) + mu_y
            self._apply_raw_outputs = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
        elif self.normalize_outputs and not self.normalize_inputs:
            mu_y, sig_y = outputs.mean(0, keepdims=True), outputs.std(0, keepdims=True)
            self.output_norm_stats = mu_y, sig_y
            self.input_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)*(sig_y + 1e-6) + mu_y
            self._apply_raw_outputs = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
        elif not self.normalize_outputs and self.normalize_inputs:
            mu_x, sig_x = inputs.mean(0, keepdims=True), inputs.std(0, keepdims=True)
            self.input_norm_stats = mu_x, sig_x
            self.output_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, (x-mu_x)/(sig_x+1e-6), **kwargs)
            self._apply_raw_outputs = self.apply
        else: # no normalization
            self.input_norm_stats = None
            self.output_norm_stats = None
            self.apply = lambda params, x, kwargs=empty_frozen_dict : self.arch.apply(params, x, **kwargs)
            self._apply_raw_outputs = self.apply
        # jits apply function for numerical consistency
        # (sometimes jitted version behaves slightly differently than non-jitted one)
        self.apply = jit(self.apply, static_argnames=('kwargs',))
        self._apply_raw_outputs = jit(self._apply_raw_outputs, static_argnames=('kwargs',))

    def per_element_loss(
        self,
        pred: jnp.ndarray,
        targets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes per-element flow matching loss.

        The model is expected to be called with concatenated inputs [x, y_t, t],
        where x is the conditioning variable, y_t is the interpolated state, and t is time.
        The targets should be the conditional velocity field u_t(y_t | x, y_1).

        Args:
            pred: Predicted velocity field, shape (batch_size, num_output_dims)
            targets: Target velocity field, shape (batch_size, num_output_dims)

        Returns:
            Array of mean squared error loss for each batch element, shape (batch_size, 1)
        """
        return jnp.mean((pred - targets) ** 2, axis=-1, keepdims=True)

    #@partial(jit, static_argnums=(0, 4, 5))
    def generate_sample(
        self,
        key: jax.random.PRNGKey,
        params: Any,
        x: jnp.ndarray,
        num_output_dims: int,
        num_steps: int = 100,
        apply_kwargs: Optional[Dict[str, Any]] = empty_frozen_dict,
        progress_bar: bool = True,
        noise_mean: Optional[jnp.ndarray] = None,
        noise_std: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Generate samples by integrating the learned velocity field from t=0 to t=1.

        Args:
            key: jax.random.PRNGKey for sampling initial noise
            params: Model parameters
            x: Conditioning variables, shape (batch_size, num_input_dims)
            num_output_dims: Number of output dimensions
            num_steps: Number of integration steps (default 100)
            apply_kwargs: Apply kwargs for the model
            progress_bar: Whether to show a progress bar
            noise_mean: Mean of the noise distribution (default 0)
            noise_std: Standard deviation of the noise distribution (default 1)
        Returns:
            Generated samples, shape (batch_size, num_output_dims)
        """

        if noise_mean is None:
            noise_mean = jnp.zeros((num_output_dims,))
        if noise_std is None:
            noise_std = jnp.ones((num_output_dims,))
        
        # Sample initial noise y_0 ~ N(0, I)
        y_0 = random.normal(key, shape=(*x.shape[:-1], num_output_dims)) * noise_std + noise_mean
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        y_t = y_0
        
        if progress_bar:
            pbar = trange(num_steps, desc="Generating samples")
        else:
            pbar = range(num_steps)
        for step in pbar:
            y_t = self._euler_step(params, x, y_t, step * dt, dt, apply_kwargs=apply_kwargs)
        
        return y_t

    def compute_velocity_field(
        self,
        params: Any,
        x: jnp.ndarray,
        y_t: jnp.ndarray,
        t: Union[float, jnp.ndarray],
        apply_kwargs: Optional[Dict[str, Any]] = empty_frozen_dict,
    ) -> jnp.ndarray:
        """
        Compute the velocity field at the given time.
        Patterned as [f(x, y_t, t)f]' = u_t(y_t | x, y_1)
        """
        if jnp.isscalar(t):
            t_array = jnp.full((*x.shape[:-1], 1), t)
        else:
            # t should be a shape (..., 1) array
            assert t.shape == (*x.shape[:-1], 1), f't must be a float have shape {(*x.shape[:-1], 1)}, but got {t.shape}'
            t_array = t
        model_input = jnp.concatenate([x, y_t, t_array], axis=-1)
        return self.apply(params, model_input, kwargs=apply_kwargs)

    #@partial(jit, static_argnums=(0,), static_argnames=('apply_kwargs',))
    def compute_flow_divergence(
        self,
        params: Any,
        x: jnp.ndarray,
        y_t: jnp.ndarray,
        t: Union[float, jnp.ndarray],
        apply_kwargs: Optional[Dict[str, Any]] = empty_frozen_dict,
    ) -> jnp.ndarray:
        """
        Compute the flow divergence at the given time.
        """
        # remove tile_inputs from apply_kwargs
        velocity_field_fn = lambda x, y_t, t : self.compute_velocity_field(params, x, y_t, t)
        return get_ensemble_cfm_div(velocity_field_fn, x, y_t, t, ensemble_size=self.ensemble_size)

    @partial(jit, static_argnums=(0,), static_argnames=('apply_kwargs',))
    def _euler_step(
        self,
        params: Any,
        x: jnp.ndarray,
        y_t: jnp.ndarray,
        t: float,
        dt: float,
        apply_kwargs: Optional[Dict[str, Any]] = empty_frozen_dict,
    ) -> jnp.ndarray:
        """
        Perform one Euler step.
        Patterned as y_t+1 = y_t + dt * u_t(y_t | x, y_1)
        """
        v_t = self.compute_velocity_field(params, x, y_t, t, apply_kwargs=apply_kwargs)
        return y_t + dt * v_t

    def compute_nll(
        self,
        params: Any,
        x: jnp.ndarray,
        y_1: jnp.ndarray,
        num_steps: int = 100,
        apply_kwargs: Optional[Dict[str, Any]] = empty_frozen_dict,
        progress_bar: bool = True,
        noise_mean: Optional[jnp.ndarray] = None,
        noise_std: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute the negative log-likelihood of the samples (given as y_1). This is done by integrating the divergence of the
        velocity field backwards in time from t=1 to t=0.

        Args:
            params: Model parameters
            x: Conditioning variables, shape (batch_size, num_input_dims)
            y_1: Samples, shape (batch_size, num_output_dims)
            num_steps: Number of integration steps (default 100)
            apply_kwargs: Apply kwargs for the model
            progress_bar: Whether to show a progress bar
            noise_mean: Mean of the noise distribution (default 0)
            noise_std: Standard deviation of the noise distribution (default 1)
        Returns:
            Negative log-likelihood, shape (batch_size, 1)
        """
        dt = 1.0 / num_steps

        y_t = y_1 # shape (batch_size, num_output_dims)
        f_t = jnp.zeros((*y_1.shape[:-1], 1)) # shape (batch_size, 1)

        if progress_bar:
            pbar = trange(num_steps, desc="Computing NLL")
        else:
            pbar = range(num_steps)
        for step in pbar:
            t_array = jnp.full((*y_1.shape[:-1], 1), 1.0 - step * dt)
            y_t, f_t = self._nll_euler_step(params, x, y_t, f_t, t_array, dt, apply_kwargs=apply_kwargs)


        if noise_mean is None:
            noise_mean = jnp.zeros_like(y_t)
        else:
            noise_mean = jnp.broadcast_to(noise_mean, y_t.shape)
        if noise_std is None:
            noise_log_std = jnp.zeros_like(y_t)
        else:
            noise_log_std = jnp.log(noise_std)
            noise_log_std = jnp.broadcast_to(noise_log_std, y_t.shape)
        y_0_nll = -log_normal_pdf(y_t, noise_mean, noise_log_std) # shape (batch_size, 1) # assume y_0 ~ N(noise_mean, noise_log_std)

        return y_0_nll + f_t # shape (batch_size, 1)

    #@partial(jit, static_argnums=(0,), static_argnames=('apply_kwargs',))
    @partial(jit, static_argnums=(0,), static_argnames=('apply_kwargs',))
    def _nll_euler_step(
        self,
        params: Any,
        x: jnp.ndarray,
        y_t: jnp.ndarray,
        f_t: jnp.ndarray,
        t: float,
        dt: float,
        apply_kwargs: Optional[Dict[str, Any]] = empty_frozen_dict,
    ) -> jnp.ndarray:
        """
        Perform one Euler step for the negative log-likelihood.
        """
        v_t = self.compute_velocity_field(params, x, y_t, t, apply_kwargs=apply_kwargs)
        div_v_t = self.compute_flow_divergence(params, x, y_t, t, apply_kwargs=apply_kwargs)
        f_t = f_t + dt * div_v_t # shape (batch_size, 1)
        y_t = y_t - dt * v_t # shape (batch_size, num_output_dims)

        return y_t, f_t

@jit
def mdn_loss_func(
    mixture_logit_weights: jnp.ndarray,
    mixture_means: jnp.ndarray,
    mixture_variances: jnp.ndarray,
    y: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Compute negative log-likelihood loss for a Gaussian mixture (MDN loss).

    Args:
        mixture_logit_weights: Array of shape (..., num_mixtures, 1)
        mixture_means: Array of shape (..., num_mixtures, num_output_dims)
        mixture_variances: Array of shape (..., num_mixtures, num_output_dims)
        y: Target values of shape (..., num_output_dims)
        eps: Small epsilon to avoid log(0)

    Returns:
        Negative log-likelihood, shape (..., 1)
    """
    # compute the log of the standard deviation and add a small epsilon to avoid log(0)
    log_std = jnp.log(mixture_variances + eps) / 2  # shape (..., num_mixtures, num_output_dims)
    # compute the log probability density function (PDF) of the normal distribution for each mixture
    log_pdf = log_normal_pdf(jnp.expand_dims(y, axis=-2), mixture_means, log_std)  # (..., num_mixtures, 1)
    # compute the -logsumexp of the mixture logit weights
    denominator_log = stable_logsumexp(mixture_logit_weights, axis=-2, keepdims=True)  # (..., 1, 1)
    # compute argument of the log-sum-exp
    logsumexp_arg = mixture_logit_weights - denominator_log + log_pdf  # (..., num_mixtures, 1)
    # compute the log-sum-exp
    logsumexp_term = stable_logsumexp(logsumexp_arg, axis=-2) # shape (..., 1)
    return -logsumexp_term # shape (..., 1)