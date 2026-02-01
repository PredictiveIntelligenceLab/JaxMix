"""
Utility functions for training MDN and CFM models on the sine inverse problem
and computing NLL/likelihood values and plots.
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt
from flax.core import FrozenDict
from typing import Optional, Tuple, Dict, Any

from jaxmix.archs import MLP, MDN, Ensemble
from jaxmix.trainers import MDNTrainer, ConditionalFlowMatchingTrainer, mdn_loss_func
from jaxmix.utils import split_data
from jaxmix.data_loaders import BatchedDataset, FlowMatchingDataset

from utils import generate_data


#########################################################
############### Ground Truth Functions ##################
#########################################################

def get_marginal_y_density(ys, dx=0.001, sigma=0.2, gt_forward=lambda x: 0.7*jnp.sin(5*x) + 0.5*x):
    """
    Computes the marginal density of y.

    Args:
        ys: jnp.ndarray, shape (...)
        dx: float, step size in x
        sigma: float, standard deviation of the Gaussian
        gt_forward: function, forward function of the ground truth

    Returns:
        jnp.ndarray, shape (...)
    """
    # y is shape (...) and dt is a scalar
    num_xs = int(3 / dx) # assumes x is uniformly distributed in [-1.5, 1.5]
    xs = jnp.linspace(-1.5, 1.5, num_xs)
    
    # Get the marginal of y
    marginal_ys = jnp.expand_dims(ys, axis=-1) # shape (..., 1)
    pdfs = jax.scipy.stats.norm.pdf(marginal_ys, gt_forward(xs), sigma) # shape (num_xs,)
    return pdfs.mean(-1) # shape (...)


def get_conditional_x_nll(xs, ys, dx=0.001, sigma=0.2, gt_forward=lambda x: 0.7*jnp.sin(5*x) + 0.5*x, mode='nll'):
    """
    Compute the conditional NLL of x given y.

    Args:
        xs: jnp.ndarray, shape (num_xs,)
        ys: jnp.ndarray, shape (num_ys,)
        dx: float, step size in x for computing marginal pdf of y
        sigma: float, standard deviation of the Gaussian noise in the forward model
        gt_forward: function, forward function of the ground truth
        mode: str, 'nll' or 'likelihood'
    Returns:
        jnp.ndarray, shape (num_ys, num_xs)
    """
    # compute marginal pdf of x and y
    marginal_y_density = get_marginal_y_density(ys, dx, sigma, gt_forward) # shape (num_ys,)
    marginal_x_density = jnp.where(jnp.abs(xs) > 1.5, 0.0, 1/3) # shape (num_xs,)
    
    # compute conditional pdf of x given y
    expanded_xs = jnp.expand_dims(xs, axis=-2) # shape (1, num_xs)
    expanded_ys = jnp.expand_dims(ys, axis=-1) # shape (num_ys, 1)
    pdf_y_conditioned_on_x = jax.scipy.stats.norm.pdf(expanded_ys, gt_forward(expanded_xs), sigma) # shape (num_ys, num_xs)
    pdf_x_conditioned_on_y = pdf_y_conditioned_on_x * marginal_x_density[None, :] / marginal_y_density[:, None] # shape (num_ys, num_xs)
    
    if mode == 'nll':
        # take negative log and return
        conditional_nll = -jnp.log(pdf_x_conditioned_on_y) # shape (num_ys, num_xs)
        return conditional_nll
    elif mode == 'likelihood':
        return pdf_x_conditioned_on_y
    else:
        raise ValueError(f"Invalid mode: {mode}")

def compute_conditional_kl_divergence(dist_1, dist_2, endpoints=(-1.5, 1.5)):
    """
    Compute the KL divergence between two conditional distributions.

    Args:
        dist_1: jnp.ndarray, shape (num_ys, num_xs)
        dist_2: jnp.ndarray, shape (num_ys, num_xs)
        endpoints: tuple, (start, end) of the x-axis

    Returns:
        jnp.ndarray, shape (num_ys,)
    """
    dx = (endpoints[1] - endpoints[0]) / dist_1.shape[1]
    return jnp.sum(dist_1 * (jnp.log(dist_1) - jnp.log(dist_2)), axis=-1) * dx # shape (num_ys,)

#########################################################
############### Default Configurations ##################
#########################################################

def get_default_mdn_config(out_dim: int = 1, hidden_features: int = 128, depth: int = 5, num_mixtures: int = 8) -> Dict[str, Any]:
    """
    Returns the default configuration for the MDN model.
    
    Args:
        out_dim: Output dimensionality.
        hidden_features: Number of hidden features in the MLP.
        depth: Depth of the MLP.
    Returns:
        Dictionary containing architecture and optimizer configurations.
    """
    base_arch = MLP(features=[hidden_features] * depth)
    ensemble_size = 12
    backbone_arch = Ensemble(base_arch, ensemble_size)
    mdn_arch = MDN(
        num_mixtures=num_mixtures,
        num_output_dims=out_dim,
        backbone=backbone_arch,
        ensemble_size=ensemble_size
    )
    
    # Optimizer with warmup + exponential decay
    peak_lr = 5e-3
    warmup_steps = 100
    lr = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=peak_lr,
                transition_steps=warmup_steps
            ),
            optax.exponential_decay(
                init_value=peak_lr,
                transition_steps=1_000,
                decay_rate=0.9,
            ),
        ],
        boundaries=[warmup_steps]
    )
    optimizer = optax.chain(
        optax.adaptive_grad_clip(0.001),
        optax.adamw(learning_rate=lr, weight_decay=1e-1)
    )
    
    return {
        'arch': mdn_arch,
        'optimizer': optimizer,
        'ensemble_size': ensemble_size,
    }


def get_default_cfm_config(out_dim: int = 1, hidden_features: int = 128, depth: int = 5) -> Dict[str, Any]:
    """
    Returns the default configuration for the CFM model.
    
    Args:
        out_dim: Output dimensionality.
        hidden_features: Number of hidden features in the MLP.
        depth: Depth of the MLP.
    Returns:
        Dictionary containing architecture and optimizer configurations.
    """
    cfm_base_arch = MLP(features=[hidden_features] * depth + [out_dim])
    ensemble_size = 12
    cfm_arch = Ensemble(cfm_base_arch, ensemble_size)
    
    # Optimizer with warmup + exponential decay
    peak_lr = 1e-3
    warmup_steps = 1000
    lr = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=peak_lr,
                transition_steps=warmup_steps
            ),
            optax.exponential_decay(
                init_value=peak_lr,
                transition_steps=1_000,
                decay_rate=0.9,
            ),
        ],
        boundaries=[warmup_steps]
    )
    optimizer = optax.chain(
        optax.adaptive_grad_clip(0.01),
        optax.adamw(learning_rate=lr, weight_decay=1e-2)
    )
    
    return {
        'arch': cfm_arch,
        'optimizer': optimizer,
        'ensemble_size': ensemble_size,
    }


#########################################################
############### MDN Training and Evaluation #############
#########################################################

def train_mdn_model(
    n_train: int,
    key: jax.Array,
    n_val: int = 10_000,
    config: Optional[Dict[str, Any]] = None,
    n_iter: int = 20_000,
    hidden_features: int = 128,
    depth: int = 5,
    batch_size: Optional[int] = None,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    plot_training_data: bool = False,
    plot_logs: bool = False,
) -> Tuple[MDNTrainer, Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Train an MDN model on the sine inverse problem with the specified training dataset size.
    
    Args:
        n_train: Number of training samples.
        key: JAX PRNGKey for reproducibility.
        n_val: Number of validation samples (default 10_000).
        config: Optional dictionary with 'arch' and 'optimizer' keys. If None, uses default config.
        n_iter: Number of training iterations (default 20_000).
        hidden_features: Number of hidden features in the MLP.
        depth: Depth of the MLP.
        batch_size: Batch size for training. If None, uses min(256, n_train//20).
        normalize_inputs: Whether to normalize inputs (default True).
        normalize_outputs: Whether to normalize outputs (default True).
        plot_training_data: Whether to plot the training data (default False).
        plot_logs: Whether to plot training logs after training (default False).
        
    Returns:
        Tuple of:
          - model: Trained MDNTrainer instance.
          - train_data: Tuple of (train_inputs, train_outputs, train_weights).
          - test_data: Tuple of (test_inputs, test_outputs, test_weights).
          - key: Updated JAX PRNGKey.
    """
    if config is None:
        config = get_default_mdn_config(hidden_features=hidden_features, depth=depth)
    
    arch = config['arch']
    optimizer = config['optimizer']
    
    # Generate and split data
    key, subkey = random.split(key)
    outputs, inputs = generate_data(n_train + n_val, key=subkey)
    key, subkey = random.split(key)
    train_data, test_data = split_data(subkey, inputs, outputs, n_train, n_val)
    train_inputs, train_outputs, train_weights = train_data
    test_inputs, test_outputs, test_weights = test_data
    
    if plot_training_data:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(train_inputs.squeeze(), train_outputs.squeeze(), s=1)
        ax.set_title(f"Training data: {n_train:,} samples")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        plt.show()
    
    # Initialize model
    init_batch = (train_inputs, train_outputs, train_weights)
    key, subkey = random.split(key)
    model = MDNTrainer(
        arch, init_batch,
        key=subkey,
        optimizer=optimizer,
        normalize_inputs=normalize_inputs,
        normalize_outputs=normalize_outputs
    )
    
    # Train model
    key, subkey = random.split(key)
    if batch_size is None:
        batch_size = min(256, n_train // 20)
    train_loader = BatchedDataset(train_data, subkey, batch_size=batch_size)
    model.train(train_loader, nIter=n_iter)
    
    if plot_logs:
        model.plot_logs()
    
    return model, train_data, test_data, key


def compute_mdn_nll_likelihood(
    model: MDNTrainer,
    test_inputs: jax.Array,
    test_outputs: jax.Array,
    grid_size: int = 200,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    plot: bool = True,
    plot_training_data: bool = False,
    train_inputs: Optional[jax.Array] = None,
    train_outputs: Optional[jax.Array] = None,
    return_values: bool = False,
    nll_vmin: Optional[float] = None,
    nll_vmax: Optional[float] = None,
    likelihood_vmin: Optional[float] = None,
    likelihood_vmax: Optional[float] = None,
) -> Optional[Dict[str, jax.Array]]:
    """
    Compute NLL and likelihood surfaces from a trained MDN model and optionally plot them.
    
    Args:
        model: Trained MDNTrainer instance.
        test_inputs: Test inputs for defining the grid range (if x_range/y_range not specified).
        test_outputs: Test outputs for defining the grid range (if x_range/y_range not specified).
        grid_size: Resolution of the grid (default 200).
        x_range: Optional tuple (x_min, x_max) for the input range.
        y_range: Optional tuple (y_min, y_max) for the output range.
        plot: Whether to plot the NLL and likelihood surfaces (default True).
        plot_training_data: Whether to overlay training data on plots (default False).
        train_inputs: Training inputs (required if plot_training_data=True).
        train_outputs: Training outputs (required if plot_training_data=True).
        return_values: Whether to return the computed values (default False).
        nll_vmin: Optional minimum value for NLL colorbar scale (default None, auto-scale).
        nll_vmax: Optional maximum value for NLL colorbar scale (default None, auto-scale).
        likelihood_vmin: Optional minimum value for likelihood colorbar scale (default None, auto-scale).
        likelihood_vmax: Optional maximum value for likelihood colorbar scale (default None, auto-scale).
        
    Returns:
        If return_values=True, returns a dictionary with:
          - 'xx': Grid x coordinates.
          - 'yy': Grid y coordinates.
          - 'per_element_nll': Per-ensemble-member NLL values.
          - 'avg_nll': Average NLL across ensemble.
          - 'ensemble_nll': Ensemble NLL (from averaged likelihoods).
          - 'per_element_likelihood': Per-ensemble-member likelihood values.
          - 'avg_likelihood': Average likelihood across ensemble.
        Otherwise returns None.
    """
    # Define grid
    if x_range is None:
        x_range = (float(test_inputs.min()), float(test_inputs.max()))
    if y_range is None:
        y_range = (float(test_outputs.min()), float(test_outputs.max()))
    
    x_grid = jnp.linspace(x_range[0], x_range[1], grid_size)
    y_grid = jnp.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    xy_inputs = xx.reshape(-1, 1)
    xy_outputs = yy.reshape(-1, 1)
    
    # Compute NLL
    pred = model.apply(model.params, xy_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    per_element_nll = mdn_loss_func(mixture_logit_weights, mixture_means, mixture_variances, xy_outputs)
    
    # Compute derived quantities
    avg_per_element_nll = per_element_nll.mean(axis=0)
    per_element_likelihood = jnp.exp(-per_element_nll)
    avg_per_element_likelihood = per_element_likelihood.mean(axis=0)
    
    # Ensemble NLL: average likelihoods, then take -log
    ensemble_likelihood = per_element_likelihood.mean(axis=0)
    ensemble_nll = -jnp.log(ensemble_likelihood)
    
    if plot:
        # Plot average NLL surface
        nll_grid = avg_per_element_nll.reshape(xx.shape)
        plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(xx, yy, nll_grid, shading='auto', cmap='jet', vmin=nll_vmin, vmax=nll_vmax)
        plt.colorbar(pcm, label="NLL")
        if plot_training_data and train_inputs is not None and train_outputs is not None:
            plt.scatter(train_inputs, train_outputs, s=1, c='white', alpha=0.6, label="Training data")
            plt.legend()
        plt.xlabel("Forward Output (y)")
        plt.ylabel("Forward Input (x)")
        plt.title("Average NLL(x|y) surface from trained MDN")
        plt.show()
        
        # Plot average likelihood surface
        likelihood_grid = avg_per_element_likelihood.reshape(xx.shape)
        plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(xx, yy, likelihood_grid, shading='auto', cmap='jet', vmin=likelihood_vmin, vmax=likelihood_vmax)
        plt.colorbar(pcm, label="Likelihood")
        if plot_training_data and train_inputs is not None and train_outputs is not None:
            plt.scatter(train_inputs, train_outputs, s=1, c='white', alpha=0.6, label="Training data")
            plt.legend()
        plt.xlabel("Forward Output (y)")
        plt.ylabel("Forward Input (x)")
        plt.title("Average Likelihood P(x|y) surface from trained MDN")
        plt.show()
        
        # Plot ensemble NLL surface
        ensemble_nll_grid = ensemble_nll.reshape(xx.shape)
        plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(xx, yy, ensemble_nll_grid, shading='auto', cmap='jet', vmin=nll_vmin, vmax=nll_vmax)
        plt.colorbar(pcm, label="NLL")
        if plot_training_data and train_inputs is not None and train_outputs is not None:
            plt.scatter(train_inputs, train_outputs, s=1, c='white', alpha=0.6, label="Training data")
            plt.legend()
        plt.xlabel("Forward Output (y)")
        plt.ylabel("Forward Input (x)")
        plt.title("Ensemble NLL(x|y) surface from trained MDN (averaged likelihoods)")
        plt.show()
    
    if return_values:
        return {
            'xx': xx,
            'yy': yy,
            'per_element_nll': per_element_nll.reshape(per_element_nll.shape[0], *xx.shape),
            'avg_nll': avg_per_element_nll.reshape(xx.shape),
            'ensemble_nll': ensemble_nll.reshape(xx.shape),
            'per_element_likelihood': per_element_likelihood.reshape(per_element_likelihood.shape[0], *xx.shape),
            'avg_likelihood': avg_per_element_likelihood.reshape(xx.shape),
        }
    return None


def compute_mdn_test_nll(
    model: MDNTrainer,
    test_inputs: jax.Array,
    test_outputs: jax.Array,
    plot: bool = True,
    num_cols: int = 3,
    plot_cutoff: Optional[int] = None,
    return_values: bool = False,
) -> Optional[jax.Array]:
    """
    Compute NLL on test data and optionally plot samples from each ensemble member.
    
    Args:
        model: Trained MDNTrainer instance.
        test_inputs: Test input data.
        test_outputs: Test output data.
        plot: Whether to plot predictive samples (default True).
        num_cols: Number of columns in the plot grid (default 3).
        plot_cutoff: Number of samples to plot. If None, plots all.
        return_values: Whether to return the per-element NLL values (default False).
        
    Returns:
        If return_values=True, returns per-element NLL array of shape (ensemble_size, num_test_samples, 1).
        Otherwise returns None.
    """
    if plot_cutoff is None:
        plot_cutoff = test_inputs.shape[0]
    
    pred = model.apply(model.params, test_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    
    # Sample from the mixture
    samples = model.sample_from_mixture(
        random.PRNGKey(23),
        mixture_logit_weights[:, :plot_cutoff],
        mixture_means[:, :plot_cutoff],
        mixture_variances[:, :plot_cutoff]
    )
    
    # Compute per-element NLL
    per_element_nll = mdn_loss_func(mixture_logit_weights, mixture_means, mixture_variances, test_outputs)
    ensemble_size = mixture_logit_weights.shape[0]
    
    if plot:
        num_rows = (ensemble_size + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        axes = axes.flatten()
        
        for net_id in range(ensemble_size):
            axes[net_id].scatter(test_inputs[:plot_cutoff].squeeze(), samples[net_id].squeeze(), s=1, label='Predicted')
            axes[net_id].scatter(test_inputs[:plot_cutoff].squeeze(), test_outputs[:plot_cutoff].squeeze(), s=1, alpha=0.5, label='True')
            axes[net_id].set_title(
                f'Network {net_id}\nMean NLL: {per_element_nll[net_id].mean():.3f}\nMedian NLL: {jnp.median(per_element_nll[net_id]):.3f}'
            )
            axes[net_id].legend()
        
        for j in range(ensemble_size, len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle(f"Predictive Samples from Ensemble of {ensemble_size} networks\n", fontsize=32)
        plt.tight_layout()
        plt.show()
    
    if return_values:
        return per_element_nll
    return None


#########################################################
############### CFM Training and Evaluation #############
#########################################################

def train_cfm_model(
    n_train: int,
    key: jax.Array,
    n_val: int = 10_000,
    config: Optional[Dict[str, Any]] = None,
    n_iter: int = 30_000,
    hidden_features: int = 128,
    depth: int = 5,
    batch_size: int = 256,
    sigma: float = 0.0,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    plot_training_data: bool = False,
    plot_logs: bool = False,
) -> Tuple[ConditionalFlowMatchingTrainer, Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Train a CFM model on the sine inverse problem with the specified training dataset size.
    
    Args:
        n_train: Number of training samples.
        key: JAX PRNGKey for reproducibility.
        n_val: Number of validation samples (default 10_000).
        config: Optional dictionary with 'arch', 'optimizer', and 'ensemble_size' keys. 
                If None, uses default config.
        n_iter: Number of training iterations (default 30_000).
        hidden_features: Number of hidden features in the MLP.
        depth: Depth of the MLP.
        batch_size: Batch size for training (default 256).
        sigma: Standard deviation for conditional flow matching (default 0.0 for OT-CFM).
        normalize_inputs: Whether to normalize inputs (default True).
        normalize_outputs: Whether to normalize outputs (default False).
        plot_training_data: Whether to plot the training data (default False).
        plot_logs: Whether to plot training logs after training (default False).
        
    Returns:
        Tuple of:
          - model: Trained ConditionalFlowMatchingTrainer instance.
          - train_data: Tuple of (train_inputs, train_outputs, train_weights).
          - test_data: Tuple of (test_inputs, test_outputs, test_weights).
          - key: Updated JAX PRNGKey.
    """
    if config is None:
        config = get_default_cfm_config(hidden_features=hidden_features, depth=depth)
    
    arch = config['arch']
    optimizer = config['optimizer']
    ensemble_size = config['ensemble_size']
    
    # Generate and split data
    key, subkey = random.split(key)
    outputs, inputs = generate_data(n_train + n_val, key=subkey)
    key, subkey = random.split(key)
    train_data, test_data = split_data(subkey, inputs, outputs, n_train, n_val)
    train_inputs, train_outputs, train_weights = train_data
    test_inputs, test_outputs, test_weights = test_data
    
    if plot_training_data:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(train_inputs.squeeze(), train_outputs.squeeze(), s=1)
        ax.set_title(f"Training data: {n_train:,} samples")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        plt.show()
    
    # Initialize model
    key, subkey = random.split(key)
    data_loader = FlowMatchingDataset(train_data, subkey, batch_size=None, sigma=sigma)
    init_batch = next(iter(data_loader))
    key, subkey = random.split(key)
    model = ConditionalFlowMatchingTrainer(
        arch, init_batch,
        key=subkey,
        optimizer=optimizer,
        normalize_inputs=normalize_inputs,
        normalize_outputs=normalize_outputs,
        ensemble_size=ensemble_size
    )
    
    # Train model
    key, subkey = random.split(key)
    train_loader = FlowMatchingDataset(train_data, subkey, batch_size=batch_size, sigma=sigma)
    model.train(train_loader, nIter=n_iter)
    
    if plot_logs:
        model.plot_logs()
    
    return model, train_data, test_data, key


def compute_cfm_nll_likelihood(
    model: ConditionalFlowMatchingTrainer,
    key: jax.Array,
    test_inputs: jax.Array,
    test_outputs: jax.Array,
    ensemble_size: int,
    grid_size: int = 200,
    num_steps: int = 10_000,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    plot: bool = True,
    plot_training_data: bool = False,
    train_inputs: Optional[jax.Array] = None,
    train_outputs: Optional[jax.Array] = None,
    return_values: bool = False,
    progress_bar: bool = True,
    nll_vmin: Optional[float] = None,
    nll_vmax: Optional[float] = None,
    likelihood_vmin: Optional[float] = None,
    likelihood_vmax: Optional[float] = None,
) -> Optional[Dict[str, jax.Array]]:
    """
    Compute NLL and likelihood surfaces from a trained CFM model and optionally plot them.
    
    Args:
        model: Trained ConditionalFlowMatchingTrainer instance.
        key: JAX PRNGKey for sample generation.
        test_inputs: Test inputs for defining the grid range (if x_range/y_range not specified).
        test_outputs: Test outputs for defining the grid range (if x_range/y_range not specified).
        ensemble_size: Number of ensemble members.
        grid_size: Resolution of the grid (default 200).
        num_steps: Number of integration steps for NLL computation (default 10_000).
        x_range: Optional tuple (x_min, x_max) for the input range.
        y_range: Optional tuple (y_min, y_max) for the output range.
        plot: Whether to plot the NLL and likelihood surfaces (default True).
        plot_training_data: Whether to overlay training data on plots (default False).
        train_inputs: Training inputs (required if plot_training_data=True).
        train_outputs: Training outputs (required if plot_training_data=True).
        return_values: Whether to return the computed values (default False).
        progress_bar: Whether to show progress bar during NLL computation (default True).
        nll_vmin: Optional minimum value for NLL colorbar scale (default None, auto-scale).
        nll_vmax: Optional maximum value for NLL colorbar scale (default None, auto-scale).
        likelihood_vmin: Optional minimum value for likelihood colorbar scale (default None, auto-scale).
        likelihood_vmax: Optional maximum value for likelihood colorbar scale (default None, auto-scale).
        
    Returns:
        If return_values=True, returns a dictionary with:
          - 'xx': Grid x coordinates.
          - 'yy': Grid y coordinates.
          - 'per_element_nll': Per-ensemble-member NLL values.
          - 'avg_nll': Average NLL across ensemble.
          - 'ensemble_nll': Ensemble NLL (from averaged likelihoods).
          - 'per_element_likelihood': Per-ensemble-member likelihood values.
          - 'avg_likelihood': Average likelihood across ensemble.
        Otherwise returns None.
    """
    # Define grid
    if x_range is None:
        x_range = (float(test_inputs.min()), float(test_inputs.max()))
    if y_range is None:
        y_range = (float(test_outputs.min()), float(test_outputs.max()))
    
    x_grid = jnp.linspace(x_range[0], x_range[1], grid_size)
    y_grid = jnp.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    xy_inputs = xx.reshape(-1, 1)
    xy_outputs = yy.reshape(-1, 1)
    
    # Tile for ensemble
    tiled_inputs = jnp.tile(xy_inputs, (ensemble_size, 1, 1))
    tiled_outputs = jnp.tile(xy_outputs, (ensemble_size, 1, 1))
    
    # Compute NLL
    per_element_nll = model.compute_nll(
        model.params,
        tiled_inputs,
        tiled_outputs,
        num_steps=num_steps,
        apply_kwargs=FrozenDict({'tile_inputs': False}),
        progress_bar=progress_bar,
    )
    
    # Compute derived quantities
    avg_per_element_nll = per_element_nll.mean(axis=0)
    per_element_likelihood = jnp.exp(-per_element_nll)
    avg_per_element_likelihood = per_element_likelihood.mean(axis=0)
    
    # Ensemble NLL: average likelihoods, then take -log
    ensemble_likelihood = per_element_likelihood.mean(axis=0)
    ensemble_nll = -jnp.log(ensemble_likelihood)
    
    if plot:
        # Plot average NLL surface
        nll_grid = avg_per_element_nll.reshape(xx.shape)
        plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(xx, yy, nll_grid, shading='auto', cmap='jet', vmin=nll_vmin, vmax=nll_vmax)
        plt.colorbar(pcm, label="NLL")
        if plot_training_data and train_inputs is not None and train_outputs is not None:
            plt.scatter(train_inputs, train_outputs, s=1, c='white', alpha=0.6, label="Training data")
            plt.legend()
        plt.xlabel("Forward Output (y)")
        plt.ylabel("Forward Input (x)")
        plt.title("Average Conditional NLL(x|y) surface from trained CFM")
        plt.show()
        
        # Plot average likelihood surface
        likelihood_grid = avg_per_element_likelihood.reshape(xx.shape)
        plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(xx, yy, likelihood_grid, shading='auto', cmap='jet', vmin=likelihood_vmin, vmax=likelihood_vmax)
        plt.colorbar(pcm, label="Likelihood")
        if plot_training_data and train_inputs is not None and train_outputs is not None:
            plt.scatter(train_inputs, train_outputs, s=1, c='white', alpha=0.6, label="Training data")
            plt.legend()
        plt.xlabel("Forward Output (y)")
        plt.ylabel("Forward Input (x)")
        plt.title("Average P(x|y) surface from trained CFM")
        plt.show()
        
        # Plot ensemble NLL surface
        ensemble_nll_grid = ensemble_nll.reshape(xx.shape)
        plt.figure(figsize=(12, 5))
        pcm = plt.pcolormesh(xx, yy, ensemble_nll_grid, shading='auto', cmap='jet', vmin=nll_vmin, vmax=nll_vmax)
        plt.colorbar(pcm, label="NLL")
        if plot_training_data and train_inputs is not None and train_outputs is not None:
            plt.scatter(train_inputs, train_outputs, s=1, c='white', alpha=0.6, label="Training data")
            plt.legend()
        plt.xlabel("Forward Output (y)")
        plt.ylabel("Forward Input (x)")
        plt.title("Ensemble Conditional NLL(x|y) surface from trained CFM (averaged likelihoods)")
        plt.show()
    
    if return_values:
        return {
            'xx': xx,
            'yy': yy,
            'per_element_nll': per_element_nll.reshape(per_element_nll.shape[0], *xx.shape),
            'avg_nll': avg_per_element_nll.reshape(xx.shape),
            'ensemble_nll': ensemble_nll.reshape(xx.shape),
            'per_element_likelihood': per_element_likelihood.reshape(per_element_likelihood.shape[0], *xx.shape),
            'avg_likelihood': avg_per_element_likelihood.reshape(xx.shape),
        }
    return None


def compute_cfm_test_nll(
    model: ConditionalFlowMatchingTrainer,
    key: jax.Array,
    test_inputs: jax.Array,
    test_outputs: jax.Array,
    ensemble_size: int,
    num_steps: int = 10_000,
    plot: bool = True,
    num_cols: int = 3,
    return_values: bool = False,
    progress_bar: bool = True,
) -> Optional[jax.Array]:
    """
    Compute NLL on test data and optionally plot samples from each ensemble member.
    
    Args:
        model: Trained ConditionalFlowMatchingTrainer instance.
        key: JAX PRNGKey for sample generation.
        test_inputs: Test input data.
        test_outputs: Test output data.
        ensemble_size: Number of ensemble members.
        num_steps: Number of integration steps (default 10_000).
        plot: Whether to plot predictive samples (default True).
        num_cols: Number of columns in the plot grid (default 3).
        return_values: Whether to return the per-element NLL values (default False).
        progress_bar: Whether to show progress bar during computations (default True).
        
    Returns:
        If return_values=True, returns per-element NLL array of shape (ensemble_size, num_test_samples, 1).
        Otherwise returns None.
    """
    # Tile inputs/outputs for ensemble
    tiled_inputs = jnp.tile(test_inputs, (ensemble_size, 1, 1))
    tiled_outputs = jnp.tile(test_outputs, (ensemble_size, 1, 1))
    
    # Generate samples
    samples = model.generate_sample(
        key,
        model.params,
        tiled_inputs,
        num_output_dims=1,
        num_steps=num_steps,
        apply_kwargs=FrozenDict({'tile_inputs': False}),
        progress_bar=progress_bar,
    )
    
    # Compute NLL
    per_element_nll = model.compute_nll(
        model.params,
        tiled_inputs,
        tiled_outputs,
        num_steps=num_steps,
        apply_kwargs=FrozenDict({'tile_inputs': False}),
        progress_bar=progress_bar,
    )
    
    if plot:
        num_rows = (ensemble_size + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        axes = axes.flatten()
        
        for net_id in range(ensemble_size):
            axes[net_id].scatter(test_inputs.squeeze(), samples[net_id].squeeze(), s=1, label='Predicted')
            axes[net_id].scatter(test_inputs.squeeze(), test_outputs.squeeze(), s=1, alpha=0.5, label='True')
            axes[net_id].set_xlabel("Forward Output (y)")
            axes[net_id].set_ylabel("Forward Input (x)")
            axes[net_id].set_title(
                f"Network {net_id}\nMean NLL: {per_element_nll[net_id].mean():.3f}\nMedian NLL: {jnp.median(per_element_nll[net_id]):.3f}"
            )
            axes[net_id].legend()
        
        for j in range(ensemble_size, len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle(f"Flow Matching Samples from Ensemble of {ensemble_size} networks\n", fontsize=32)
        plt.tight_layout()
        plt.show()
    
    if return_values:
        return per_element_nll
    return None


#########################################################
############### Data Efficiency Comparison ##############
#########################################################
def compare_data_efficiency(
    dataset_sizes: list,
    key: jax.Array,
    n_val: int = 10_000,
    mdn_config: Optional[Dict[str, Any]] = None,
    cfm_config: Optional[Dict[str, Any]] = None,
    mdn_n_iter: int = 30_000,
    cfm_n_iter: int = 30_000,
    grid_size: int = 100,
    cfm_nll_num_steps: int = 1_000,
    x_range: Optional[Tuple[float, float]] = (-1.5, 1.5),
    y_range: Optional[Tuple[float, float]] = (-1.5, 1.5),
    plot_training_data: bool = True,
    nll_vmin: Optional[float] = None,
    nll_vmax: Optional[float] = 50,
    figsize_per_subplot: Tuple[float, float] = (6, 5),
    progress_bar: bool = True,
    nll_max_clip: Optional[float] = 100,
    plot_nll_surface_dataset_sizes: Optional[list] = None,  # New argument for custom 2D NLL plotting
    slice_mode: str = 'nll',
    plot_slice: bool = True,
    return_nll_values: bool = False,
    title_fontsize: int = 16,
    axis_label_fontsize: int = 14,
) -> Tuple[plt.Figure, jax.Array, Optional[Dict[str, jax.Array]]]:
    """
    Compare data efficiency of MDN and CFM models across different training dataset sizes.

    For each dataset size, this function:
      1. Trains an MDN model
      2. Trains a CFM model
      3. Computes the average NLL surfaces for both models (for certain dataset sizes as specified)
      4. Creates a comparison figure with 3 columns (MDN surface, NLL slice at y=0.5, CFM surface) 
         and len(plot_nll_surface_dataset_sizes) rows

    Args:
        dataset_sizes: List of training dataset sizes to compare (all included in summary NLL-vs-size curve).
        key: JAX PRNGKey for reproducibility.
        n_val: Number of validation samples (default 10_000).
        mdn_config: Optional MDN configuration. If None, uses default config.
        cfm_config: Optional CFM configuration. If None, uses default config.
        mdn_n_iter: Number of training iterations for MDN (default 20_000).
        cfm_n_iter: Number of training iterations for CFM (default 30_000).
        grid_size: Resolution of the NLL surface grid (default 200).
        cfm_num_steps: Number of integration steps for CFM NLL computation (default 10_000).
        x_range: Optional tuple (x_min, x_max) for the input range of plots.
        y_range: Optional tuple (y_min, y_max) for the output range of plots.
        plot_training_data: Whether to overlay training data on plots (default True).
        nll_vmin: Optional minimum value for NLL colorbar scale (default None, auto-scale).
        nll_vmax: Optional maximum value for NLL colorbar scale (default None, auto-scale).
        figsize_per_subplot: Size of each subplot (width, height) in inches (default (6, 5)).
        progress_bar: Whether to show progress bar during CFM computations (default True).
        plot_nll_surface_dataset_sizes: Optional list specifying which sizes in `dataset_sizes` to plot with 2D NLL surfaces
            (all others are included in the summary curve). If None, plot all.
        nll_max_clip: Optional maximum value for NLL colorbar scale (default None, auto-scale).
        slice_mode: Mode for the slice plot (default 'nll', can be 'nll' or 'likelihood').
            'nll' plots the NLL(x|y) slice at y=0.5, 'likelihood' plots the likelihood P(x|y) slice at y=0.5.
        plot_slice: Whether to plot the middle column of NLL/likelihood slices (default True).
        title_fontsize: Font size for subplot titles (default 14).
        axis_label_fontsize: Font size for axis labels (default 12).
    Returns:
        Tuple of:
          - fig: Matplotlib figure containing the comparison plots.
          - key: Updated JAX PRNGKey.
    """
    if plot_nll_surface_dataset_sizes is None:
        plot_nll_surface_dataset_sizes = dataset_sizes
    else:
        if not set(plot_nll_surface_dataset_sizes).issubset(set(dataset_sizes)):
            raise ValueError("All values in plot_nll_surface_dataset_sizes must be present in dataset_sizes.")

    num_surface_plots = len(plot_nll_surface_dataset_sizes)
    num_sizes = len(dataset_sizes)

    # We'll need to store the means and stds for each dataset size for both models
    mdn_nll_means = []
    mdn_nll_stds = []
    cfm_nll_means = []
    cfm_nll_stds = []
    used_dataset_sizes = []  # Incase some dataset sizes are skipped

    # For plotting 2D NLL (surface) results only for selected sizes, keep mapping
    surface_plot_indices = {n: i for i, n in enumerate(plot_nll_surface_dataset_sizes)}

    # Create figure with an extra row on top for the summary/comparison curve
    # Number of columns depends on plot_slice: 3 if True (MDN, slice, CFM), 2 if False (MDN, CFM)
    num_cols = 3 if plot_slice else 2
    fig, axes = plt.subplots(
        num_surface_plots + 1, num_cols,
        figsize=(figsize_per_subplot[0] * num_cols, figsize_per_subplot[1] * (num_surface_plots + 1))
    )

    # Handle case when there's only one surface plot row
    if num_surface_plots == 1:
        axes = axes.reshape(2, -1)  # 2 rows: curve + single data row

    # Get configs
    if mdn_config is None:
        mdn_config = get_default_mdn_config()
    if cfm_config is None:
        cfm_config = get_default_cfm_config()

    cfm_ensemble_size = cfm_config['ensemble_size']

    # NOTE:
    # Row index for 2D NLL axes is (surface_plot_indices[n_train] + 1) when n_train ∈ plot_nll_surface_dataset_sizes

    # Store trained models and data to avoid recomputation for each size
    _cached_models_data = {}

    # Loop through dataset sizes (always for summary curve; surfaces only for subset)
    for row_idx, n_train in enumerate(dataset_sizes):
        print(f"\n{'='*60}")
        print(f"Training with {n_train:,} samples (row {row_idx + 1}/{num_sizes})")
        print(f"{'='*60}\n")

        # Train MDN model
        print(f"Training MDN model with {n_train:,} samples...")
        key, subkey = random.split(key)
        mdn_model, mdn_train_data, mdn_test_data, key = train_mdn_model(
            n_train=n_train,
            key=subkey,
            n_val=n_val,
            config=mdn_config,
            n_iter=mdn_n_iter,
            plot_training_data=False,
            plot_logs=False,
        )
        mdn_train_inputs, mdn_train_outputs, _ = mdn_train_data
        mdn_test_inputs, mdn_test_outputs, _ = mdn_test_data

        # Train CFM model
        print(f"Training CFM model with {n_train:,} samples...")
        cfm_model, cfm_train_data, cfm_test_data, key = train_cfm_model(
            n_train=n_train,
            key=subkey,
            n_val=n_val,
            config=cfm_config,
            n_iter=cfm_n_iter,
            plot_training_data=False,
            plot_logs=False,
        )
        cfm_train_inputs, cfm_train_outputs, _ = cfm_train_data
        cfm_test_inputs, cfm_test_outputs, _ = cfm_test_data

        # Optionally plot 2D NLL surfaces for subset of dataset sizes
        if n_train in surface_plot_indices:
            surface_row_idx = surface_plot_indices[n_train]
            # Compute MDN NLL surface
            print(f"Computing MDN NLL surface...")
            mdn_results = compute_mdn_nll_likelihood(
                model=mdn_model,
                test_inputs=mdn_test_inputs,
                test_outputs=mdn_test_outputs,
                grid_size=grid_size,
                x_range=x_range,
                y_range=y_range,
                plot=False,
                return_values=True,
            )

            # Compute CFM NLL surface
            print(f"Computing CFM NLL surface...")
            key, subkey = random.split(key)
            cfm_results = compute_cfm_nll_likelihood(
                model=cfm_model,
                key=subkey,
                test_inputs=cfm_test_inputs,
                test_outputs=cfm_test_outputs,
                ensemble_size=cfm_ensemble_size,
                grid_size=grid_size,
                num_steps=cfm_nll_num_steps,
                x_range=x_range,
                y_range=y_range,
                plot=False,
                return_values=True,
                progress_bar=progress_bar,
            )
        else:
            mdn_results = None
            cfm_results = None

        # ---- NLL computation for summary curve (use test data instead of grid!) ----
        # Compute MDN mean/std test NLL over ensemble members
        mdn_test_nlls = compute_mdn_test_nll(
            mdn_model,
            test_inputs=mdn_test_inputs,
            test_outputs=mdn_test_outputs,
            return_values=True,
            plot=False,
        )  # shape: (ensemble, n_test, 1)
        mdn_test_nlls = mdn_test_nlls.mean(axis=(-1, -2)) # shape: (ensemble,)

        cfm_test_nlls = compute_cfm_test_nll(
            cfm_model,
            key=subkey,
            return_values=True,
            plot=False,
            ensemble_size=cfm_ensemble_size,
            num_steps=cfm_nll_num_steps,
            test_inputs=cfm_test_inputs,
            test_outputs=cfm_test_outputs,
        )  # shape: (ensemble, n_test, 1)
        cfm_test_nlls = cfm_test_nlls.mean(axis=(-1, -2)) # shape: (ensemble,)

        mdn_nll_means.append(mdn_test_nlls.mean())  # append scalar value
        mdn_nll_stds.append(mdn_test_nlls.std())    # append scalar value

        cfm_nll_means.append(cfm_test_nlls.mean())  # append scalar value
        cfm_nll_stds.append(cfm_test_nlls.std())    # append scalar value

        used_dataset_sizes.append(n_train)

        if n_train in surface_plot_indices:
            # Plot MDN NLL surface (left column, appropriate row)
            ax_mdn = axes[surface_row_idx + 1, 0]
            xx_mdn = mdn_results['xx']
            yy_mdn = mdn_results['yy']
            nll_grid_mdn = mdn_results['avg_nll']

            if nll_max_clip is not None:
                nll_grid_mdn = jnp.clip(nll_grid_mdn, max=nll_max_clip)

            pcm_mdn = ax_mdn.pcolormesh(xx_mdn, yy_mdn, nll_grid_mdn, shading='auto', cmap='jet', vmin=nll_vmin, vmax=nll_vmax)
            plt.colorbar(pcm_mdn, ax=ax_mdn, label="NLL")
            if plot_training_data:
                ax_mdn.scatter(mdn_train_inputs, mdn_train_outputs, s=5, c='white', alpha=0.8, edgecolors='black', label='Training data')
            if plot_slice: # Only plot the y=0.5 slice if plot_slice is True
               ax_mdn.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='y=0.5 slice')
            ax_mdn.set_xlabel("Forward Output (y)", fontsize=axis_label_fontsize)
            ax_mdn.set_ylabel("Forward Input (x)", fontsize=axis_label_fontsize)
            ax_mdn.set_title(f"MDN - {n_train:,} samples", fontsize=title_fontsize)
            ax_mdn.set_xlim(x_range[0], x_range[1])
            ax_mdn.set_ylim(y_range[0], y_range[1])
            ax_mdn.legend()

            # Plot NLL slice at y=0.5 (middle column, appropriate row) - only if plot_slice is True
            if plot_slice:
                ax_slice = axes[surface_row_idx + 1, 1]
                
                # Extract x-coordinates from the grid (first row of xx, or first column of x_grid)
                xs = xx_mdn[0, :]  # shape: (grid_size,)
                ys = yy_mdn[:, 0]  # shape: (grid_size,)
                
                # Get the ground truth conditional NLL at y=0.5
                gt_conditional_nll = get_conditional_x_nll(xs, ys, mode='nll')  # shape: (grid_size, grid_size)
                gt_conditional_likelihood = get_conditional_x_nll(xs, ys, mode='likelihood')  # shape: (grid_size, grid_size)
                
                # Extract slice at index -66 (corresponding to y=0.5)
                slice_idx = 66
                gt_nll_slice = gt_conditional_nll[slice_idx, :]  # shape: (grid_size,)
                gt_likelihood_slice = gt_conditional_likelihood[slice_idx, :]  # shape: (grid_size,)
                mdn_nll_slice = nll_grid_mdn[:, slice_idx]  # shape: (grid_size,)
                mdn_likelihood_slice = mdn_results['avg_likelihood'][:, slice_idx]  # shape: (grid_size,)
                cfm_nll_slice = cfm_results['avg_nll'][:, slice_idx]  # shape: (grid_size,)
                cfm_likelihood_slice = cfm_results['avg_likelihood'][:, slice_idx]  # shape: (grid_size,)
                
                # Plot the slices
                if slice_mode == 'nll':
                    ax_slice.plot(xs, gt_nll_slice, label='Ground Truth', linewidth=2, color='black')
                    ax_slice.plot(xs, mdn_nll_slice, label='MDN', linewidth=2, color='tab:blue')
                    ax_slice.plot(xs, cfm_nll_slice, label='CFM', linewidth=2, color='tab:orange')
                    ax_slice.set_ylabel(f'NLL(x|y={ys[slice_idx]:.2f})', fontsize=axis_label_fontsize)
                    ax_slice.set_title(f'NLL Slice at y={ys[slice_idx]:.2f} ({n_train:,} samples)', fontsize=title_fontsize)
                elif slice_mode == 'likelihood':
                    ax_slice.plot(xs, gt_likelihood_slice, label='Ground Truth', linewidth=2, color='black')
                    ax_slice.plot(xs, mdn_likelihood_slice, label='MDN', linewidth=2, color='tab:blue')
                    ax_slice.plot(xs, cfm_likelihood_slice, label='CFM', linewidth=2, color='tab:orange')
                    ax_slice.set_ylabel(f'Likelihood P(x|y={ys[slice_idx]:.2f})', fontsize=axis_label_fontsize)
                    ax_slice.set_title(f'Likelihood Slice at y={ys[slice_idx]:.2f} ({n_train:,} samples)', fontsize=title_fontsize)
                ax_slice.legend()
                ax_slice.set_xlabel('x (Forward Input)', fontsize=axis_label_fontsize)
                ax_slice.set_xlim(x_range[0], x_range[1])
                ax_slice.grid(True, alpha=0.3)
            
            # Plot CFM NLL surface (right column if plot_slice, otherwise second column)
            cfm_col_idx = 2 if plot_slice else 1
            ax_cfm = axes[surface_row_idx + 1, cfm_col_idx]
            xx_cfm = cfm_results['xx']
            yy_cfm = cfm_results['yy']
            nll_grid_cfm = cfm_results['avg_nll']

            if nll_max_clip is not None:
                nll_grid_cfm = jnp.clip(nll_grid_cfm, max=nll_max_clip)

            pcm_cfm = ax_cfm.pcolormesh(xx_cfm, yy_cfm, nll_grid_cfm, shading='auto', cmap='jet', vmin=nll_vmin, vmax=nll_vmax)
            plt.colorbar(pcm_cfm, ax=ax_cfm, label="NLL")
            if plot_training_data:
                ax_cfm.scatter(cfm_train_inputs, cfm_train_outputs, s=5, c='white', alpha=0.6, edgecolors='black', label='Training data')
            if plot_slice:
                ax_cfm.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='y=0.5 slice')
            ax_cfm.set_xlabel("Forward Output (y)", fontsize=axis_label_fontsize)
            ax_cfm.set_ylabel("Forward Input (x)", fontsize=axis_label_fontsize)
            ax_cfm.set_xlim(x_range[0], x_range[1])
            ax_cfm.set_ylim(y_range[0], y_range[1])
            ax_cfm.set_title(f"CFM - {n_train:,} samples", fontsize=title_fontsize)
            ax_cfm.legend()

    # After all dataset sizes are processed, plot summary NLL-vs-size on top row (spans all columns)
    # Uncertainty is shown as mean +/- 2*std
    ax_curve = fig.add_subplot(num_surface_plots + 1, 1, 1)  # New axes over top row, spanning
    xs = used_dataset_sizes
    xs = [float(i) for i in xs]
    # Plot MDN curve with error band
    ax_curve.plot(xs, mdn_nll_means, 'o-', label="MDN", color='tab:blue')
    ax_curve.fill_between(xs,
                         [m-s for m, s in zip(mdn_nll_means, mdn_nll_stds)],
                         [m+s for m, s in zip(mdn_nll_means, mdn_nll_stds)],
                         alpha=0.2, color='tab:blue')
    # Plot CFM curve with error band
    ax_curve.plot(xs, cfm_nll_means, 's-', label="CFM", color='tab:orange')
    ax_curve.fill_between(xs,
                         [m-s for m, s in zip(cfm_nll_means, cfm_nll_stds)],
                         [m+s for m, s in zip(cfm_nll_means, cfm_nll_stds)],
                         alpha=0.2, color='tab:orange')
    ax_curve.set_ylabel("Mean NLL", fontsize=axis_label_fontsize)
    ax_curve.set_xlabel("Training set size", fontsize=axis_label_fontsize)
    ax_curve.set_title("Mean Test NLL vs Training Size (shaded: ±1 std dev, ensemble)", fontsize=title_fontsize)
    ax_curve.legend()
    ax_curve.set_xscale('log')
    ax_curve.grid(True)

    # Hide all axes of top row in the "main" axes array to avoid duplicate/unneeded
    for col_idx in range(num_cols):
        axes[0, col_idx].set_visible(False)

    #fig.suptitle("Data Efficiency Comparison: Average NLL(x|y) Surfaces", fontsize=16, y=1.0)
    plt.tight_layout()

    if return_nll_values:
        nll_values = {
            'mdn_nll_means': mdn_nll_means,
            'mdn_nll_stds': mdn_nll_stds,
            'cfm_nll_means': cfm_nll_means,
            'cfm_nll_stds': cfm_nll_stds,
        }
        return fig, key, nll_values
    else:
        return fig, key
