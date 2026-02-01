# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad

from flax import linen as nn
from flax.core import FrozenDict
import optax
import diffrax

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as onp
import copy
import pickle
import pandas as pd
from tqdm import trange
import math
from typing import Tuple, Optional, Any, Callable
from functools import partial


def lorenz_velocity_vector_field(t, y, args):
    """
    Lorenz velocity vector field.
    Args:
        t: Time.
        y: State.
        args: Arguments.

    Returns:
        Velocity vector field.
    """
    sigma = args[0]
    rho = args[1]
    beta = args[2]
    y_1, y_2, y_3 = jnp.split(y, 3, axis=-1)
    return jnp.concatenate([sigma * (y_2 - y_1), y_1 * (rho - y_3) - y_2, y_1 * y_2 - beta * y_3], axis=-1)


def evolve_lorenz_system(
    initial_positions: jnp.ndarray,
    final_time: float,
    time_steps: int,
    dt: Optional[float] = None,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> jnp.ndarray:
    """
    Evolves the Lorenz system for a given initial position using the diffrax ODE solver.

    Args:
        initial_positions: Array of shape (n_trajectories, 3) containing the initial positions.
        final_time: Final time of the simulation.
        time_steps: Number of time steps to save in each trajectory.
        dt: Time step size for the ODE solver.
        sigma: Lorenz system parameter.
        rho: Lorenz system parameter.
        beta: Lorenz system parameter.
    Returns:
        Array of shape (time_steps, n_trajectories, 3) containing the trajectories.
    """
    if dt is None:
        dt = final_time / time_steps

    # compute the number of evolution iteration steps
    num_evolution_steps = int(final_time / dt) + 1

    # Set up ODE system in diffrax    
    args = (sigma, rho, beta)
    term = diffrax.ODETerm(lorenz_velocity_vector_field)
    solver = diffrax.Dopri5()

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=final_time,
        dt0=dt,
        y0=initial_positions,
        args=args,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, final_time, time_steps)),
        max_steps=num_evolution_steps,
        )

    trajectories = solution.ys # shape (time_steps, n_trajectories, 3)

    return trajectories

def generate_lorenz_data(
    n_trajectories: int,
    final_time: float = 5,
    time_steps: int = 100,
    dt: Optional[float] = None,
    key: jax.Array = random.PRNGKey(0),
    sigma: float = 10.0, # Lorenz system parameter
    rho: float = 28.0, # Lorenz system parameter
    beta: float = 8.0 / 3.0, # Lorenz system parameter
    observation_noise: float = 0.0,
    format: str = 'one_step',
    return_difference: bool = False,
    shuffle_data: bool = False,
) -> tuple:
    """
    Generates synthetic trajectory data for the Lorenz system.
    
    Simulates multiple trajectories of the Lorenz chaotic system and returns
    input-output pairs suitable for autoregressive training.
    
    Args:
        n_trajectories: Number of independent trajectories to simulate.
        final_time: Total simulation time for each trajectory (default: 5).
        time_steps: Number of time steps to save in each trajectory (default: 100).
        dt: Time step size for the ODE solver. If None, computed as final_time / time_steps.
        key: JAX random key for reproducibility (default: random.PRNGKey(0)).
        sigma: Lorenz system parameter (default: 10.0).
        rho: Lorenz system parameter (default: 28.0).
        beta: Lorenz system parameter (default: 8.0/3.0).
        observation_noise: Standard deviation of Gaussian noise added to observations (default: 0.0).
        return_difference: Whether to return the difference between the outputs and the inputs (default: False).
        shuffle_data: Whether to shuffle the data order before returning it (default: False).
    Returns:
        A tuple (inputs, outputs) where:
            - inputs: Array of shape ((time_steps-1) * n_trajectories, 3) containing states at time t.
            - outputs: Array of shape ((time_steps-1) * n_trajectories, 3) containing states at time t+1.
    """

    # Generate initial positions
    key, subkey = random.split(key)
    initial_positions = random.normal(subkey, shape=(n_trajectories, 3)) + jnp.array([0, 0, 24.5])
    # Solve ODE using diffrax

    trajectories = evolve_lorenz_system(initial_positions, final_time, time_steps, dt, sigma, rho, beta)

    # Add observation noise if requested
    if observation_noise > 0:
        key, subkey = random.split(key)
        observation_noise = random.normal(subkey, shape=(time_steps, n_trajectories, 3)) * observation_noise
        trajectories = trajectories + observation_noise

    if format == 'one_step':
        # Re-organize trajectories for autoregressive training
        inputs = trajectories[:-1] # shape (time_steps-1, n_trajectories, 3)
        outputs = trajectories[1:] # shape (time_steps-1, n_trajectories, 3)
        if return_difference:
            outputs = outputs - inputs

        # Flatten inputs and outputs
        inputs = inputs.reshape(-1, 3)
        outputs = outputs.reshape(-1, 3)

        if shuffle_data:
            key, subkey = random.split(key)
            permutation = random.permutation(subkey, jnp.arange(inputs.shape[0]))
            inputs = inputs[permutation]
            outputs = outputs[permutation]

        return inputs, outputs

    elif format == 'sequence':
        if shuffle_data:
            key, subkey = random.split(key)
            permutation = random.permutation(subkey, jnp.arange(trajectories.shape[1]))
            trajectories = trajectories[:, permutation, :]

        # trajectories is currently in shape (time_steps, n_trajectories, 3)
        # we want to return a sequence of shape (n_trajectories, time_steps, 3)
        return jnp.moveaxis(trajectories, (0, 1), (1, 0))

    else:
        raise ValueError(f"Invalid format: {format}")



def plot_mse_fields(
    model: Any,
    true_inputs: jnp.ndarray,
    true_outputs: jnp.ndarray,
    cutoff: Optional[int] = None,
    arrow_length_factor: float = 0.5,
    num_cols: int = 3,
) -> None:
    """
    Visualize MSE fields from an ensemble of MSE models.

    For each ensemble member, the function generates plots of example trajectories for both predicted and true outputs.

    Args:
        model: The trained ensemble of MSE models with `.apply()` and `.params`.
        true_inputs: Array of initial conditions, shape (batch, n_dim).
        true_outputs: Array of true output trajectories, shape (batch, n_dim).
        cutoff: Number of trajectories to display per mixture for prediction and target comparisons.
        arrow_length_factor: Factor to scale the length of the arrows in the flow field plots.
        num_cols: Number of columns in the figure.
    Returns:
        None. Displays matplotlib figures for MSE fields.
    """
    if cutoff is not None:
        true_inputs = true_inputs[:cutoff] # shape (cutoff, n_dim)
        true_outputs = true_outputs[:cutoff] # shape (cutoff, n_dim)

    pred = model.apply(model.params, true_inputs) # shape (ensemble_size, batch_size, n_dim)
    ensemble_size = pred.shape[0]

    fig = plt.figure(figsize=(20, 10 * num_cols))
    nrows = math.ceil(ensemble_size / num_cols)
    ncols = min(ensemble_size, num_cols)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for net_id in range(ensemble_size):
        
        # First column: plot arrows of the predicted flow field
        ax1 = fig.add_subplot(gs[net_id // num_cols, net_id % num_cols], projection='3d')
        ax1.quiver(true_inputs[:, 0], true_inputs[:, 1], true_inputs[:, 2], 
                pred[net_id, :, 0], pred[net_id, :, 1], pred[net_id, :, 2], 
                length=arrow_length_factor, color='red', label='Predicted', alpha=0.5)
        ax1.quiver(true_inputs[:, 0], true_inputs[:, 1], true_inputs[:, 2], 
                true_outputs[:, 0], true_outputs[:, 1], true_outputs[:, 2], 
                length=arrow_length_factor, color='blue', label='True', alpha=0.5)
        ax1.set_title(f'Network {net_id} Flow Field')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()

    plt.tight_layout()
    plt.show()


def plot_mse_ensemble_rollouts(
    model: Any,
    tiled_initial_positions: jnp.ndarray,
    true_trajectories: jnp.ndarray,
    num_plot_trajectories: int,
    max_cols: int = 3,
    return_predicted_trajectories: bool = False,
    display_progress_bar: bool = True,
) -> None:
    """
    Visualize rollouts from an ensemble of MSE models.

    Args:
        model: The trained ensemble of MSE models with `.apply()` and `.params`.
        tiled_initial_positions: Array of shape (ensemble_size, n_trajectories, n_dim) containing the initial positions.
        true_trajectories: Array of shape (time_steps, n_trajectories, n_dim) containing the true output trajectories.
        num_plot_trajectories: Number of trajectories to plot.
        max_cols: Number of columns in the figure.
        return_predicted_trajectories: If True, returns the predicted trajectories instead of None.
        display_progress_bar: If True, displays a progress bar.
    Returns:
        None or predicted_trajectories (jnp.ndarray of shape [time_steps, ensemble_size, num_plot_trajectories, n_dim])
            if return_predicted_trajectories is True.
    """
    time_steps = true_trajectories.shape[0]
    ensemble_size = tiled_initial_positions.shape[0]

    nrows = math.ceil(ensemble_size / max_cols)
    ncols = min(ensemble_size, max_cols)
    fig = plt.figure(figsize=(6 * ncols, 6 * (nrows + 1)))

    # Set up a GridSpec with nrows+1 for rows, ncols for columns
    gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig)

    # Plot the true trajectories in the first row
    ax_true = fig.add_subplot(gs[0, :], projection='3d')
    for i in range(num_plot_trajectories):
        trajectory = true_trajectories[:, i, :]
        ax_true.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax_true.set_title('True Trajectories')
    ax_true.set_xlabel('x')
    ax_true.set_ylabel('y')
    ax_true.set_zlabel('z')

    # Compute the predicted trajectories
    predicted_trajectories = []
    inputs = tiled_initial_positions
    if display_progress_bar:
        pbar = trange(time_steps)
    else:
        pbar = range(time_steps)
    for i in pbar:
        pred = model.apply(model.params, inputs, kwargs=FrozenDict({'tile_inputs': False}))
        inputs = inputs + pred
        predicted_trajectories.append(inputs)
    predicted_trajectories = jnp.array(predicted_trajectories) # shape (time_steps, ensemble_size, num_plot_trajectories, n_dim)
    print(f'predicted_trajectories shape: {predicted_trajectories.shape}')

    # Plot the predicted trajectories
    for net_id in range(ensemble_size):
        row = (net_id // max_cols) + 1  # Start from row 1 (since 0 is the true traj plot)
        col = net_id % max_cols
        ax_pred = fig.add_subplot(gs[row, col], projection='3d')
        for i in range(num_plot_trajectories):
            trajectory = predicted_trajectories[:, net_id, i, :]
            ax_pred.plot(trajectory[..., 0], trajectory[..., 1], trajectory[..., 2])
        ax_pred.set_title(f'Network {net_id} Predictions')
        ax_pred.set_xlabel('x')
        ax_pred.set_ylabel('y')
        ax_pred.set_zlabel('z')

    plt.tight_layout()
    plt.show()

    if return_predicted_trajectories:
        return predicted_trajectories


def plot_mdn_mixture_elements(
    model: Any,
    true_inputs: jnp.ndarray,
    true_outputs: jnp.ndarray,
    net_id: int,
    cutoff: Optional[int] = None,
    arrow_length_factor: float = 0.5,
) -> None:
    """
    Visualize mixture component means and mixture probabilities from an MDN (Mixture Density Network).

    For each mixture component, the function generates:
      - (Left) Plots of example trajectories for both predicted mixture means and their true counterparts.
      - (Right) A scatter plot of the initial positions colored by the mixture's predicted assignment probability.

    Args:
        model: The trained MDN model with `.apply()` and `.params`.
        true_inputs: Array of initial conditions, shape (batch, n_dim).
        true_outputs: Array of true output trajectories, shape (batch, time_steps * n_dim).
        net_id: Index of the ensemble network to visualize (int).
        cutoff: Number of trajectories to display per mixture for prediction and target comparisons.
        arrow_length_factor: Factor to scale the length of the arrows in the flow field plots.

    Returns:
        None. Displays matplotlib figures for mixture means and probabilities.
    """
    if cutoff is not None:
        true_inputs = true_inputs[:cutoff]
        true_outputs = true_outputs[:cutoff]

    pred = model.apply(model.params, true_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    probabilities = jax.nn.softmax(mixture_logit_weights, axis=-2)

    num_mixtures = mixture_logit_weights.shape[-2]
    fig = plt.figure(figsize=(20, 10 * num_mixtures))

    for i in range(num_mixtures):
        marginal_prob = probabilities[net_id, :, i].mean()
        
        # First column: plot arrows of the predicted flow field
        ax1 = fig.add_subplot(num_mixtures, 2, 2*i + 1, projection='3d')
        ax1.quiver(true_inputs[:, 0], true_inputs[:, 1], true_inputs[:, 2], 
                mixture_means[net_id, :, i, 0], mixture_means[net_id, :, i, 1], mixture_means[net_id, :, i, 2], 
                length=arrow_length_factor, color='red', label='Predicted', alpha=0.5)
        ax1.quiver(true_inputs[:, 0], true_inputs[:, 1], true_inputs[:, 2], 
                true_outputs[:, 0], true_outputs[:, 1], true_outputs[:, 2], 
                length=arrow_length_factor, color='blue', label='True', alpha=0.5)
        ax1.set_title(f'Mixture {i} Flow Field (p={marginal_prob:.3f})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        
        # Second column: plot probabilities for each mixture
        ax2 = fig.add_subplot(num_mixtures, 2, 2*i + 2, projection='3d')
        sc = ax2.scatter(true_inputs[:, 0], true_inputs[:, 1], true_inputs[:, 2], 
                        s=3, c=probabilities[net_id, :, i].flatten(), cmap='jet', vmin=0, vmax=1)
        ax2.set_title(f'Mixture {i} Probabilities (p={marginal_prob:.3f})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        fig.colorbar(sc, ax=ax2, label='Probability', shrink=0.5)

    plt.tight_layout()
    plt.show()


def plot_mdn_ensemble_samples(
    model: Any,
    tiled_initial_positions: jnp.ndarray,
    true_trajectories: jnp.ndarray,
    num_plot_trajectories: int,
    max_cols: int = 3,
    key: jax.Array = random.PRNGKey(7827),
    return_predicted_trajectories: bool = False,
    restrict_rare_event_rate: Optional[float] = None,
    truncated_normal_std_limit: Optional[float] = None,
) -> None:
    """
    Plots ensemble samples generated from a Mixture Density Network (MDN) model
    for the Lorenz system, comparing them visually to true reference trajectories.

    Args:
        model: Trained MDN ensemble model. Must have `.params` for model parameters
            and a `.sample_from_mixture` method for sampling.
        tiled_initial_positions: jnp.ndarray of shape (ensemble_size, num_trajectories, n_dim);
            Initial conditions tiled for each ensemble member.
        true_trajectories: jnp.ndarray of shape (time_steps, num_trajectories, n_dim);
            True reference trajectories corresponding to the initial positions.
        num_plot_trajectories: Number of trajectories to visualize in each plot.
        max_cols: Maximum number of subplot columns (default: 3).
        key: JAX random.PRNGKey for stochastic model sampling (default: random.PRNGKey(7827)).
        return_predicted_trajectories: If True, returns the predicted trajectories instead of None.
        restrict_rare_event_rate: If not None, restricts the rare event rate to the specified value.
        truncated_normal_std_limit: If not None, limits the standard deviation of the truncated normal distribution to the specified value.
    Returns:
        None or predicted_trajectories (jnp.ndarray of shape [time_steps, ensemble_size, num_plot_trajectories, n_dim])
            if return_predicted_trajectories is True.

    Displays:
        - A matplotlib figure with:
            - The true trajectories (as a single large subplot spanning all columns at the top)
            - Each ensemble member's predicted trajectories (as subplots in the lower part of the figure)
    """
    time_steps = true_trajectories.shape[0]
    ensemble_size = tiled_initial_positions.shape[0]

    # compute the predicted trajectories
    predicted_trajectories = []
    inputs = tiled_initial_positions
    pbar = trange(time_steps)
    for i in pbar:
        pred = model.apply(model.params, inputs, kwargs=FrozenDict({'backbone_kwargs': {'tile_inputs': False}}))
        mixture_logit_weights, mixture_means, mixture_variances = pred
        key, subkey = random.split(key)
        position_delta = model.sample_from_mixture(
            subkey,
            mixture_logit_weights,
            mixture_means, mixture_variances,
            restrict_rare_event_rate=restrict_rare_event_rate,
            truncated_normal_std_limit=truncated_normal_std_limit
            ).squeeze(-2)
        next_position = inputs + position_delta
        predicted_trajectories.append(next_position)
        inputs = next_position

    predicted_trajectories = jnp.array(predicted_trajectories)

    nrows = math.ceil(ensemble_size / max_cols)
    ncols = min(ensemble_size, max_cols)
    fig = plt.figure(figsize=(6 * ncols, 6 * (nrows + 1)))

    # Set up a GridSpec with nrows+1 for rows, ncols for columns
    gs = gridspec.GridSpec(nrows + 1, ncols, figure=fig)

    # Large true trajectory plot at the top spanning all columns
    ax_true = fig.add_subplot(gs[0, :], projection='3d')
    for i in range(num_plot_trajectories):
        ax_true.plot(true_trajectories[:, i, 0], true_trajectories[:, i, 1], true_trajectories[:, i, 2], label=f"Traj {i+1}")
    ax_true.set_title("True Trajectories (no observation noise)")
    ax_true.set_xlabel('x')
    ax_true.set_ylabel('y')
    ax_true.set_zlabel('z')
    if num_plot_trajectories <= 7:
        ax_true.legend()

    # Small subplots for each ensemble's predicted trajectories
    for net_id in range(ensemble_size):
        row = (net_id // max_cols) + 1  # Start from row 1 (since 0 is the true traj plot)
        col = net_id % max_cols
        ax_pred = fig.add_subplot(gs[row, col], projection='3d')
        for i in range(num_plot_trajectories):
            trajectory = predicted_trajectories[:, net_id, i, :]
            ax_pred.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        ax_pred.set_title(f"Rollout Predictions From Network {net_id}")
        ax_pred.set_xlabel('x')
        ax_pred.set_ylabel('y')
        ax_pred.set_zlabel('z')

    plt.tight_layout()
    plt.show()

    if return_predicted_trajectories:
        return predicted_trajectories


def plot_rnn_mdn_mixture_elements(
    model: Any,
    initial_positions: jnp.ndarray,
    time_steps: int,
    dt: float,
    cutoff: Optional[int] = None,
    key: jax.Array = random.PRNGKey(375184730),
    return_samples: bool = False,
    arrow_length_factor: float = 0.5,
    restrict_rare_event_rate: Optional[float] = None,
    truncated_normal_std_limit: Optional[float] = None,
    sort_by_marginal_prob: bool = False,  # New argument
) -> None:
    """
    Plots mixture elements from a RNN-MDN model for the Lorenz system.

    Args:
        model: The RNN-MDN model.
        initial_positions: The initial positions of the trajectories.
        time_steps: The number of time steps to roll out.
        dt: The time step size.
        cutoff: The number of trajectories to plot.
        key: The random key.
        return_samples: If True, returns the samples.
        sort_by_marginal_prob: If True, plot mixture elements with highest marginal probability at the top.

    Returns:
        None or samples (jnp.ndarray of shape [num_trajectories, time_steps, 3])
            if return_samples is True.
    """
    key, subkey = random.split(key)
    samples, lw, m, v = model.rollout(
        model.params,
        initial_positions,
        subkey,
        rollout_steps=time_steps,
        difference_prediction=True,
        restrict_rare_event_rate=restrict_rare_event_rate,
        truncated_normal_std_limit=truncated_normal_std_limit,
    )
    num_mixtures = lw.shape[-3]

    # samples is shape (num_trajectories, num_steps, 3)
    # lw is shape (num_trajectories, num_mixtures, num_steps, 1)
    # m is shape (num_trajectories, num_mixtures, num_steps, 3)
    # v is shape (num_trajectories, num_mixtures, num_steps, 3)
    # reshape so that we have (num_trajectories*num_steps, num_mixtures, 3)
    mixture_logit_weights = jnp.moveaxis(lw[..., 1:, :], (0, 1, 2), (0, 2, 1)).reshape(-1, num_mixtures, 1)
    mixture_means = jnp.moveaxis(m[..., 1:, :], (0, 1, 2), (0, 2, 1)).reshape(-1, num_mixtures, 3)
    mixture_variances = jnp.moveaxis(v[..., 1:, :], (0, 1, 2), (0, 2, 1)).reshape(-1, num_mixtures, 3)
    flat_positions = samples[..., :-1, :].reshape(-1, 3)

    # Compute probabilities for each mixture
    probabilities = jax.nn.softmax(mixture_logit_weights, axis=-2) # shape (num_trajectories*num_steps, num_mixtures, 1)

    # Compute true flow field by evolving the system forward in time and taking the difference
    true_evolution = evolve_lorenz_system(flat_positions, dt, 2, 0.001, 10.0, 28.0, 8.0 / 3.0) # shape (2, num_trajectories*num_steps, 3)
    true_outputs = true_evolution[1] - true_evolution[0] # shape (num_trajectories*num_steps, 3)

    if cutoff is not None:
        key, subkey = random.split(key)
        # pick random indices to keep
        indices = random.choice(subkey, jnp.arange(flat_positions.shape[0]), shape=(cutoff,), replace=False)
        flat_positions = flat_positions[indices]
        mixture_logit_weights = mixture_logit_weights[indices]
        mixture_means = mixture_means[indices]
        mixture_variances = mixture_variances[indices]
        probabilities = probabilities[indices]
        true_outputs = true_outputs[indices]

    # Compute marginal probability for each mixture
    marginal_probs = jnp.mean(probabilities.squeeze(-1), axis=0)  # shape (num_mixtures,)

    # Optionally sort mixtures by marginal probability (highest prob at the top)
    if sort_by_marginal_prob:
        sorted_indices = jnp.argsort(-marginal_probs)  # descending order
    else:
        sorted_indices = jnp.arange(num_mixtures)

    # Plotting
    fig = plt.figure(figsize=(20, 10 * num_mixtures))
    for rank, i in enumerate(sorted_indices):
        marginal_prob = marginal_probs[i]

        # First column: plot arrows of the predicted flow field
        ax1 = fig.add_subplot(num_mixtures, 2, 2*rank + 1, projection='3d')
        ax1.quiver(flat_positions[:, 0], flat_positions[:, 1], flat_positions[:, 2], 
                   mixture_means[:, i, 0], mixture_means[:, i, 1], mixture_means[:, i, 2], 
                   length=arrow_length_factor, color='red', label='Predicted', alpha=0.5)
        ax1.quiver(flat_positions[:, 0], flat_positions[:, 1], flat_positions[:, 2], 
                   true_outputs[:, 0], true_outputs[:, 1], true_outputs[:, 2], 
                   length=arrow_length_factor, color='blue', label='True', alpha=0.5)
        ax1.set_title(f'Mixture {i} Flow Field (p={marginal_prob:.3f})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        
        # Second column: plot probabilities for each mixture
        ax2 = fig.add_subplot(num_mixtures, 2, 2*rank + 2, projection='3d')
        sc = ax2.scatter(flat_positions[:, 0], flat_positions[:, 1], flat_positions[:, 2],
                         s=3, c=probabilities[:, i].flatten(), cmap='jet', vmin=0, vmax=1)
        ax2.set_title(f'Mixture {i} Probabilities (p={marginal_prob:.3f})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        fig.colorbar(sc, ax=ax2, label='Probability', shrink=0.5)

    plt.tight_layout()
    plt.show()

    if return_samples:
        return samples


def rbf_kernel(
    dataset_1: jnp.ndarray,
    dataset_2: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """
    Compute the RBF kernel between two datasets.

    Args:
        dataset_1: The first dataset. Shape (n_samples_1, n_dim).
        dataset_2: The second dataset. Shape (n_samples_2, n_dim).
        sigma: The bandwidth of the RBF kernel.

    Returns:
        The RBF kernel matrix. Shape (n_samples_1, n_samples_2).
    """
    difference_matrix = dataset_1[:, None, :] - dataset_2[None, :, :] # shape (n_samples_1, n_samples_2, n_dim)
    kernel_matrix = jnp.exp(-jnp.linalg.norm(difference_matrix, axis=-1)**2 / (2 * sigma**2))
    return kernel_matrix

def compute_mmd(
    dataset_1: jnp.ndarray,
    dataset_2: jnp.ndarray,
    kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets.

    Args:
        dataset_1: The first dataset. Shape (n_samples_1, n_dim).
        dataset_2: The second dataset. Shape (n_samples_2, n_dim).
        kernel_fn: The kernel function.

    Returns:
        The MMD between the two datasets. Float.
    """
    # compute the kernel matrix for each dataset
    n_samples_1 = dataset_1.shape[0]
    n_samples_2 = dataset_2.shape[0]
    kernel_matrix_1_1 = kernel_fn(dataset_1, dataset_1) # shape (n_samples_1, n_samples_1)
    kernel_matrix_1_2 = kernel_fn(dataset_1, dataset_2) # shape (n_samples_1, n_samples_2)
    kernel_matrix_2_2 = kernel_fn(dataset_2, dataset_2) # shape (n_samples_2, n_samples_2)

    # set diagonal elements to 0 to kernel matrices 1_1 and 2_2
    kernel_matrix_1_1 = kernel_matrix_1_1.at[jnp.diag_indices_from(kernel_matrix_1_1)].set(0)
    kernel_matrix_2_2 = kernel_matrix_2_2.at[jnp.diag_indices_from(kernel_matrix_2_2)].set(0)

    # compute the MMD
    mmd = (
        kernel_matrix_1_1.sum() / (n_samples_1 * (n_samples_1 - 1)) +
        kernel_matrix_2_2.sum() / (n_samples_2 * (n_samples_2 - 1)) -
        2 * kernel_matrix_1_2.sum() / (n_samples_1 * n_samples_2)
    )
    return mmd


@partial(jit, static_argnames=('exclude_diagonal',))
def _compute_rbf_kernel_block_sum(
    chunk_1: jnp.ndarray,
    chunk_2: jnp.ndarray,
    sigma: float,
    exclude_diagonal: bool,
) -> float:
    """
    Compute the RBF kernel for a block of data and return the sum of kernel values.

    This is a helper function for memory-efficient MMD computation. It computes
    the kernel values for a block and immediately sums them, avoiding the need
    to store the full kernel matrix.

    Args:
        chunk_1: First chunk of data. Shape (chunk_size_1, n_dim).
        chunk_2: Second chunk of data. Shape (chunk_size_2, n_dim).
        sigma: The bandwidth of the RBF kernel.
        exclude_diagonal: Whether to exclude diagonal elements (for k(X,X) and k(Y,Y)).

    Returns:
        The sum of kernel values in this block.
    """
    diff = chunk_1[:, None, :] - chunk_2[None, :, :]  # shape (chunk_size_1, chunk_size_2, n_dim)
    kernel_block = jnp.exp(-jnp.sum(diff**2, axis=-1) / (2 * sigma**2))  # shape (chunk_size_1, chunk_size_2)

    # Zero out diagonal elements if needed (for unbiased MMD estimator)
    if exclude_diagonal:
        mult_matrix = jnp.ones(kernel_block.shape) - jnp.eye(kernel_block.shape[0])
        kernel_block = kernel_block * mult_matrix
        #kernel_block = kernel_block.at[jnp.diag_indices_from(kernel_block)].set(0.0)

    return kernel_block.sum()


def compute_mmd_chunked(
    dataset_1: jnp.ndarray,
    dataset_2: jnp.ndarray,
    sigma: float,
    chunk_size: int = 2000,
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets using a
    memory-efficient chunked approach.

    This function computes kernel values in blocks and accumulates their sums,
    avoiding the need to materialize the full kernel matrices. This is essential
    for large datasets where the full kernel matrices would cause out-of-memory errors.

    Args:
        dataset_1: The first dataset. Shape (n_samples_1, n_dim).
        dataset_2: The second dataset. Shape (n_samples_2, n_dim).
        sigma: The bandwidth of the RBF kernel.
        chunk_size: The size of chunks to process at a time. Larger chunks are faster
                   but use more memory. Default is 2000, which uses ~16MB per block.

    Returns:
        The MMD between the two datasets. Float.
    """
    n1 = dataset_1.shape[0]
    n2 = dataset_2.shape[0]

    # Accumulate kernel sums
    sum_k11 = 0.0  # sum of k(X, X) excluding diagonal
    sum_k22 = 0.0  # sum of k(Y, Y) excluding diagonal
    sum_k12 = 0.0  # sum of k(X, Y)

    # Process k(X, X) in chunks - need to exclude diagonal on diagonal blocks
    for i in range(0, n1, chunk_size):
        i_end = min(i + chunk_size, n1)
        chunk_i = dataset_1[i:i_end]
        for j in range(0, n1, chunk_size):
            j_end = min(j + chunk_size, n1)
            chunk_j = dataset_1[j:j_end]
            sum_k11 = sum_k11 + _compute_rbf_kernel_block_sum(
                chunk_i, chunk_j, sigma,
                exclude_diagonal=(i == j),
            )

    # Process k(Y, Y) in chunks - need to exclude diagonal on diagonal blocks
    for i in range(0, n2, chunk_size):
        i_end = min(i + chunk_size, n2)
        chunk_i = dataset_2[i:i_end]
        for j in range(0, n2, chunk_size):
            j_end = min(j + chunk_size, n2)
            chunk_j = dataset_2[j:j_end]
            sum_k22 = sum_k22 + _compute_rbf_kernel_block_sum(
                chunk_i, chunk_j, sigma,
                exclude_diagonal=(i == j),
            )

    # Process k(X, Y) in chunks - no diagonal exclusion needed
    for i in range(0, n1, chunk_size):
        i_end = min(i + chunk_size, n1)
        chunk_i = dataset_1[i:i_end]
        for j in range(0, n2, chunk_size):
            j_end = min(j + chunk_size, n2)
            chunk_j = dataset_2[j:j_end]
            sum_k12 = sum_k12 + _compute_rbf_kernel_block_sum(
                chunk_i, chunk_j, sigma,
                exclude_diagonal=False,
            )

    # print(f"sum_k11: {sum_k11}")
    # print(f"sum_k22: {sum_k22}")
    # print(f"sum_k12: {sum_k12}")

    term_11 = sum_k11 / float(n1 * (n1 - 1))
    term_22 = sum_k22 / float(n2 * (n2 - 1))
    term_12 = 2 * sum_k12 / float(n1 * n2)
    mmd = term_11 + term_22 - term_12
    return mmd