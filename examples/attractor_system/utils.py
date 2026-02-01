# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad

from flax import linen as nn
from flax.core import FrozenDict
import optax

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as onp
import copy
import pickle
import pandas as pd
from tqdm import trange
import math
from typing import Tuple, Optional, Any


def generate_attractor_data(
    n_samples: int,
    n_dim: int,
    attractors: jnp.ndarray,
    time_steps: int = 100,
    key: jax.Array = random.PRNGKey(0),
    final_time: float = 5,
    dt: Optional[float] = None,
    initial_position_radius: float = 0.05,
    gravitational_constant: float = 100,
    friction_coefficient: float = 10,
    noise_factor: float = 0.1,
    random_attractor_sampling: bool = False,
    observation_noise: float = 0.,
    input_observation_noise: float = 0.,
) -> tuple:
  """
  Generates synthetic trajectory data for a system of particles influenced by attractors in n-dimensional space.

  Args:
      n_samples (int): Number of trajectories (particles) to generate.
      n_dim (int): Dimension of the space (e.g., 2 for 2D).
      attractors (jnp.ndarray): Array of shape (n_attractors, n_dim) specifying the positions of attractors.
      time_steps (int, optional): Number of time steps to include in each trajectory. Default is 100.
      key (jax.random.PRNGKey, optional): JAX PRNG key for reproducibility. Default is random.PRNGKey(0).
      final_time (float, optional): Total length of simulated time. Default is 5.
      dt (float or None, optional): Time step size. Computed from final_time / time_steps if None.
      initial_position_radius (float, optional): Radius for sampling initial positions uniformly from a disk. Default is 0.05.
      gravitational_constant (float, optional): Constant controlling magnitude of attraction toward attractors. Default is 100.
      friction_coefficient (float, optional): Friction coefficient for velocity damping. Default is 10.
      noise_factor (float, optional): Magnitude of the true evolution noise per time step. Default is 0.1.
      random_attractor_sampling (bool, optional): If True, each sample's dynamics are governed by a randomly chosen attractor. Otherwise, all attractors influence each sample. Default is False.
      observation_noise (float, optional): Standard deviation of Gaussian noise added to positions at each time step. Default is 0.
      input_observation_noise (float, optional): Standard deviation of Gaussian noise added to inputs. Default is 0.

  Returns:
      Tuple containing subsampled positions and the final random key. (Actual return not shown in this code snippet.)
  """
  # attractors should be shape (n_attractors, n_dim)
  assert attractors.shape[-1] == n_dim
  if random_attractor_sampling:
    key, subkey = random.split(key)
    # choose randomly one of the attractors for each sample
    attractors = random.choice(subkey, attractors, shape=(n_samples, 1)) # shape (n_samples, 1, n_dim)
  else:
    attractors = attractors[None,:,:] # shape (1, n_attractors, n_dim)
  if dt is None:
    dt = final_time / time_steps
  total_time_steps = int(final_time / dt)
  trajectories = jnp.zeros((n_samples, total_time_steps, n_dim))
  # sample initial positions uniformly on the disk of radius initial_position_radius
  # use polar coordinates
  key, subkey = random.split(key)
  angles = random.uniform(subkey, shape=(n_samples, 1), minval=0, maxval=2*jnp.pi)
  key, subkey = random.split(key)
  radii = initial_position_radius*jnp.sqrt(random.uniform(subkey, shape=(n_samples, 1), minval=0, maxval=1))
  initial_positions = jnp.concatenate([radii*jnp.cos(angles), radii*jnp.sin(angles)], axis=-1)
  current_positions = initial_positions
  trajectories = trajectories.at[:,0,:].set(initial_positions)
  current_velocities = jnp.zeros((n_samples, n_dim))

  @jit
  def compute_next_state(current_positions, current_velocities, attractors, key):
    distances = current_positions[:,None,:] - attractors # shape (n_samples, n_attractors, n_dim)
    # compute the force towards the attractors
    euclidean_distances = jnp.linalg.norm(distances, axis=-1, keepdims=True) # shape (n_samples, n_attractors, 1)
    directions = distances / (euclidean_distances + 1e-6) # shape (n_samples, n_attractors, n_dim)
    gravitational_forces = -gravitational_constant * directions / (euclidean_distances**2 + 1e-2)  # shape (n_samples, n_attractors, n_dim)
    velocity_norm = jnp.linalg.norm(current_velocities, axis=-1, keepdims=True)
    frictional_forces = -friction_coefficient * velocity_norm * current_velocities # shape (n_samples, n_dim, 1)
    total_force = frictional_forces + jnp.sum(gravitational_forces, axis=-2) # shape (n_samples, n_dim)
    # update the trajectory
    noise = random.normal(key, shape=(n_samples, n_dim)) * noise_factor * jnp.sqrt(dt)
    new_positions = current_positions + dt * current_velocities + dt**2 * total_force / 2 + noise
    # update the velocity
    new_velocities = current_velocities + dt * total_force
    return new_positions, new_velocities

  pbar = trange(1, total_time_steps)
  for t in pbar:
    key, subkey = random.split(key)
    current_positions, current_velocities = compute_next_state(current_positions, current_velocities, attractors, subkey)
    trajectories = trajectories.at[:,t,:].set(current_positions)
  # subsample the trajectories to the final time
  subsample_rate = math.ceil(total_time_steps / time_steps)
  subsampled_trajectories = trajectories[:,::subsample_rate,:]
  key, subkey = random.split(key)
  observation_noise = random.normal(subkey, shape=(n_samples, time_steps, n_dim)) * observation_noise
  subsampled_trajectories = subsampled_trajectories + observation_noise
  print(f"{subsample_rate=}")
  print(f"{subsampled_trajectories.shape=}")
  print(f"{trajectories.shape=}")

  if input_observation_noise > 0:
    key, subkey = random.split(key)
    input_observation_noise = random.normal(subkey, shape=(n_samples, n_dim)) * input_observation_noise
    initial_positions = initial_positions + input_observation_noise
  return initial_positions, subsampled_trajectories.reshape(n_samples, -1)



def plot_mdn_mixture_elements(
    model: Any,
    true_inputs: jnp.ndarray,
    true_outputs: jnp.ndarray,
    net_id: int,
    cutoff: int,
    probability_cutoff: int = -1,
    attractors: jnp.ndarray = None,
) -> None:
    """
    Visualize mixture component means and mixture probabilities from an MDN (Mixture Density Network).

    For each mixture component, the function generates:
      - (Left) Plots of example trajectories for both predicted mixture means and their true counterparts.
      - (Right) A scatter plot of the initial positions colored by the mixture's predicted assignment probability.

    Args:
        model: The trained MDN or similar model with `.apply()` and `.params`.
        true_inputs: Array of initial conditions, shape (batch, n_dim).
        true_outputs: Array of true output trajectories, shape (batch, time_steps * n_dim).
        net_id: Index of the ensemble network to visualize (int).
        cutoff: Number of trajectories to display per mixture for prediction and target comparisons.
        probability_cutoff: Number of initial positions to include in the probability scatter plot
            (if -1, show all samples; default: -1).
        attractors: Optional. Array of attractor positions to plot for visual reference.

    Returns:
        None. Displays matplotlib figures for mixture means and probabilities.
    """
    n_dim = true_inputs.shape[-1]
    time_steps = true_outputs.shape[-1] // n_dim

    pred = model.apply(model.params, true_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    num_mixtures = mixture_logit_weights.shape[-2]

    probabilities = nn.softmax(mixture_logit_weights[net_id], axis=-2)  # (batch, num_mixtures, 1) or (batch, num_mixtures)

    # Create a figure with num_mixtures rows and 2 columns
    n_rows = num_mixtures
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))

    # If there's only one mixture, axes may not be a 2d array
    if num_mixtures == 1:
        axes = axes[None, :]

    # Plot for each mixture
    for mix_id in range(num_mixtures):
        # Left column: plot sample trajectories for this mixture
        ax_traj = axes[mix_id, 0]
        for i in range(cutoff):
            traj = mixture_means[net_id, i, mix_id, :].reshape(time_steps, n_dim)
            xs = traj[:, 0]
            ys = traj[:, 1]
            true_traj = true_outputs[i].reshape(time_steps, n_dim)
            if i == cutoff - 1:
                ax_traj.plot(xs, ys, c='red', alpha=0.1, label='Pred')
                ax_traj.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=0.1, label='True')
            else:
                ax_traj.plot(xs, ys, c='red', alpha=0.1)
                ax_traj.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=0.1)
        if attractors is not None:
            ax_traj.scatter(attractors[:, 0], attractors[:, 1], s=10, c='green', label='Attractors')
        ax_traj.set_aspect('equal', 'box')
        ax_traj.set_xlim(-1.5, 1.5)
        ax_traj.set_ylim(-1.5, 1.5)
        ax_traj.set_title(f"Mixture {mix_id} (p={probabilities[:,mix_id].mean():.3f})")
        if mix_id == 0:
            ax_traj.legend()
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        
        # Right column: probability scatter for initial positions
        ax_prob = axes[mix_id, 1]
        Xs = true_inputs[:probability_cutoff, 0]
        Ys = true_inputs[:probability_cutoff, 1]
        probs = probabilities[:probability_cutoff, mix_id]
        sc = ax_prob.scatter(
            Xs, 
            Ys, 
            c=probs, 
            s=5, vmin=0, vmax=1, cmap='jet'
        )
        fig.colorbar(sc, ax=ax_prob, fraction=0.05, pad=0.04, label="Probability")
        ax_prob.set_title(f"P(Mixture {mix_id} | initial position)")
        ax_prob.set_xlabel("initial x")
        ax_prob.set_ylabel("initial y")

    plt.suptitle(f"Network {net_id}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_mdn_mixture_samples(
    key: jax.Array,
    model: Any,
    true_inputs: jnp.ndarray,
    true_outputs: jnp.ndarray,
    net_id: int,
    cutoff: int,
    probability_cutoff: int = -1,
    attractors: jnp.ndarray = None,
    reorder_mixtures: bool = False,
    weighted_opacity: bool = True,
    suptitle: str = None,
) -> None:
    """
    Visualize mixture component samples from an MDN (Mixture Density Network).

    For each mixture component, the function generates:
      - (Left) Plots of sample trajectories and their true counterparts.
      - (Right) A scatter plot of the initial positions colored by the mixture's predicted assignment probability.

    Args:
        key: jax.Array, random key for sampling.
        model: The trained MDN or similar model with `.apply()` and `.params`.
        true_inputs: Array of initial conditions, shape (batch, n_dim).
        true_outputs: Array of true output trajectories, shape (batch, time_steps * n_dim).
        net_id: Index of the ensemble network to visualize (int).
        cutoff: Number of trajectories to display per mixture for prediction and target comparisons.
        probability_cutoff: Number of initial positions to include in the probability scatter plot
            (if -1, show all samples; default: -1).
        attractors: Optional. Array of attractor positions to plot for visual reference.
        reorder_mixtures: Optional. Whether to reorder the mixtures by probability. Default is False.
        weighted_opacity: Optional. Whether to use probability weighted opacity for the trajectories. Default is False.
        suptitle: Optional. Title for the figure. Default is to print the network id.
    Returns:
        None. Displays matplotlib figures for mixture means and probabilities.
    """
    n_dim = true_inputs.shape[-1]
    time_steps = true_outputs.shape[-1] // n_dim

    pred = model.apply(model.params, true_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    num_mixtures = mixture_logit_weights.shape[-2]

    probabilities = nn.softmax(mixture_logit_weights[net_id], axis=-2)  # (batch, num_mixtures, 1)
    if reorder_mixtures:
        new_order = jnp.argsort(probabilities.mean(0).squeeze(), descending=True)
    # Create a figure with num_mixtures rows and 2 columns
    n_rows = num_mixtures
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4 * n_rows))

    # If there's only one mixture, axes may not be a 2d array
    if num_mixtures == 1:
        axes = axes[None, :]

    # Plot for each mixture
    for plot_id in range(num_mixtures):
        if reorder_mixtures:
            mix_id = new_order[plot_id]
        else:
            mix_id = plot_id
        # Left column: plot sample trajectories for this mixture
        ax_traj = axes[plot_id, 0]
        for i in range(cutoff):
            key, subkey = random.split(key)
            traj_mean = mixture_means[net_id, i, mix_id, :].reshape(time_steps, n_dim)
            traj_std = jnp.sqrt(mixture_variances[net_id, i, mix_id, :]).reshape(time_steps, n_dim)
            if weighted_opacity:
                alpha = float(probabilities[i, mix_id].squeeze())*0.2
            else:
                alpha = 0.1
            traj = traj_mean + traj_std * random.normal(subkey, shape=(time_steps, n_dim))
            xs = traj[:, 0]
            ys = traj[:, 1]
            true_traj = true_outputs[i].reshape(time_steps, n_dim)
            if i == cutoff - 1:
                ax_traj.plot(xs, ys, c='red', alpha=alpha, label='Pred')
                ax_traj.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=alpha, label='True')
            else:
                ax_traj.plot(xs, ys, c='red', alpha=alpha)
                ax_traj.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=alpha)
        if attractors is not None:
            ax_traj.scatter(attractors[:, 0], attractors[:, 1], s=10, c='green', label='Attractors')
        ax_traj.set_aspect('equal', 'box')
        ax_traj.set_xlim(-1.5, 1.5)
        ax_traj.set_ylim(-1.5, 1.5)
        ax_traj.set_title(f"Mixture {plot_id} (p={probabilities[:,mix_id].mean():.3f})")
        if plot_id == 0:
            ax_traj.legend(handles=[Line2D([0], [0], color='red'),
                                    Line2D([0], [0], color='blue')],
                           labels=['Pred', 'True'])
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        
        # Right column: probability scatter for initial positions
        ax_prob = axes[plot_id, 1]
        Xs = true_inputs[:probability_cutoff, 0]
        Ys = true_inputs[:probability_cutoff, 1]
        probs = probabilities[:probability_cutoff, mix_id]
        sc = ax_prob.scatter(
            Xs, 
            Ys, 
            c=probs, 
            s=5, vmin=0, vmax=1, cmap='jet'
        )
        fig.colorbar(sc, ax=ax_prob, fraction=0.05, pad=0.04, label="Probability")
        ax_prob.set_title(f"P(Mixture {plot_id} | initial position)")
        ax_prob.set_xlabel("initial x")
        ax_prob.set_ylabel("initial y")

    if suptitle is None:
        plt.suptitle(f"Network {net_id}", fontsize=18)
    else:
        plt.suptitle(suptitle, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_mdn_ensemble_samples(
    model: Any,
    true_inputs: jnp.ndarray,
    true_outputs: jnp.ndarray,
    cutoff: int,
    attractors: jnp.ndarray = None,
    compute_nll: bool = True,
) -> None:
    """
    Plot sample trajectories generated from an ensemble of Mixture Density Networks (MDNs) alongside ground truth trajectories.

    For each network in the ensemble, this function:
      - Draws sample trajectories for a specified number of initial conditions (`cutoff`)
      - Optionally overlays attractor positions if provided
      - Optionally computes and displays negative log likelihood (NLL) statistics for each network

    Args:
        model (Any): The MDN ensemble model with an 'apply' and 'sample_from_mixture' method.
        true_inputs (jnp.ndarray): Input array of initial conditions, shape (batch, input_dim).
        true_outputs (jnp.ndarray): Ground truth output trajectories, shape (batch, time_steps * output_dim).
        cutoff (int): Number of initial conditions/trajectories to plot per network.
        attractors (jnp.ndarray, optional): Array of attractor positions, shape (num_attractors, 2).
        compute_nll (bool, optional): Whether to compute and display NLL statistics for each network. Default is True.

    Returns:
        None. Displays matplotlib plots for ensemble members.
    """

    n_dim = true_inputs.shape[-1]
    time_steps = true_outputs.shape[-1] // n_dim
    
    pred = model.apply(model.params, true_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    samples = model.sample_from_mixture(
        random.PRNGKey(123), 
        mixture_logit_weights[:, :cutoff], 
        mixture_means[:, :cutoff], 
        mixture_variances[:, :cutoff], 
        restrict_rare_event_rate=0.05
    )

    if compute_nll:
        nll_per_element = model.per_element_loss(pred, true_outputs)[:, :cutoff]
        print(f"NLL shape: {nll_per_element.shape}")
        mean_nll_per_ensemble = nll_per_element.mean(axis=(1,2)) # shape (ensemble_size,)
        print(f"Mean NLL: {mean_nll_per_ensemble.mean():.3f}")
        print(f"Std NLL: {jnp.std(mean_nll_per_ensemble):.3f}")
    
    ensemble_size = samples.shape[0]
    max_cols = 3
    n_rows = math.ceil(ensemble_size / max_cols)
    n_cols = min(max_cols, ensemble_size)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten() if ensemble_size > 1 else [axes]

    for net_id in range(ensemble_size):
        ax = axes[net_id]
        for i in range(cutoff):
            traj = samples[net_id, i].reshape(time_steps, n_dim)
            xs = traj[:, 0]
            ys = traj[:, 1]
            true_traj = true_outputs[i].reshape(time_steps, n_dim)
            if i == cutoff-1:
                ax.plot(xs, ys, c='red', alpha=0.1, label='Pred')
                ax.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=0.1, label='True')
            else:
                ax.plot(xs, ys, c='red', alpha=0.1)
                ax.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=0.1)
        ax.scatter(true_inputs[:cutoff, 0], true_inputs[:cutoff, 1], s=1, label='Initial Positions')
        if attractors is not None:
            ax.scatter(attractors[:, 0], attractors[:, 1], s=10, c='green', label='Attractors')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        if compute_nll:
            ax.set_title(f"Ensemble Network {net_id}\nMean NLL: {nll_per_element[net_id].mean():.3f}\nMedian NLL: {jnp.median(nll_per_element[net_id]):.3f}")
        else:
            ax.set_title(f"Ensemble Network {net_id}")
        ax.legend()

    # Hide unused subplots if any
    for idx in range(ensemble_size, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Predictive Samples from Ensemble of {ensemble_size} networks\n", fontsize=18)

    plt.tight_layout()
    plt.show()



def plot_cfm_ensemble_samples(
    model,
    true_inputs,
    true_outputs,
    cutoff,
    ensemble_size: int,
    max_cols: int = 3,
    key=random.PRNGKey(1234),
    num_euler_steps: int = 2_500,
    compute_nll: bool = True,
    attractors: jnp.ndarray = None,
    ):
    """
    Plots ensemble samples generated from a Conditional Flow Matching (CFM) model for given true inputs and outputs.

    Args:
        model: The CFM model object used for generation and (optionally) NLL computation.
        true_inputs: jnp.ndarray of shape (batch_size, input_dim); initial input positions.
        true_outputs: jnp.ndarray of shape (batch_size, output_dim * time_steps); ground-truth trajectories.
        cutoff: int, number of trajectories per ensemble member to plot.
        ensemble_size: int, number of ensemble models/samples to generate and plot.
        max_cols: int, optional, max columns in plot grid (default: 3).
        key: jax.random.PRNGKey, optional, for randomness (default: PRNGKey(1234)).
        num_euler_steps: int, optional, number of Euler steps for integration/generation (default: 1000).
        compute_nll: bool, optional, whether to compute and display Negative Log-Likelihood (default: False).
        attractors: jnp.ndarray, optional, array of attractor positions to plot (default: None).

    Returns:
        None. Displays a matplotlib figure of predicted and true trajectories for each ensemble member.
    """
    n_dim = true_inputs.shape[-1]
    time_steps = true_outputs.shape[-1] // n_dim

    # tile inputs and outputs
    tiled_true_inputs = jnp.tile(true_inputs, (ensemble_size, 1, 1))
    tiled_true_outputs = jnp.tile(true_outputs, (ensemble_size, 1, 1))

    samples = model.generate_sample(
        key,
        model.params,
        tiled_true_inputs,
        num_output_dims=n_dim*time_steps,
        num_steps=num_euler_steps,
        apply_kwargs=FrozenDict({'tile_inputs': False}),
    )

    if compute_nll:
        nll_per_element = model.compute_nll(
            model.params,
            tiled_true_inputs,
            tiled_true_outputs,
            num_steps=num_euler_steps,
            apply_kwargs=FrozenDict({'tile_inputs': False},)
        )
        print(f"NLL shape: {nll_per_element.shape}")
        mean_nll_per_ensemble = nll_per_element.mean(axis=(1,2)) # shape (ensemble_size,)
        print(f"Mean NLL: {mean_nll_per_ensemble.mean():.3f}")
        print(f"Std NLL: {jnp.std(mean_nll_per_ensemble):.3f}")
    
    ensemble_size = samples.shape[0]
    n_rows = math.ceil(ensemble_size / max_cols)
    n_cols = min(max_cols, ensemble_size)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten() if ensemble_size > 1 else [axes]

    for net_id in range(ensemble_size):
        ax = axes[net_id]
        for i in range(cutoff):
            traj = samples[net_id, i].reshape(time_steps, n_dim)
            xs = traj[:, 0]
            ys = traj[:, 1]
            true_traj = tiled_true_outputs[net_id, i].reshape(time_steps, n_dim)
            if i == cutoff-1:
                ax.plot(xs, ys, c='red', alpha=0.1, label='Pred')
                ax.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=0.1, label='True')
            else:
                ax.plot(xs, ys, c='red', alpha=0.1)
                ax.plot(true_traj[:, 0], true_traj[:, 1], c='blue', alpha=0.1)
        ax.scatter(true_inputs[:cutoff, 0], true_inputs[:cutoff, 1], s=1, label='Initial Positions')
        if attractors is not None:
            ax.scatter(attractors[:, 0], attractors[:, 1], s=10, c='green', label='Attractors')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        if compute_nll:
            axes[net_id].set_title(
                f"Network {net_id}\nMean NLL: {nll_per_element[net_id].mean():.3f}\nMedian NLL: {jnp.median(nll_per_element[net_id]):.3f}"
            )
        else:
            axes[net_id].set_title(f"Generated Samples for Network {net_id}")
        ax.legend()

    # Hide unused subplots if any
    for idx in range(ensemble_size, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Predictive Samples from Ensemble of {ensemble_size} networks\n", fontsize=18)

    plt.tight_layout()
    plt.show()