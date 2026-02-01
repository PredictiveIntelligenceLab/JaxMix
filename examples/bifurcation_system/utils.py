# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad
from jax import lax
from functools import partial

from flax import linen as nn
from flax.core import FrozenDict
import optax

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from tqdm import trange
import math
from typing import Tuple, Optional, Any

from jaxmix.utils import log_normal_pdf

def _to_np(x):
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    return np.asarray(x)

def _time_axis_from_outputs(outputs):
    """Derive time points from outputs shape if no physical time vector is provided."""
    T = int(outputs.shape[-1])
    return np.arange(T)

def _mixture_probs(logits, axis=-2):
    """
    Softmax over mixture axis; if a time axis exists, average across time
    to get per-sample mixture probabilities.
    Expected shapes:
      - (E, N, M)  -> (N, M)
      - (E, N, M, T) -> (N, M) by mean over T
    """
    probs = nn.softmax(logits, axis=axis)
    # Remove leading ensemble dim after indexing later; handle time dim if present
    if probs.ndim == 4:   # (E, N, M, T)
        probs = probs.mean(axis=-1)  # average over time
    elif probs.ndim == 3: # (E, N, M)
        pass
    elif probs.ndim == 2: # (N, M) (already squeezed ensemble)
        pass
    else:
        raise ValueError(f"Unexpected logits/prob shape: {probs.shape}")
    return probs


def plot_mixture_outputs_1d(
    inputs, outputs, pred, net_id, cutoff, probability_cutoff=-1, title_prefix="Saddle-Node"
):
    """
    Plot mixture predictions for a 1D saddle-node system.

    Args:
        inputs:  (N, 1) -> [x0] or (N, 2) -> [x0, r] if bifurcation parameter is available
        outputs: (N, T) true trajectories
        pred: Tuple (mixture_logit_weights, mixture_means, mixture_variances)
              Expected shapes (E, N, M, T) for means/vars; logits can be (E, N, M) or (E, N, M, T)
        net_id: int, which ensemble member to visualize
        cutoff: int, # of trajectories to overlay for time-series panes
        probability_cutoff: int, how many samples to show in prob scatters (-1 => all)
        title_prefix: str, shown in figure title
    """
    mixture_logit_weights, mixture_means, mixture_variances = pred

    # Basic dims
    E = mixture_means.shape[0]
    N = mixture_means.shape[1]
    M = mixture_means.shape[2]
    T = mixture_means.shape[3]

    if not (0 <= net_id < E):
        raise ValueError(f"net_id {net_id} out of range [0,{E-1}]")

    cutoff = int(min(cutoff, N))
    if probability_cutoff == -1:
        probability_cutoff = N
    probability_cutoff = int(min(probability_cutoff, N))

    # Mixture probabilities per-sample (N, M)
    probs = _mixture_probs(mixture_logit_weights[net_id], axis=-2)  # (N,M)
    probs = _to_np(probs)

    # Inputs / outputs slices for scatter
    inputs_np  = _to_np(inputs)
    outputs_np = _to_np(outputs)
    x0_all = inputs_np[:probability_cutoff, 0]
    has_r = inputs_np.shape[1] > 1
    if has_r:
        r_all  = inputs_np[:probability_cutoff, 1]
    else:
        r_all = None

    # Time vector
    time_points = _time_axis_from_outputs(outputs_np)

    # Figure grid: one row per mixture: [ Traj | P(m|r or x0) | P(m|x0,r) (if r exists) ]
    n_rows, n_cols = M, 3 if has_r else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 if has_r else 10, 4 * n_rows), squeeze=False)

    for m in range(M):
        # --- Left column: trajectories (pred mean of that mixture vs truth) ---
        ax_traj = axes[m, 0]
        for i in range(cutoff):
            alpha_factor = max(0.01, float(probs[i, m]))
            pred_traj = _to_np(mixture_means[net_id, i, m, :])  # (T,)
            true_traj = outputs_np[i, :]
            ax_traj.plot(time_points, pred_traj, c='red', alpha=0.5*alpha_factor)
            ax_traj.fill_between(time_points, pred_traj - 2 * jnp.sqrt(mixture_variances[net_id, i, m, :]), pred_traj + 2 * jnp.sqrt(mixture_variances[net_id, i, m, :]), color='orange', alpha=0.2*alpha_factor)
            ax_traj.plot(time_points, true_traj, c='black', alpha=0.8*alpha_factor)
        ax_traj.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax_traj.set_xlabel("Time")
        ax_traj.set_ylabel("x(t)")
        ax_traj.set_title(f"Mixture {m} Trajectories  (mean p={probs[:, m].mean():.3f})")
        ax_traj.grid(True, alpha=0.3)

        # Create legend just once per row to avoid clutter
        ax_traj.plot([], [], c='red', alpha=0.7, label='Pred (mixture mean)')
        ax_traj.plot([], [], c='black', alpha=0.7, label='True')
        ax_traj.legend(frameon=False)
        ax_traj.set_ylim(-3.0, 12.5)

        # --- Middle column: Probability vs r (or x0) ---
        ax_mid = axes[m, 1]
        pm = probs[:probability_cutoff, m]
        if has_r:
            sc = ax_mid.scatter(r_all, pm, c=x0_all, s=6, alpha=0.6, cmap='viridis')
            ax_mid.axvline(x=0.0, color='red', linestyle='--', alpha=0.5, label='Bifurcation r=0')
            ax_mid.set_xlabel("Bifurcation parameter r")
            ax_mid.set_title("Mixture probability vs r")
            fig.colorbar(sc, ax=ax_mid, fraction=0.05, pad=0.04, label="x0")
        else:
            sc = ax_mid.scatter(x0_all, pm, s=6, alpha=0.6)
            ax_mid.set_xlabel("Initial state x0")
            ax_mid.set_title("Mixture probability vs x0")
        ax_mid.set_ylabel(f"P(Mixture {m})")
        ax_mid.set_ylim(0, 1)
        ax_mid.grid(True, alpha=0.3)
        ax_mid.legend(frameon=False)

        # --- Right column: Probability in (x0, r) plane when r exists ---
        if has_r:
            ax_right = axes[m, 2]
            sc2 = ax_right.scatter(x0_all, r_all, c=pm, s=6, vmin=0.0, vmax=1.0,
                                   cmap='RdYlBu_r', alpha=0.7)
            ax_right.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Bifurcation r=0')
            ax_right.set_xlabel("Initial state x0")
            ax_right.set_ylabel("Bifurcation parameter r")
            ax_right.set_title(f"P(Mixture {m} | x0, r)")
            ax_right.grid(True, alpha=0.3)
            ax_right.legend(frameon=False)
            fig.colorbar(sc2, ax=ax_right, fraction=0.05, pad=0.04, label="Probability")

    plt.suptitle(f"{title_prefix} — Network {net_id}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def plot_ensemble_samples_1d(model, inputs, outputs, cutoff, rng_key=123, compute_nll=True, restrict_rare_event_rate=None, truncated_normal_std_limit=None, save_path=None):
    """
    Plot predictive samples from each ensemble member vs. truth for the 1D system.
    Assumes model.sample_from_mixture(key, logits, means, vars, ...) -> (E, N, T) or (E, cutoff, T)

    Args:
        model: The MDN ensemble model with 'apply' and 'sample_from_mixture' methods
        inputs: Input array of initial conditions, shape (batch, 1) -> [x0]
        outputs: Ground truth output trajectories, shape (batch, time_steps)
        cutoff: Number of trajectories per ensemble member to plot
        rng_key: Random key for sampling
        compute_nll: Whether to compute and display Negative Log-Likelihood statistics
        restrict_rare_event_rate: Optional rate for rare event restriction
        truncated_normal_std_limit: Optional truncation limit for normal distribution
        save_path: If provided, save the figure to this path
    """
    pred = model.apply(model.params, inputs) 
    mixture_logit_weights, mixture_means, mixture_variances = pred

    N = mixture_means.shape[1]
    T = mixture_means.shape[3]
    cutoff = int(min(cutoff, N))

    if compute_nll:
        nll_per_element = model.per_element_loss(pred, outputs)[:, :cutoff]

    samples = model.sample_from_mixture(
        random.PRNGKey(rng_key),
        mixture_logit_weights[:, :cutoff],
        mixture_means[:, :cutoff],
        mixture_variances[:, :cutoff],
        restrict_rare_event_rate=restrict_rare_event_rate,
        truncated_normal_std_limit=truncated_normal_std_limit
    )  # shape expected: (E, cutoff, T)

    samples_np = _to_np(samples)
    # Squeeze out any extra dimensions that might be present
    if samples_np.ndim > 3:
        samples_np = samples_np.squeeze()
    outputs_np = _to_np(outputs[:cutoff])
    time_points = _time_axis_from_outputs(outputs_np)

    ensemble_size = samples_np.shape[0]
    max_cols = 3
    n_rows = math.ceil(ensemble_size / max_cols)
    n_cols = min(max_cols, ensemble_size)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for e in range(ensemble_size):
        ax = axes_flat[e]
        for i in range(cutoff):
            ax.plot(time_points, samples_np[e, i, :], c='red', alpha=0.35)
            ax.plot(time_points, outputs_np[i, :],     c='blue', alpha=0.25)
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("x(t)")
        if compute_nll:
            ax.set_title(f"Ensemble member {e}\nMean NLL: {nll_per_element[e].mean():.3f}\nMedian NLL: {jnp.median(nll_per_element[e]):.3f}")
        else:
            ax.set_title(f"Ensemble member {e}")
        ax.grid(True, alpha=0.3)
        # Add legend entries for the first trajectory only
        if cutoff > 0:
            ax.plot([], [], c='red',  alpha=0.7, label='Sampled pred')
            ax.plot([], [], c='blue', alpha=0.7, label='True')
        ax.legend(frameon=False)

    # Remove any empty axes if E < n_rows*n_cols
    for j in range(ensemble_size, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(f"MDN: Predictive samples from ensemble of {ensemble_size} networks", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_bifurcation_diagram(inputs, outputs, title="Bifurcation Diagram"):
    """
    Plot bifurcation diagram for saddle-node system.
    
    Args:
        inputs: Input array, shape (N, 1) -> [x0] or (N, 2) -> [x0, r]
        outputs: Output trajectories, shape (N, T)
        title: Title for the plot
        
    For 1D inputs: Shows x0 vs x(T) relationship
    For 2D inputs: Shows bifurcation diagram with r parameter and theoretical fixed points
    """
    inputs_np  = _to_np(inputs)
    outputs_np = _to_np(outputs)

    # Handle both 1D and 2D inputs
    if inputs_np.shape[1] == 1:
        # 1D input case
        x0_vals = inputs_np[:, 0]
        final_states = outputs_np[:, -1]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Single plot: initial vs final state
        sc = ax.scatter(x0_vals, final_states, c=x0_vals, s=20, alpha=0.6, cmap='viridis')
        lo = min(np.min(x0_vals), np.min(final_states))
        hi = max(np.max(x0_vals), np.max(final_states))
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='x0 = x(T)')
        ax.set_xlabel("Initial state x0")
        ax.set_ylabel("Final state x(T)")
        ax.set_title("Initial vs Final state (1D input)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.04, label="x0")
        
    else:
        # 2D input case (original functionality)
        x0_vals = inputs_np[:, 0]
        r_vals  = inputs_np[:, 1]
        final_states = outputs_np[:, -1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: final state vs r, colored by x0
        ax1 = axes[0]
        sc1 = ax1.scatter(r_vals, final_states, c=x0_vals, s=4, alpha=0.6, cmap='viridis')
        ax1.axvline(x=0.0, color='red', linestyle='--', alpha=0.6, label='Bifurcation r=0')

        # Theoretical fixed points for r <= 0: x* = ±sqrt(-r)
        r_neg = np.linspace(min(-1.0, float(r_vals.min())), 0.0, 200)
        stable_fp   = -np.sqrt(-r_neg)   # stable
        unstable_fp =  +np.sqrt(-r_neg)  # unstable
        ax1.plot(r_neg, stable_fp,   'b-',  linewidth=2, alpha=0.8, label='Stable fixed point')
        ax1.plot(r_neg, unstable_fp, 'r--', linewidth=2, alpha=0.8, label='Unstable fixed point')

        ax1.set_xlabel("Bifurcation parameter r")
        ax1.set_ylabel("Final state x(T)")
        ax1.set_title("Bifurcation diagram")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)
        fig.colorbar(sc1, ax=ax1, fraction=0.05, pad=0.04, label="x0")

        # Right: initial vs final, colored by r
        ax2 = axes[1]
        sc2 = ax2.scatter(x0_vals, final_states, c=r_vals, s=4, alpha=0.6, cmap='RdYlBu')
        lo = min(np.min(x0_vals), np.min(final_states))
        hi = max(np.max(x0_vals), np.max(final_states))
        ax2.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='x0 = x(T)')
        ax2.set_xlabel("Initial state x0")
        ax2.set_ylabel("Final state x(T)")
        ax2.set_title("Initial vs Final state")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False)
        fig.colorbar(sc2, ax=ax2, fraction=0.05, pad=0.04, label="r")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_cfm_ensemble_samples_1d(
    model,
    true_inputs,
    true_outputs,
    cutoff,
    ensemble_size: int = 12,
    max_cols: int = 3,
    key=random.PRNGKey(1234),
    num_euler_steps: int = 2_500,
    compute_nll: bool = True,
    final_time: float = 10.0,
    save_path: str = None,
    ):
    """
    Plots ensemble samples generated from a Conditional Flow Matching (CFM) model for 1D bifurcation system.

    Args:
        model: The CFM model object used for generation and (optionally) NLL computation.
        true_inputs: jnp.ndarray of shape (batch_size, 1); initial input positions [x0].
        true_outputs: jnp.ndarray of shape (batch_size, time_steps); ground-truth trajectories.
        cutoff: int, number of trajectories per ensemble member to plot.
        ensemble_size: int, number of ensemble models/samples to generate and plot.
        max_cols: int, optional, max columns in plot grid (default: 3).
        key: jax.random.PRNGKey, optional, for randomness (default: PRNGKey(1234)).
        num_euler_steps: int, optional, number of Euler steps for integration/generation (default: 2_500).
        compute_nll: bool, optional, whether to compute and display Negative Log-Likelihood (default: True).
        final_time: float, optional, final time for trajectory plotting (default: 10.0).
        save_path: str, optional, path to save the figure (default: None).

    Returns:
        None. Displays a matplotlib figure of predicted and true trajectories for each ensemble member.
    """
    n_dim = 1  # 1D system
    time_steps = true_outputs.shape[-1]

    # tile inputs and outputs
    tiled_true_inputs = jnp.tile(true_inputs, (ensemble_size, 1, 1))
    tiled_true_outputs = jnp.tile(true_outputs, (ensemble_size, 1, 1))

    # Use model methods directly (they handle normalization internally)
    samples = model.generate_sample(
        key,
        model.params,
        tiled_true_inputs,
        num_output_dims=time_steps,
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
    
    ensemble_size = samples.shape[0]
    n_rows = math.ceil(ensemble_size / max_cols)
    n_cols = min(max_cols, ensemble_size)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if ensemble_size > 1 else [axes]

    # Create time points for plotting
    time_points = np.linspace(0, final_time, time_steps)

    for net_id in range(ensemble_size):
        ax = axes[net_id]
        for i in range(cutoff):
            traj = samples[net_id, i]  # shape: (time_steps,)
            true_traj = tiled_true_outputs[net_id, i]  # shape: (time_steps,)
            
            if i == 0:  # Only label the first trajectory to avoid legend clutter
                ax.plot(time_points, traj, c='red', alpha=0.3, label='Pred')
                ax.plot(time_points, true_traj, c='blue', alpha=0.3, label='True')
            else:
                ax.plot(time_points, traj, c='red', alpha=0.3)
                ax.plot(time_points, true_traj, c='blue', alpha=0.3)
        
        # Add horizontal line at y=0 to show bifurcation
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        
        # Mark initial conditions
        x0_vals = tiled_true_inputs[net_id, :cutoff, 0]  # initial x values for this ensemble member
        ax.scatter([0] * len(x0_vals), x0_vals, s=20, c='green', alpha=0.7, label='Initial x0')
        
        ax.set_xlabel("Time")
        ax.set_ylabel("x(t)")
        ax.grid(True, alpha=0.3)
        
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

    plt.suptitle(f"CFM: Predictive Samples from Ensemble of {ensemble_size} networks\n(Saddle-Node Bifurcation System)", fontsize=18)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()

def plot_mse_ensemble_samples_1d(model, inputs, outputs, cutoff, final_time=5.0, save_path=None):
    """
    Plot MSE baseline predictions for 1D saddle-node system.
    Shows ensemble mean and spread (uncertainty from ensemble disagreement).

    Args:
        model: Trained MSE model
        inputs: Test inputs
        outputs: Test outputs
        cutoff: Maximum number of samples to plot
        final_time: Final time for x-axis
        save_path: If provided, save the figure to this path
    """
    pred = model.apply(model.params, inputs)  # shape: (ensemble_size, batch, out_dim)

    N = pred.shape[1]
    T = pred.shape[2]
    cutoff = int(min(cutoff, N))

    pred_np = np.asarray(pred[:, :cutoff, :])
    outputs_np = np.asarray(outputs[:cutoff])
    time_points = np.linspace(0, final_time, T)

    # Compute MSE on test set
    mse_per_sample = np.mean((pred_np.mean(axis=0) - outputs_np) ** 2, axis=-1)

    ensemble_size = pred_np.shape[0]
    max_cols = 3
    n_rows = math.ceil(ensemble_size / max_cols)
    n_cols = min(max_cols, ensemble_size)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for e in range(ensemble_size):
        ax = axes_flat[e]
        for i in range(cutoff):
            ax.plot(time_points, pred_np[e, i, :], c='red', alpha=0.35)
            ax.plot(time_points, outputs_np[i, :], c='blue', alpha=0.25)
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("x(t)")

        # Compute per-ensemble MSE
        ensemble_mse = np.mean((pred_np[e] - outputs_np) ** 2)
        ax.set_title(f"Ensemble member {e}\nMSE: {ensemble_mse:.4f}")
        ax.grid(True, alpha=0.3)
        if cutoff > 0:
            ax.plot([], [], c='red', alpha=0.7, label='MSE pred')
            ax.plot([], [], c='blue', alpha=0.7, label='True')
        ax.legend(frameon=False)

    for j in range(ensemble_size, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    overall_mse = np.mean(mse_per_sample)
    plt.suptitle(f"MSE Baseline: Predictive samples from ensemble of {ensemble_size} networks\nOverall Test MSE: {overall_mse:.4f}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()

    return overall_mse


def generate_saddle_node_data(
    n_samples,
    time_steps=100,
    key=random.PRNGKey(0),
    final_time=10.0,
    dt=None,
    x0_min=-2.0,
    x0_max=2.0,
    noise_factor=0.0,              # diffusion σ (Euler–Maruyama uses σ*sqrt(dt))
    observation_noise=0.0,         # std dev of measurement noise on outputs
    input_observation_noise=0.0,   # std dev of noise added to [x0] inputs
    r_noise=0.0,                   # std dev noise added to r before sim
    clip_threshold=10.0,
    discrete_r_values=None         # discrete r values to sample from (bifurcation at r=0)
):
    """
    Generates synthetic trajectories for the saddle-node system: dx/dt = r + x^2
    
    Uses discrete r values from a predefined set for consistent bifurcation behavior.
    The bifurcation occurs at r=0:
      - r < 0: Two fixed points at x* = ±sqrt(-r) (stable at -sqrt(-r), unstable at +sqrt(-r))
      - r = 0: Saddle-node bifurcation point (fixed points collide at x* = 0)
      - r > 0: No fixed points, trajectories diverge to infinity

    Args:
        n_samples: Number of trajectories to generate
        time_steps: Number of time points in each trajectory
        key: Random key for reproducibility
        final_time: Final time for integration
        dt: Time step (auto-calculated if None)
        x0_min, x0_max: Initial condition bounds
        noise_factor: Diffusion coefficient for SDE
        observation_noise: Measurement noise on outputs
        input_observation_noise: Noise on input measurements
        r_noise: Noise added to r parameter
        clip_threshold: Trajectory clipping threshold
        discrete_r_values: Array of discrete r values to sample from. 
                          Default: [-1.5, 1.5, 0, -0.5, 0.5]

    Returns:
        inputs:  (n_samples, 1)  -> [x0]
        outputs: (n_samples, time_steps)  trajectories including t=0
    """
    # Choose dt so we land exactly on 'time_steps' samples (including t=0)
    if dt is None:
        dt = float(final_time) / float(time_steps - 1)
    total_time_steps = int(round(final_time / dt)) + 1

    # Sample parameters and initial states
    key, k_r = random.split(key)
    # Use discrete r values for consistent bifurcation behavior
    if discrete_r_values is None:
        discrete_r_values = jnp.array([-1.5, 1.5, 0, -0.5, 0.5])
    else:
        discrete_r_values = jnp.asarray(discrete_r_values)
    r_values = random.choice(k_r, discrete_r_values, shape=(n_samples,))

    key, k_x0 = random.split(key)
    x0_values = random.uniform(k_x0, (n_samples,), minval=x0_min, maxval=x0_max)

    if r_noise > 0.0:
        key, k_rn = random.split(key)
        r_values = r_values + random.normal(k_rn, (n_samples,)) * r_noise

    # Allocate & set initial conditions
    trajectories = jnp.zeros((n_samples, total_time_steps), dtype=jnp.float32)
    trajectories = trajectories.at[:, 0].set(x0_values.astype(jnp.float32))

    cs = x0_values.astype(jnp.float32)
    r_values = r_values.astype(jnp.float32)
    dt32 = jnp.float32(dt)
    clip_thr32 = jnp.float32(clip_threshold)

    keys = random.split(key, total_time_steps - 1)

    @jit
    def step_fn(carry, key):
        """Single step function for scan - computes next state and stores in trajectory."""
        cs, trajectories, t = carry
        
        # Saddle-node equation: dx/dt = r + x^2
        drift = r_values + cs * cs
        noise = random.normal(key, (n_samples,)).astype(jnp.float32) \
                * (noise_factor * jnp.sqrt(dt32))
        new_cs = cs + dt32 * drift + noise
        new_cs = jnp.clip(new_cs, -clip_thr32, clip_thr32)
        
        t_int = t.astype(jnp.int32)
        trajectories = trajectories.at[:, t_int].set(new_cs)
        
        return (new_cs, trajectories, t + 1), None

    initial_carry = (cs, trajectories, jnp.int32(1))
    (cs, trajectories, _), _ = lax.scan(step_fn, initial_carry, keys)

    @partial(jit, static_argnames=('total_time_steps', 'time_steps', 'n_samples'))
    def process_trajectories(trajectories, total_time_steps, time_steps, n_samples,
                            observation_noise, key_obs):
        """JIT-compiled function for subsampling and adding observation noise."""
        # Subsample to exactly 'time_steps' samples via linspace indices
        idx = jnp.linspace(0, total_time_steps - 1, time_steps, dtype=jnp.int32)
        subsampled = trajectories[:, idx]
        
        # Add observation noise if specified
        obs_noise = random.normal(key_obs, subsampled.shape) * observation_noise
        subsampled = subsampled + obs_noise.astype(subsampled.dtype)
        
        return subsampled

    @jit
    def process_inputs(x0_values, input_observation_noise, key_in):
        """JIT-compiled function for building inputs and adding input noise."""
        inputs = x0_values.reshape(-1, 1).astype(jnp.float32)
        input_noise = random.normal(key_in, inputs.shape) * input_observation_noise
        inputs = inputs + input_noise.astype(inputs.dtype)
        return inputs

    key, k_obs = random.split(key)
    subsampled_trajectories = process_trajectories(
        trajectories, total_time_steps, time_steps, n_samples,
        observation_noise, k_obs
    )

    key, k_in = random.split(key)
    inputs = process_inputs(x0_values, input_observation_noise, k_in)

    print(f"Generated {n_samples} trajectories with dt={float(dt):.6f}, total_time_steps={int(total_time_steps)}, subsampled_steps={int(subsampled_trajectories.shape[1])}")
    return inputs, subsampled_trajectories

