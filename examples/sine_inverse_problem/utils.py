# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, value_and_grad

from flax import linen as nn
from flax.core import FrozenDict
import optax

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as onp
import copy
import pickle
import pandas as pd

from typing import Tuple, Optional, Any

from jaxmix.trainers import mdn_loss_func

def generate_data(
    n_samples: int, 
    key: jax.Array = random.PRNGKey(0)
) -> Tuple[jax.Array, jax.Array]:
    """
    Generate synthetic training data for the sine inverse problem.

    Args:
        n_samples: Number of data samples to generate.
        key: JAX PRNGKey for reproducibility.

    Returns:
        Tuple of:
          - x_data: Shape (n_samples, 1) array of uniformly sampled inputs in [-1.5, 1.5].
          - y_data: Shape (n_samples, 1) array of outputs (noisy, nonlinear function of input).
    """
    key, subkey = random.split(key)
    epsilon = 0.2 * random.normal(subkey, shape=(n_samples, 1))
    key, subkey = random.split(key)
    x_data = random.uniform(subkey, shape=(n_samples, 1), minval=-1.5, maxval=1.5)
    y_data = 0.7*jnp.sin(5*x_data) + 0.5*x_data + epsilon
    return x_data, y_data



#########################################################
############### Plotting Utilities ######################
#########################################################

def plot_mdn_mixture_elements(
    model: Any, 
    true_inputs: jax.Array, 
    true_outputs: jax.Array, 
    net_id: Optional[int] = None, 
    num_cols: int = 3
) -> None:
    """
    Plot the mean, standard deviation, and probability of each mixture component in a Mixture Density Network (MDN).

    Args:
        model: The trained MDN model.
        true_inputs: Array of true input values, shape (N, 1).
        true_outputs: Array of true output values, shape (N, 1).
        net_id: If using an ensemble, index of model in ensemble to plot; if None, expects non-ensemble.
        num_cols: Number of columns for subplot grid.

    Returns:
        None. Displays matplotlib plots.
    """
    pred = model.apply(model.params, true_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    # If using an ensemble of networks, we need to specify the net_id
    if net_id is None:
        if len(mixture_logit_weights.shape) == 4:
            raise ValueError("If using an ensemble of networks, you must specify the net_id")
    else:
        mixture_logit_weights = mixture_logit_weights[net_id]
        mixture_means = mixture_means[net_id]
        mixture_variances = mixture_variances[net_id]

    probabilites = nn.softmax(mixture_logit_weights, axis=-2)
    num_mixtures = probabilites.shape[-2]
    num_rows = (num_mixtures + num_cols - 1) // num_cols  # Ceiling division
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for i in range(num_mixtures):
        # Order inputs for plotting
        ordered_indices = jnp.argsort(true_inputs.squeeze())
        mean_vals = mixture_means[ordered_indices, i].squeeze()
        std_vals = jnp.sqrt(mixture_variances[ordered_indices, i].squeeze())
        x_vals = true_inputs[ordered_indices].squeeze()
        y_vals = true_outputs[ordered_indices].squeeze()
        prob_vals = probabilites[ordered_indices, i].squeeze()

        # Set up main axis and a minor axis below sharing x
        ax_main = axes[i]
        divider = make_axes_locatable(ax_main)
        ax_minor = divider.append_axes("bottom", size="25%", pad=0.1, sharex=ax_main)

        # Plot in the main axis
        ax_main.scatter(x_vals, mean_vals, s=1, label='Mean')
        ax_main.scatter(x_vals, y_vals, s=1, alpha=0.1, c='black', label='True')
        ax_main.fill_between(
            x_vals,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color='orange',
            alpha=0.2,
            label='±1 Std'
        )
        ax_main.legend()
        title_str = f"Mixture Element {i}\nAverage weight across dataset: {jnp.mean(probabilites[:, i]) :.3f}"
        ax_main.set_title(title_str)
        
        # Plot in the minor axis (probabilities)
        ax_minor.scatter(x_vals, prob_vals, s=1, color='blue', alpha=0.5)
        ax_minor.set_ylabel('Mixture\nProb.', fontsize=8)
        ax_minor.set_ylim(0, 1)
        ax_minor.tick_params(axis='y', labelsize=8)
        ax_minor.tick_params(axis='x', labelsize=8)
        # Remove x-labels on main plot to save space & prevent label tick repetition
        plt.setp(ax_main.get_xticklabels(), visible=False)
        # Optionally, set grid for minor axis for easier reading
        ax_minor.grid(True, axis='y', alpha=0.1)

    # Add super title
    if net_id is not None:
        fig.suptitle(f"MDN Mixture Elements for Network {net_id}", fontsize=32)
    else:
        fig.suptitle(f"MDN Mixture Elements", fontsize=32)

    # Hide any unused subplots
    for j in range(num_mixtures, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_mdn_samples(
    model: Any, 
    true_inputs: jax.Array, 
    true_outputs: jax.Array, 
    num_cols: int = 3, 
    plot_cutoff: Optional[int] = None,
    return_nll: bool = False,
) -> Optional[jax.Array]:
    """
    Plot predictive samples generated by the MDN and compare with true outputs for each ensemble network.

    Args:
        model: The trained MDN model.
        true_inputs: Array of true input values, shape (N, 1).
        true_outputs: Array of true output values, shape (N, 1).
        num_cols: Number of columns for subplot grid.
        plot_cutoff: Number of samples from true_inputs/outputs to plot. If None, plot all.

    Returns:
        None. Displays matplotlib plots.
    """
    if plot_cutoff is None:
        plot_cutoff = true_inputs.shape[0]
    pred = model.apply(model.params, true_inputs)
    mixture_logit_weights, mixture_means, mixture_variances = pred
    samples = model.sample_from_mixture(
        random.PRNGKey(23), 
        mixture_logit_weights[:, :plot_cutoff], 
        mixture_means[:, :plot_cutoff], 
        mixture_variances[:, :plot_cutoff]
    )
    #per_element_nll = model.per_element_loss(pred, true_outputs)
    per_element_nll = mdn_loss_func(mixture_logit_weights, mixture_means, mixture_variances, true_outputs)
    # Determine number of networks in ensemble
    ensemble_size = mixture_logit_weights.shape[0]

    num_rows = (ensemble_size + num_cols - 1) // num_cols  # Ceiling division
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()

    for net_id in range(ensemble_size):
        axes[net_id].scatter(true_inputs[:plot_cutoff].squeeze(), samples[net_id].squeeze(), s=1, label='Predicted')
        axes[net_id].scatter(true_inputs[:plot_cutoff].squeeze(), true_outputs[:plot_cutoff].squeeze(), s=1, alpha=0.5, label='True')
        axes[net_id].set_title(
            f'Network {net_id}\nMean NLL: {per_element_nll[net_id].mean():.3f}\nMedian NLL: {jnp.median(per_element_nll[net_id]):.3f}'
        )
        axes[net_id].legend()

    # Hide any unused subplots
    for j in range(ensemble_size, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"MDN Samples from Ensemble of {ensemble_size} networks\n", fontsize=32)

    plt.tight_layout()
    plt.show()

    if return_nll:
        return per_element_nll


def plot_flow_matching_samples(
    model: Any, 
    subkey: jax.Array, 
    true_inputs: jax.Array, 
    true_outputs: jax.Array, 
    ensemble_size: int, 
    num_cols: int = 3, 
    num_steps: int = 10_000,
    compute_nll: bool = True,
    return_nll: bool = False,
) -> Optional[jax.Array]:
    """
    Generate and plot flow-matching samples for an ensemble of networks, optionally displaying per-element NLL.

    Args:
        model: Trained flow-based model.
        subkey: JAX PRNGKey for sample generation.
        true_inputs: True inputs array, shape (N, 1).
        true_outputs: True outputs array, shape (N, 1).
        ensemble_size: Number of networks in the ensemble.
        num_cols: Number of subplot columns.
        num_steps: Number of integration steps for sample generation.
        compute_nll: Whether to compute and show the NLL per element/network.

    Returns:
        None. Displays matplotlib plots.
    """
    tiled_true_inputs = jnp.tile(true_inputs, (ensemble_size, 1, 1))
    tiled_true_outputs = jnp.tile(true_outputs, (ensemble_size, 1, 1))
    samples = model.generate_sample(
        subkey,
        model.params,
        tiled_true_inputs,
        num_output_dims=1,
        num_steps=num_steps,
        apply_kwargs=FrozenDict({'tile_inputs': False})
    )

    if compute_nll:
        nll_per_element = model.compute_nll(
            model.params,
            tiled_true_inputs,
            tiled_true_outputs,
            num_steps=num_steps,
            apply_kwargs=FrozenDict({'tile_inputs': False},)
        )

    num_rows = (ensemble_size + num_cols - 1) // num_cols  # Ceiling division
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()

    for net_id in range(ensemble_size):
        axes[net_id].scatter(true_inputs.squeeze(), samples[net_id].squeeze(), s=1, label='Predicted')
        axes[net_id].scatter(true_inputs.squeeze(), true_outputs.squeeze(), s=1, alpha=0.5, label='True')
        axes[net_id].set_xlabel("Input")
        axes[net_id].set_ylabel("Output")
        axes[net_id].legend()
        if compute_nll:
            axes[net_id].set_title(
                f"Network {net_id}\nMean NLL: {nll_per_element[net_id].mean():.3f}\nMedian NLL: {jnp.median(nll_per_element[net_id]):.3f}"
            )
        else:
            axes[net_id].set_title(f"Generated Samples for Network {net_id}")

    # Hide any unused subplots
    for j in range(ensemble_size, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Flow Matching Samples from Ensemble of {ensemble_size} networks\n", fontsize=32)

    plt.tight_layout()
    plt.show()

    if return_nll:
        assert compute_nll, "compute_nll must be True if return_nll is True"
        return nll_per_element