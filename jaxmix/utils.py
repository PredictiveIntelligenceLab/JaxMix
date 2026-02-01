import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp
from jax import vmap, jacfwd

from flax.core.frozen_dict import FrozenDict

import numpy as onp

from typing import Optional
import matplotlib.pyplot as plt

from typing import Callable


empty_frozen_dict = FrozenDict({})
identity = lambda x: x

JaxArray = type(jnp.arange(5))
def create_mask_by_name(params, freeze_layer_name):
    """
    Traverse the parameter tree and create a mask where parameters
    under `freeze_layer_name` are set to False (frozen), and others to True (trainable).
    """
    mask = {}

    for layer_name, layer_content in params.items():
        if freeze_layer_name in layer_name: # everything below this node should be frozen
            mask[layer_name] = jax.tree.map(lambda x : False, layer_content)
        
        else:
            if type(layer_content) != JaxArray: # not a leaf yet. Recursively call this function again
                mask[layer_name] = create_mask_by_name(layer_content, freeze_layer_name)
            else: # reached a leaf
                # Keep all parameters in this layer trainable
                mask[layer_name] = True

    return mask

def stable_logsumexp(x : jnp.ndarray, axis : int = -1, keepdims : bool = False) -> jnp.ndarray:
    """
    Compute the log-sum-exp of an array in a numerically stable way.

    Args:
        x (jnp.ndarray): Input array.
        axis (int, optional): Axis or axes over which the sum is taken. Default is -1.
        keepdims (bool, optional): If True, retains reduced dimensions with length 1. Default is False.

    Returns:
        jnp.ndarray: The result of log(sum(exp(x))) computed in a numerically stable manner.
    """
    if keepdims:
        x_max = jnp.max(x, axis=axis, keepdims=True)
        return logsumexp(x-x_max, axis=axis, keepdims=True) + x_max
    else:
        x_max = jnp.max(x, axis=axis, keepdims=True)
        return logsumexp(x-x_max, axis=axis, keepdims=False) + x_max.squeeze(axis=axis)

logSqrtTwoPI = jnp.log(2.0 * jnp.pi)/2 # log(2*pi)/2 = log(sqrt(2*pi))

def log_normal_pdf(y, mean, logstd):
    '''
    Compute the log probability density function (PDF) of a multivariate normal distribution.
    It is assumed that the covariance matrix is diagonal, with the diagonal elements specified by the log standard deviations.

    Args:
        y (jnp.ndarray): The observed values. Shape (..., num_output_dims).
        mean (jnp.ndarray): The mean of the normal distribution. Shape (..., num_output_dims).
        logstd (jnp.ndarray): The log of the standard deviation of the normal distribution. Shape (..., num_output_dims).

    Returns:
        jnp.ndarray: The log probability density function (PDF) of the normal distribution. Shape (..., 1).
    '''
    num_output_dims = y.shape[-1]
    scaled_diff = abs(y - mean) / jnp.exp(logstd) # shape (..., num_output_dims)
    return -(scaled_diff**2).sum(-1, keepdims=True)/2 - logstd.sum(-1, keepdims=True) - logSqrtTwoPI*num_output_dims # shape (..., 1)

def split_data(
    key : JaxArray,
    inputs : JaxArray,
    outputs : JaxArray,
    n_train : int,
    n_val : Optional[int] = None,
    weights : Optional[JaxArray] = None,
):
    """
    Splits the data into train and validation sets. Each set is a tuple of
    (inputs, outputs, weights), where inputs is a JaxArray of shape (n_samples, n_features), outputs
    is a JaxArray of shape (n_samples, n_outputs), and weights is a JaxArray of shape (n_samples,).

    Args:
        key: PRNG key
        inputs: inputs of shape (n_data, n_features)
        outputs: outputs of shape (n_data, n_outputs)
        weights: weights of shape (n_data,)
        n_train: number of training samples
        n_val: number of validation samples. If None, will be set to n_data - n_train.

    Returns:
        train_data: train data tuple (inputs, outputs, weights)
        val_data: validation data tuple (inputs, outputs, weights)
    """
    n_data = inputs.shape[0]
    if n_val is None:
        n_val = n_data - n_train
    assert n_data >= n_train + n_val, "Not enough samples to split"

    # if weights are not provided, use uniform weights
    if weights is None:
        weights = jnp.ones((n_data, 1))

    train_idx = random.choice(key, jnp.arange(n_data), shape=(n_train,), replace=False)
    collected_idx = train_idx
    remaining_idx = jnp.setdiff1d(jnp.arange(n_data), train_idx)
    val_idx = random.choice(key, remaining_idx, shape=(n_val,), replace=False)
    collected_idx = jnp.concatenate([collected_idx, val_idx])
    remaining_idx = jnp.setdiff1d(remaining_idx, val_idx)

    # train set
    train_inputs = inputs[train_idx]
    train_outputs = outputs[train_idx]
    train_weights = weights[train_idx]

    # validation set
    val_inputs = inputs[val_idx]
    val_outputs = outputs[val_idx]
    val_weights = weights[val_idx]

    # aggregate values in data tuples
    train_data = (train_inputs, train_outputs, train_weights)
    val_data = (val_inputs, val_outputs, val_weights)
    
    return train_data, val_data

    
def get_ensemble_cfm_div(func, x, y_t, t, ensemble_size):
    def fw_fn_single(i,x_single, y_t_single, t_single):
        # x_single should be shape (in_dim,)
        # y_t_single should be shape (out_dim,)
        # t_single should be shape (1,)
        # computes the forward pass of netowrk i
        pred = func(x_single[None, None, :], y_t_single[None, None, :], t_single[None, None, :])
        # pred should be shape (ensemble_size, 1, out_dim)
        return pred[i,0] # shape (out_dim, )

    grad_fn_single = jacfwd(fw_fn_single, argnums=2)

    def all_grad_fn(x, y_t, t, ensemble_size):
        # x should be shape (num_ensemble, batch_size, in_dim)
        # y_t should be shape (num_ensemble, batch_size, in_dim)
        # t should be shape (num_ensemble, batch_size, 1)
        grads = []
        for i in range(ensemble_size):
            grads.append(vmap(grad_fn_single, in_axes=(None, 0, 0, 0))(i, x[i], y_t[i], t[i]))
        return jnp.stack(grads).squeeze(-3)

    

    # jacobian below will be shape (ensemble_size, batch_size, out_dim, out_dim)
    jacobian = all_grad_fn(x, y_t, t, ensemble_size=ensemble_size)
    
    div = jnp.trace(jacobian, axis1=-1, axis2=-2) # shape (ensemble_size, batch_size)

    return div[..., None] # shape (ensemble_size, batch_size, 1)

# Functions to help plotting
def plot_logs(loss_log, grad_norm_log, test_rel_l2_log=None, window=None, steps_per_check=100, log_loss=False):
    if test_rel_l2_log is None:
        plt.figure(figsize=(12, 4))

        # Plotting losses
        plt.subplot(121)
        if window is None:
            plt.plot(steps_per_check*jnp.arange(len(loss_log)), loss_log)
        else:
            assert type(window) is int , f'window must be an integer or None, not {type(window)}'
            plt.plot(steps_per_check*jnp.arange(len(loss_log) - window), [onp.mean(loss_log[i:i+window]) for i in range(len(loss_log) - window)])
        if log_loss:
            plt.yscale('log')
        plt.title('Loss through iterations')

        # Plotting gradient norms
        plt.subplot(122)
        if window is None:
            plt.plot(steps_per_check*jnp.arange(len(grad_norm_log)), grad_norm_log)
        else:
            assert type(window) is int , f'window must be an integer or None, not {type(window)}'
            plt.plot(steps_per_check*jnp.arange(len(grad_norm_log) - window), [onp.mean(grad_norm_log[i:i+window]) for i in range(len(grad_norm_log) - window)])
        plt.yscale('log')
        plt.title('Global gradient norm through iterations')
    else:
        plt.figure(figsize=(20, 6))

        # Plotting losses
        plt.subplot(131)
        if window is None:
            plt.plot(steps_per_check*jnp.arange(len(loss_log)), loss_log)
        else:
            assert type(window) is int , f'window must be an integer or None, not {type(window)}'
            plt.plot(steps_per_check*jnp.arange(len(loss_log) - window), [onp.mean(loss_log[i:i+window]) for i in range(len(loss_log) - window)])
        plt.yscale('log')
        plt.title('Loss through iterations')

        # Plotting test relative L2 error
        plt.subplot(132)
        if window is None:
            plt.plot(steps_per_check*jnp.arange(len(test_rel_l2_log)), test_rel_l2_log)
        else:
            assert type(window) is int , f'window must be an integer or None, not {type(window)}'
            plt.plot(steps_per_check*jnp.arange(len(test_rel_l2_log) - window), [onp.mean(test_rel_l2_log[i:i+window]) for i in range(len(test_rel_l2_log) - window)])
        plt.yscale('log')
        plt.title('Test relative L2 error through iterations')

        # Plotting gradient norms
        plt.subplot(133)
        if window is None:
            plt.plot(steps_per_check*jnp.arange(len(grad_norm_log)), grad_norm_log)
        else:
            assert type(window) is int , f'window must be an integer or None, not {type(window)}'
            plt.plot(steps_per_check*jnp.arange(len(grad_norm_log) - window), [onp.mean(grad_norm_log[i:i+window]) for i in range(len(grad_norm_log) - window)])
        plt.yscale('log')
        plt.title('Global gradient norm through iterations')

    plt.show()

def plot_parity_plots(y_true, y_pred, n_cols=5, title=None):
    """
    Plots parity plots for each output dimension.
    y_true: array of shape (n_samples, n_outputs)
    y_pred: array of shape (n_samples, n_outputs)
    n_cols: number of columns in each row (default: 5)
    title: optional string, overarching title for the entire figure
    """
    import math
    n_outputs = y_true.shape[1]
    n_cols = min(n_outputs, n_cols)
    n_rows = math.ceil(n_outputs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=False, sharey=False)
    axes = onp.array(axes).reshape(-1)  # flatten for easy indexing

    for i in range(n_outputs):
        ax = axes[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ax.scatter(yt, yp, s=2, alpha=0.5)
        min_val = min(yt.min(), yp.min())
        max_val = max(yt.max(), yp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        ax.set_xlabel(f"True Output {i+1}")
        ax.set_ylabel(f"Predicted Output {i+1}")
        ax.set_title(f"Parity Plot Output {i+1}")

    # Hide any unused axes
    for j in range(n_outputs, n_rows * n_cols):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title, fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()


def apply_output_scaling(y, mu_y, sig_y, is_nll=False, log_var_min=-1e1, log_var_max=1e1):
    """
    Applies output scaling to the data.
    y: array of shape (..., n_outputs) if is_nll is False, or (..., 2*n_outputs) if is_nll is True
    mu_y: array of shape (n_outputs,)
    sig_y: array of shape (n_outputs,)
    is_nll: boolean, whether the predictions are in the Gaussian NLL format, i.e. shape (..., 2*n_outputs)
    """
    if is_nll:
        # under Gaussian NLL format, the predictions are in the form of (mu, log_var)
        n_outputs = y.shape[-1]//2
        assert y.shape[-1] == 2*n_outputs, f'y should have shape (..., 2*n_outputs), but got {y.shape}'
        pred_mu = mu_y + (sig_y+1e-6)*y[...,:n_outputs] # shape (..., n_outputs)
        pred_log_var = y[...,n_outputs:] 
        if (log_var_min is not None) or (log_var_max is not None): # optionally limit the log_var to be at most limit_log_var
            pred_log_var = jnp.clip(pred_log_var, a_min=log_var_min, a_max=log_var_max)
        pred_log_var = pred_log_var + jnp.log(sig_y) # shape (..., n_outputs)
        return jnp.concatenate([pred_mu, pred_log_var], axis=-1) # shape (..., 2*n_outputs)
    else:
        return mu_y + (sig_y+1e-6)*y # shape (..., n_outputs)


def sample_from_gaussian_mixture(
        key: jax.random.PRNGKey,
        mixture_logit_weights: jnp.ndarray,
        mixture_means: jnp.ndarray,
        mixture_variances: jnp.ndarray,
        restrict_rare_event_rate: Optional[float] = None,
        truncated_normal_std_limit: Optional[float] = None,
    ) -> jnp.ndarray:
        """
        Sample from a mixture of Gaussians defined by the provided logits, means, and variances.

        Args:
            key: jax.random.PRNGKey
            mixture_logit_weights: Array of shape (..., num_mixtures, 1)
            mixture_means: Array of shape (..., num_mixtures, num_output_dims)
            mixture_variances: Array of shape (..., num_mixtures, num_output_dims)
            restrict_rare_event_rate: (Optional) Restrict the rare event rate to the specified value.
                If None, no restriction is applied. Otherwise, mixtures with a probability less than 
                the specified value are not sampled.
            truncated_normal_std_limit: (Optional) Limit the standard deviation of the truncated normal distribution to the specified value.
                If None, no limit is applied. Otherwise, the standard deviation is limited to the specified value.

        Returns:
            samples: Array of shape (..., num_output_dims)
        """
        if restrict_rare_event_rate is not None:
            assert restrict_rare_event_rate > 0 and restrict_rare_event_rate < 1, (
                f'restrict_rare_event_rate must be between 0 and 1, but got {restrict_rare_event_rate}'
            )
            # restrict the rare event rate to the specified value
            logsumexp_logits = stable_logsumexp(mixture_logit_weights, axis=-2, keepdims=True) # shape (..., 1, 1)
            logit_cutoffs = jnp.log(restrict_rare_event_rate) + logsumexp_logits # shape (..., 1, 1)
            mixture_logit_weights = jnp.where(mixture_logit_weights < logit_cutoffs, -jnp.inf, mixture_logit_weights)
        key, subkey = random.split(key)
        mixture_idx = random.categorical(subkey, mixture_logit_weights, axis=-2) # shape (..., 1)
        # add singleton dimension to make it compatible with take_along_axis
        mixture_idx = mixture_idx[..., None]  # (..., 1, 1)

        # Gather means and variances for the sampled mixture components
        selected_means = jnp.take_along_axis(mixture_means, mixture_idx, axis=-2)    # (..., num_output_dims)
        selected_variances = jnp.take_along_axis(mixture_variances, mixture_idx, axis=-2)  # (..., num_output_dims)

        key, subkey = random.split(key)
        if truncated_normal_std_limit is not None:
            # make sure the truncated normal std limit is positive
            assert truncated_normal_std_limit > 0, (
                f'truncated_normal_std_limit must be positive, but got {truncated_normal_std_limit}'
            )
            # sample from truncated normal distribution
            noise = random.truncated_normal(
                subkey,
                shape=selected_means.shape,
                lower=-truncated_normal_std_limit,
                upper=truncated_normal_std_limit,
            )
        else:
            # sample from normal distribution (no truncation)
            noise = random.normal(subkey, shape=selected_means.shape)

        return selected_means + jnp.sqrt(selected_variances) * noise



#################################################
### Functions to help with synthetic datasets ###
#################################################

def create_test_fn(in_dim, out_dim, seed=0, low_rank=None):
    key = random.PRNGKey(seed)

    # create clean test function (no aleatoric noise)
    if low_rank is None:
        key, subkey = random.split(key)
        w_test = random.normal(subkey, (in_dim, 2*out_dim))
        #w_test = jnp.clip(w_test, a_min=-3, a_max=3)
    else:
        assert type(low_rank) == int
        print(f"Creating low-rank test function with rank {low_rank}")
        key, subkey = random.split(key)
        w_1 = random.normal(subkey, (in_dim, low_rank))
        key, subkey = random.split(key)
        w_2 = random.normal(subkey, (low_rank, 2*out_dim))
        w_test = w_1 @ w_2
    key, subkey = random.split(key)
    #b_test = random.normal(subkey, (2*out_dim,))
    b_test = random.uniform(subkey, (2*out_dim,), minval=-jnp.pi, maxval=jnp.pi)
    def test_fn(x, return_weights=False):
        # x is shape (batch_dim, in_dim)
        features = x**2 @ w_test + b_test
        if return_weights:
            return jnp.sin(features[:, :out_dim])+jnp.cos(features[:, out_dim:]), w_test
        else:
            return jnp.sin(features[:, :out_dim])+jnp.cos(features[:, out_dim:]) # shape (batch_dim, out_dim)
    
    def aleatoric_level_fn(x, scale=0.1):
        # x is shape (batch_dim, in_dim)
        raw_level = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
        return abs(scale*raw_level) # shape (batch_dim, 1)
        #return scale*(1-jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))) # shape (batch_dim, 1)
    return test_fn, aleatoric_level_fn
    
    
def create_synthetic_dataset(n_data, in_dim, out_dim, seed=0, aleatoric_scale=0.1):
    test_fn, aleatoric_level_fn = create_test_fn(in_dim, out_dim, seed=seed)
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    x_data = random.uniform(subkey, (n_data, in_dim), minval=-1, maxval=1)
    key, subkey = random.split(key)
    aleatoric_noise = random.normal(subkey, shape=(n_data, out_dim))*aleatoric_level_fn(x_data, scale=aleatoric_scale)
    y_data = test_fn(x_data) + aleatoric_noise
    return x_data, y_data