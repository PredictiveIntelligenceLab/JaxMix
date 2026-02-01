# Basic Library Imports
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit

from flax import linen as nn
from flax.core import FrozenDict

from typing import Any, Callable, Sequence, Tuple, Union, Optional


PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]
identity = lambda x : x
empty_frozen_dict = FrozenDict({})

#####################################################################
######################## Basic Architectures ########################
#####################################################################

#############
#### MLP ####
#############

class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP).

    Args:
        features: Sequence of integers specifying the number of units in each layer.
        activation: Activation function for hidden layers (default: gelu).
        output_activation: Activation function for the output layer (default: identity).
        precision: Numerical precision for Dense layers (default: None).
    """
    features: Sequence[int]
    activation: Callable = nn.gelu
    output_activation: Callable = identity
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the MLP to the input.

        Args:
            x: Input array of shape (..., input_dim).

        Returns:
            Output array after MLP forward pass.
        """
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat, precision=self.precision)(x))
        x = nn.Dense(self.features[-1], precision=self.precision)(x)
        return self.output_activation(x)


########################################################
#### Mixture Density Network (MDN) Architecture ########
########################################################

class MDN(nn.Module):
    """
    Mixture Density Network (MDN) architecture.

    Args:
        num_mixtures: Number of mixture components.
        num_output_dims: Number of output dimensions for each mixture component.
        backbone: Backbone network to process input features.
        mean_output_activation: Activation function for mixture means (default: identity).
        variance_output_activation: Activation function for mixture variances (ensures positivity, default: softplus).
        precision: Numerical precision for Dense layers (default: None).
        ensemble_size: Number of ensemble members. If None, assumes backbone is not an ensemble.
    """
    num_mixtures: int
    num_output_dims: int
    backbone: nn.Module
    mean_output_activation: Callable = identity
    variance_output_activation: Callable = nn.softplus
    precision: PrecisionLike = None
    ensemble_size: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply the MDN to the input, producing mixture parameters.

        Args:
            x: Input array of shape (batch_size, input_dim).
            kwargs: Optional dict of extra arguments for the backbone.

        Returns:
            Tuple containing:
                - mixture_logit_weights: (batch_size, num_mixtures, 1)
                - mixture_means: (batch_size, num_mixtures, num_output_dims)
                - mixture_variances: (batch_size, num_mixtures, num_output_dims)
                NOTE: if using ensemble, each output will have an extra axis for the ensemble member (ensemble_size, ...)
        """
        backbone_kwargs = kwargs.get('backbone_kwargs', empty_frozen_dict)
        # x is shape (batch_size, input_dim)
        x = self.backbone(x, **backbone_kwargs)  # shape (batch_size, hidden_dim)

        if self.ensemble_size is not None:
            single_last_layer = nn.DenseGeneral(
                (self.num_mixtures, 1 + 2 * self.num_output_dims),
                axis=-1,
                precision=self.precision
            )
            last_layer = Ensemble(single_last_layer, self.ensemble_size)
            # compute the mixture weights, means, and variances
            flattened_output = last_layer(x, tile_inputs=False)  # shape (ensemble_size, batch_size, num_mixtures, 1 + 2*num_output_dims)
        else:
            last_layer = nn.DenseGeneral(
                (self.num_mixtures, 1 + 2 * self.num_output_dims),
                axis=-1,
                precision=self.precision
            )
            # compute the mixture weights, means, and variances
            flattened_output = last_layer(x)  # shape (batch_size, num_mixtures, 1 + 2*num_output_dims)

        # split the output into the mixture weights, means, and variances
        mixture_logit_weights = flattened_output[..., 0][..., None]  # (batch_size, num_mixtures, 1)
        mixture_means = flattened_output[..., 1 : self.num_output_dims + 1]  # (batch_size, num_mixtures, num_output_dims)
        mixture_variances = flattened_output[..., -self.num_output_dims :]    # (batch_size, num_mixtures, num_output_dims)

        # apply the output activation to the means and variances
        mixture_means = self.mean_output_activation(mixture_means)              # (batch_size, num_mixtures, num_output_dims)
        mixture_variances = self.variance_output_activation(mixture_variances)  # (batch_size, num_mixtures, num_output_dims)

        return mixture_logit_weights, mixture_means, mixture_variances


###################
#### Ensembles ####
###################

class Ensemble(nn.Module):
    """
    Wrapper module for constructing ensembles of a base architecture.

    Args:
        arch: The base architecture/module to ensemble.
        ensemble_size: Number of ensemble members.
        num_inputs: Number of input arguments the architecture expects (default: 1).
    """
    arch: nn.Module
    ensemble_size: int
    num_inputs: int = 1

    @nn.compact
    def __call__(
        self, 
        *args: jnp.ndarray,
        tile_inputs: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Apply the ensemble of models to the input(s).

        Args:
            *args: Input arrays. Number of inputs should match num_inputs.
                   Shape of each input depends on tile_inputs:
                      - If tile_inputs=True, shape is (batch_size, ...), then tiles to (ensemble_size, batch_size, ...)
                      - If tile_inputs=False, expected shape is (ensemble_size, batch_size, ...)
            tile_inputs: If True, inputs are tiled for each ensemble member.
            kwargs: Optional dict of extra arguments for the architecture.

        Returns:
            Stacked outputs of the ensemble, shape (ensemble_size, batch_size, ...)
        """
        return self._apply_ensemble('__call__', *args, tile_inputs=tile_inputs, **kwargs)
    
    def _apply_ensemble(
        self,
        method_name: str,
        *args: jnp.ndarray,
        tile_inputs: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Apply an ensemble of a specific method from the base architecture.

        Args:
            method_name: Name of the method to call on each ensemble member.
            *args: Input arrays. For __call__, number should match num_inputs.
            tile_inputs: If True, inputs are tiled for each ensemble member.
            kwargs: Optional dict of extra arguments for the method.

        Returns:
            Stacked outputs of the ensemble.
        """
        # For __call__, validate number of inputs
        if method_name == '__call__' and len(args) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs for __call__, but got {len(args)}")
        
        # Determine the actual number of arguments for this method
        num_args = len(args)
        
        # Create vmap function that handles variable number of inputs
        if num_args == 0:
            # No arguments (rare, but handle it)
            ensemble = nn.vmap(
                lambda mdl: getattr(mdl, method_name)(**kwargs),
                in_axes=0,
                out_axes=0,
                variable_axes={'params': 0},
                split_rngs={'params': True}
            )
        elif num_args == 1:
            ensemble = nn.vmap(
                lambda mdl, x: getattr(mdl, method_name)(x, **kwargs),
                in_axes=0,
                out_axes=0,
                variable_axes={'params': 0},
                split_rngs={'params': True}
            )
        else:
            # For multiple inputs, we need to specify in_axes for each input
            ensemble = nn.vmap(
                lambda mdl, *xs: getattr(mdl, method_name)(*xs, **kwargs),
                in_axes=(0,) + (0,) * num_args,  # 0 for mdl, then 0 for each input
                out_axes=0,
                variable_axes={'params': 0},
                split_rngs={'params': True}
            )
        
        if num_args == 0:
            # No arguments to tile
            outputs = ensemble(self.arch)
        elif tile_inputs:
            # Tile each input from (batch_size, ...) to (ensemble_size, batch_size, ...)
            args_tiled = tuple(
                jnp.broadcast_to(arg, (self.ensemble_size, *arg.shape)) if hasattr(arg, 'shape')
                else jnp.broadcast_to(arg, (self.ensemble_size,))
                for arg in args
            )
            outputs = ensemble(self.arch, *args_tiled)
        else:
            # Inputs are already tiled, shape (ensemble_size, batch_size, ...)
            outputs = ensemble(self.arch, *args)
        
        return outputs
    
    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access to allow calling methods on the ensemble.
        
        Args:
            name: Name of the attribute/method being accessed.
            
        Returns:
            A wrapped method that applies the ensemble to the base architecture's method.
        """
        # First check if it's a standard attribute
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        
        # Check if the base architecture has this method
        if hasattr(self.arch, name):
            attr = getattr(self.arch, name)
            if callable(attr):
                # Return a wrapper that applies the ensemble
                def ensemble_method(*args, tile_inputs: bool = True, **kwargs):
                    return self._apply_ensemble(name, *args, tile_inputs=tile_inputs, **kwargs)
                return ensemble_method
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    

class RPNEnsemble(nn.Module):
    """
    Reduced Parameter Network (RPN) Ensemble that combines outputs from train and prior ensembles.
    Outputs from prior and train ensembles should have the same shape.

    Args:
        arch_train: The trainable architecture/module for the ensemble.
        arch_prior: The prior architecture/module for the ensemble (parameters not updated during training).
        ensemble_size: Number of ensemble members.
        beta: Mixing parameter that weights the prior outputs.
        num_inputs: Number of input arguments the architecture expects (default: 1).
    """
    arch_train: nn.Module
    arch_prior: nn.Module
    ensemble_size: int
    beta: float = 1.
    num_inputs: int = 1

    @nn.compact
    def __call__(
        self, 
        *args: jnp.ndarray,
        tile_inputs: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Apply the RPN ensemble to the input(s).

        Args:
            *args: Input arrays. Number of inputs should match num_inputs.
                   Shape of each input depends on tile_inputs:
                      - If tile_inputs=True, shape is (batch_size, ...), then tiles to (ensemble_size, batch_size, ...)
                      - If tile_inputs=False, expected shape is (ensemble_size, batch_size, ...)
            tile_inputs: If True, inputs are tiled for each ensemble member.
            kwargs: Optional dict of extra arguments for the architecture.

        Returns:
            Combined ensemble outputs as either a tuple or single array, depending on the underlying model outputs.
        """
        # compute all outputs using a single forward pass
        train_outputs = Ensemble(self.arch_train, self.ensemble_size, self.num_inputs)(*args, tile_inputs=tile_inputs, **kwargs)  # shape (ensemble_size, ...)
        prior_outputs = Ensemble(self.arch_prior, self.ensemble_size, self.num_inputs)(*args, tile_inputs=tile_inputs, **kwargs)  # shape (ensemble_size, ...)

        # the stop_gradient operation makes it so that prior parameters are not changed during training
        # if using weight decay during training, the prior parameters should be masked
        prior_outputs = jax.lax.stop_gradient(prior_outputs)  # shape (ensemble_size, ...)

        # combines outputs and returns prediction
        if isinstance(prior_outputs, tuple):
            return tuple(
                train_component + self.beta * prior_component 
                for train_component, prior_component in zip(train_outputs, prior_outputs)
            )
        else:
            return train_outputs + self.beta * prior_outputs