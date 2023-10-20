"""Definitions of models and loss functions."""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import kfac_jax
from functools import partial


def create_model(name, **constructor_kwargs):
    """Create and transform an instance of hk.nets.`name` using `kwargs`."""
    model_constructor = getattr(hk.nets, name)
    if 'activation' in constructor_kwargs:
        constructor_kwargs['activation'] = getattr(jax.nn, constructor_kwargs['activation'])
    return hk.without_apply_rng(
        hk.transform_with_state(
            lambda x, **kwargs: model_constructor(**constructor_kwargs)(x, **kwargs)))


def create_loss(name, **kwargs):
    """Create an instance of `name` using ``kwargs``."""
    loss_function = globals()[name]
    return partial(loss_function, **kwargs)


def cross_entropy_loss(logits, labels, kfac_mask, num_classes):
    """Cross-entropy loss function, with necessary registration calls for KFAC-JAX."""
    # KFAC_JAX needs to be told to ignore padded data, but `mask` will only zero it,
    # so also set a corresponding correcting `weight`
    kfac_jax.register_softmax_cross_entropy_loss(
        jnp.where(jnp.isfinite(logits), logits, 0),
        labels,
        mask=kfac_mask,
        weight=kfac_mask.shape[0]/kfac_mask.sum())
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    return jnp.nanmean(
        optax.softmax_cross_entropy(
            logits, one_hot_labels))


def mse_loss(predictions, targets, kfac_mask):
    """MSE loss function, with necessary registration calls for KFAC-JAX."""
    kfac_jax.register_squared_error_loss(
        predictions,
        targets,
        weight=kfac_mask.shape[0]/kfac_mask.sum())
    return jnp.nanmean(
        optax.l2_loss(predictions, targets))
