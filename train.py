"""Root script for starting individual training runs."""

import jax
import jax.numpy as jnp
import numpy as np
import kfac_jax
import optax
import ray.tune as tune
import torch as to
from torch.utils.tensorboard import SummaryWriter

import pickle
import json
import os

import config
import datasets
import extern.optax_wrapper
import models
import optimisers
import util
from datetime import datetime


class TrainingState():
    """Convenience wrapper around Haiku/KFAC-JAX state objects."""

    def __init__(self, model, optimiser):
        """Construct blank state using objects provided."""
        self.model = model
        self.optimiser = optimiser
        self.model_params = None
        self.model_state = None
        self.optimiser_state = None
        self.statistics = None
        self.global_step = 0

    def init(self, rng, sample_input, sample_output, **model_kwargs):
        """Initialise state to its full form."""
        rng1, rng2 = jax.random.split(rng)
        self.model_params, self.model_state = self.model.init(
            rng1, jnp.ones(sample_input.shape), **model_kwargs)
        self.optimiser_state = self.optimiser.init(
            self.model_params, rng2,
            (sample_input, sample_output),
            func_state=self.model_state)

    def advance(self, batch, rng):
        """Perform one optimiser step, advancing the state by one iteration."""
        (self.model_params,
         self.optimiser_state,
         self.model_state,
         self.statistics) = self.optimiser.step(
             self.model_params,
             self.optimiser_state,
             func_state=self.model_state,
             batch=batch,
             rng=rng,
             global_step_int=self.global_step)
        self.global_step += 1

    def save(self, path):
        """Save the exportable components of `self` to `path`."""
        optimiser = self.optimiser
        model = self.model
        self.optimiser = None
        self.model = None
        with open(path, 'wb') as save_file:
            pickle.dump(self, save_file)
        self.optimiser = optimiser
        self.model = model

    def can_continue(self):
        if isinstance(self.optimiser_state, dict):
            if 'last_update' in self.optimiser_state:
                last_update = self.optimiser_state['last_update']
                if isinstance(last_update, dict):
                    if not all(np.isfinite(t).all()
                               for t in jax.tree_util.tree_flatten(last_update)[0]):
                        return False
                elif not np.isfinite(last_update).all():
                    return False
        return True



def construct_forward_pass_extra_kwargs(extra_kwargs, is_training=None):
    """Given the list of `extra_kwargs` required for this model, and
    corresponding specifiers of our current state, populate extra kwargs for
    the forward_pass call.
    """
    model_kwargs = {}
    if 'is_training' in extra_kwargs:
        model_kwargs['is_training'] = is_training
    return model_kwargs


def forward_pass(model, params, model_state, loss_function, batch, **model_kwargs):
    """Compute a forward pass of `model` to compute a loss value."""
    # Must pass finite data to the model, otherwise JAX grads break
    finite_data = jnp.isfinite(batch[0]).all(
        axis=range(1, batch[0].ndim), keepdims=True)
    model_output, state = model.apply(params,
                                      model_state,
                                      jnp.where(finite_data, batch[0], 0),
                                      **model_kwargs)
    model_output = jnp.where(finite_data.all(axis=range(model_output.ndim, batch[0].ndim)),
                             model_output,
                             jnp.nan)
    return loss_function(model_output, batch[1], kfac_mask=finite_data.squeeze()), state


non_training_forward_pass = jax.jit(forward_pass, static_argnames=('model', 'loss_function', 'is_training'))


def initialise_randomness(seed):
    """Reproducibly instantiate a random number generator."""
    if seed is None:
        seed = int(datetime.now().timestamp() * 1e6)
    to.manual_seed(seed)
    return jax.random.split(
        jax.random.PRNGKey(seed)
    )[0]


def create_optimiser(name, forward_pass_fn, **kwargs):
    """Construct and wrap (if necessary) the optimiser `class_name`."""
    wrapper_kwargs = dict(
        value_and_grad_func=jax.value_and_grad(forward_pass_fn, has_aux=True),
        value_func_has_aux=False,
        value_func_has_state=True,
        value_func_has_rng=False)

    if name == 'kfac_jax':
        if 'learning_rate' in kwargs:
            learning_rate = kwargs.pop('learning_rate')
            kwargs['learning_rate_schedule'] = lambda _: learning_rate
        if 'momentum' in kwargs:
            momentum = kwargs.pop('momentum')
            kwargs['momentum_schedule'] = lambda _: momentum
        if 'initial_damping' in kwargs and not kwargs['use_adaptive_damping']:
            initial_damping = kwargs.pop('initial_damping')
            kwargs['damping_schedule'] = lambda _: initial_damping
        optimiser = kfac_jax.Optimizer(**kwargs,
                                       **wrapper_kwargs,
                                       multi_device=False)
    elif hasattr(optax, name):
        optimiser = getattr(optax, name)(**kwargs)
        optimiser = extern.optax_wrapper.OptaxWrapper(
            optax_optimizer=optimiser, **wrapper_kwargs)
    elif hasattr(optimisers, name):
        optimiser = getattr(optimisers, name)(
            value_and_grad_func=jax.value_and_grad(forward_pass_fn, has_aux=True),
            **kwargs)
    else:
        raise NameError(f"Unknown optimiser {name}")

    return optimiser


def log_losses(state,
               model,
               loss_function,
               validation_dataset,
               test_dataset,
               logger,
               model_kwarg_spec,
               _ray_tune_config):
    """Log training loss from `state`, then calculate and log other losses."""
    for key in ('rho', 'damping', 'learning_rate', 'momentum', 'scale_factor'):
        if isinstance(state.optimiser_state, dict) and key in state.optimiser_state:
            value = state.optimiser_state[key]
        elif (isinstance(state.optimiser_state, tuple)
              and len(state.optimiser_state) == 2
              and key in state.optimiser_state[-1]):
            logger.add_scalar(f'Underlying_KFAC/{key.title()}',
                              getattr(state.optimiser_state[0][0], key,
                                      state.optimiser_state[0][1][key]).item(),
                              state.global_step)
            value = state.optimiser_state[-1][key]
        elif hasattr(state.optimiser_state, key):
            value = getattr(state.optimiser_state, key)
            if value is None:
                continue
        elif isinstance(state.statistics, dict) and key in state.statistics:
            value = state.statistics[key]
        elif hasattr(state.statistics, key):
            value = getattr(state.statistics, key)
        else:
            continue
        logger.add_scalar(f'Adaptive/{key.title()}',
                          value.item(),
                          state.global_step)

    for key in ('sfn_eigenvalues', 'sfn_inv_eigenvalues'):
        if key in state.statistics:
            logger.add_histogram(f'{key.title()}',
                                 np.array(state.statistics[key]),
                                 state.global_step,
                                 bins=100)

    training_loss = state.statistics['loss'].item()
    test_loss = non_training_forward_pass(model,
                                          state.model_params,
                                          state.model_state,
                                          loss_function,
                                          test_dataset,
                                          **construct_forward_pass_extra_kwargs(
                                              model_kwarg_spec,
                                              is_training=False))[0].item()
    if validation_dataset is not None:
        validation_loss = non_training_forward_pass(model,
                                                    state.model_params,
                                                    state.model_state,
                                                    loss_function,
                                                    validation_dataset,
                                                    **construct_forward_pass_extra_kwargs(
                                                        model_kwarg_spec,
                                                        is_training=False))[0].item()
        logger.add_scalar('Loss/Validation',
                        validation_loss,
                        state.global_step)
    else:
        validation_loss = None

    if _ray_tune_config:
        tune.report(training_loss=training_loss,
                    validation_loss=validation_loss,
                    test_loss=test_loss)
        with tune.checkpoint_dir(step=state.global_step) as checkpoint_dir:
            state.save(os.path.join(checkpoint_dir, 'checkpoint.pickle'))

    # Put at end to mitigate first-batch logging artifacts
    logger.add_scalar('Loss/Training',
                      training_loss,
                      state.global_step)
    logger.add_scalar('Loss/Test',
                      test_loss,
                      state.global_step)



def train(model,
          loss_function,
          optimiser,
          split_datasets,
          batch_size,
          num_epochs,
          rng,
          logger,
          model_kwarg_spec,
          initial_state=None,
          pytorch_import_path=None,
          _ray_tune_config=False):
    """Core training loop."""

    training_dataset, validation_dataset, test_dataset = split_datasets
    rng, sub_rng = jax.random.split(rng)
    if initial_state:
        state = initial_state
    else:
        state = TrainingState(model, optimiser)
        sample_batch = next(
            datasets.make_batches(
                training_dataset, batch_size, sub_rng
            ))
        rng, sub_rng = jax.random.split(rng)
        state.init(sub_rng,
                   *sample_batch,
                   **construct_forward_pass_extra_kwargs(model_kwarg_spec,
                                                         is_training=True))

    if pytorch_import_path:
        state = util.import_resnet18_state(state,
                                           to.load(pytorch_import_path))

    for epoch in range(num_epochs):
        rng, sub_rng = jax.random.split(rng)
        for batch in datasets.make_batches(
                training_dataset, batch_size, sub_rng):
            rng, sub_rng = jax.random.split(rng)
            state.advance(batch, rng=sub_rng)
            log_losses(state,
                       model,
                       loss_function,
                       validation_dataset,
                       test_dataset,
                       logger,
                       model_kwarg_spec,
                       _ray_tune_config)
            print('.', end='', flush=True)
            if not state.can_continue():
                return state
        if epoch % 1 == 0:
            training_loss = state.statistics['loss'].item()
            print(f'Epoch={epoch}  TrainingLoss={training_loss}', flush=True)

    return state


def main(config_dict=None, config_overrides={}):
    """Main entry point for performing a training run."""
    if config_dict is None:
        config_dict = config.load_config(config_overrides)
    else:
        util.nested_update(config_dict, config_overrides)

    train_kwargs = dict(
        rng=initialise_randomness(config_dict.get('seed', None)),
        model=models.create_model(**config_dict['model']),
        loss_function=models.create_loss(**config_dict['loss']),
        split_datasets=datasets.make_split_datasets(
            pad_to_equal_training_batches=config_dict['batch_size'],
            **config_dict['dataset']),
        num_epochs=config_dict['num_epochs'],
        batch_size=config_dict['batch_size'],
        _ray_tune_config=config_dict.get('_ray_tune_config', False),
        pytorch_import_path=config_dict.get('pytorch_import_path', None)
    )

    training_forward_pass_kwargs = construct_forward_pass_extra_kwargs(
        config_dict['forward_pass_extra_kwargs'],
        is_training=True)

    wrapped_forward_pass_fn = lambda params, model_state, batch: forward_pass(
        model=train_kwargs['model'],
        params=params,
        model_state=model_state,
        loss_function=train_kwargs['loss_function'],
        batch=batch,
        **training_forward_pass_kwargs)

    train_kwargs['optimiser'] = create_optimiser(forward_pass_fn=wrapped_forward_pass_fn,
                                                 **config_dict['optimiser'])

    if config_dict.get('load_state', False):
        with open(config_dict['load_state'], 'rb') as state_file:
            # This state has no model or optimiser,
            # so calls to .init() will fail
            train_kwargs['initial_state'] = pickle.load(state_file)
    log_directory = config.log_directory(config_dict)
    with SummaryWriter(log_dir=log_directory) as logger:
        with open(os.path.join(log_directory, 'config.json'), 'w') as config_file:
            json.dump(config_dict, config_file)
        state = train(logger=logger,
                      model_kwarg_spec=config_dict['forward_pass_extra_kwargs'],
                      **train_kwargs)
    if config_dict.get('save_state', False):
        state.save(config_dict['save_state'])


if __name__ == '__main__':
    main()
