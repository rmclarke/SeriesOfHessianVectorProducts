
import jax
import jax.numpy as jnp
import jax.lax as lax
import scipy.special

from functools import partial

import kfac_jax
import matplotlib.pyplot as plt
import optax
import util
from extern.minimize import minimize as jax_scipy_optimize_minimize
from extern._lbfgs import LBFGSResults

class HessianSeriesOptimiser():

    def __init__(self,
                 value_and_grad_func,
                 num_update_steps,
                 initial_scale_factor,
                 scale_factor_multiplier,
                 scale_factor_method,
                 acceleration_type,
                 acceleration_order,
                 series_name,
                 acceleration_kwargs={},
                 scale_factor_kwargs={},
                 learning_rate=None,
                 momentum=None,
                 adaptive_update=False,
                 hessian_damping_factor=None,
                 initial_damping=0,
                 damping_min=None,
                 damping_max=None):
        self.value_and_grad_func = value_and_grad_func
        self.num_update_steps = num_update_steps
        self.initial_scale_factor = initial_scale_factor
        self.scale_factor_multiplier = scale_factor_multiplier
        self.scale_factor_method = scale_factor_method
        self.acceleration_type = acceleration_type
        self.acceleration_order = acceleration_order
        self.acceleration_kwargs = acceleration_kwargs
        self.scale_factor_kwargs = scale_factor_kwargs
        self.series_name = series_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.adaptive_update = adaptive_update
        self.hessian_damping_factor = hessian_damping_factor
        self.initial_damping = initial_damping
        self.damping_clipping = (damping_min, damping_max)

        assert self.num_update_steps >= 2*self.acceleration_order

    def init(self, model_params, rng, *args, **kwargs):
        del args, kwargs
        zeroed_params_pytree = jax.tree_util.tree_map(
            jnp.zeros_like,
            model_params)
        state = {'scale_factor': jnp.array(self.initial_scale_factor, dtype=float),
                 'damping': jnp.array(self.initial_damping, dtype=float),
                 'rho': jnp.array(0.0, dtype=float),
                 'last_update': zeroed_params_pytree,
                 'last_learning_rate': jnp.array(0.0, dtype=float),
                 'rng': rng}
        if self.adaptive_update:
            ravelled_zeroed_params, _ = jax.flatten_util.ravel_pytree(zeroed_params_pytree)
            state['simplified_momentum_buffer'] = ravelled_zeroed_params
        return state

    def compute_scale_factor(self, hessian2_grad, gradient):
        return (
            jnp.linalg.norm(
                jax.flatten_util.ravel_pytree(hessian2_grad)[0],
                ord=2)
            / jnp.linalg.norm(
                jax.flatten_util.ravel_pytree(gradient)[0],
                ord=2))

    def compute_max_eigenvalue(self, flat_params, params_treedef, func_state, batch, rng):
        jvp_func = lambda *p: self.value_and_grad_func(
            jax.tree_util.tree_unflatten(params_treedef, p),
            func_state,
            batch)[1]
        return util.approx_max_eigenvalue_jvp(
            jvp_func,
            flat_params,
            rng,
            **self.scale_factor_kwargs) + self.scale_factor_kwargs['tolerance']

    def compute_new_vector_cache(self, vector_cache, hessian2_vector_cache, state):
        if self.series_name == 'quasi-damped':
            vector_cache = ((1-state['damping']/state['scale_factor'])*vector_cache
                            - (1/state['scale_factor'])*hessian2_vector_cache)
        elif self.series_name == 'damped':
            vector_cache = (vector_cache
                            - (1/state['scale_factor'])*hessian2_vector_cache)
        return vector_cache

    def compute_new_coefficient_cache(self, coefficient_cache, state, update_step):
        if self.series_name == 'quasi-damped':
            coefficient_cache = coefficient_cache * (1/4) * 2*update_step * (2*update_step - 1) / (update_step**2)
        elif self.series_name == 'damped':
            coefficient_cache = ((-1/((state['damping']**2 / state['scale_factor']) - 1))
                                 * (coefficient_cache
                                    + scipy.special.binom(0.5, update_step) * (-1)**update_step))
        return coefficient_cache

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, optimiser_state, batch, func_state, *args, **kwargs):
        del args, kwargs
        statistics = {}
        (loss, new_func_state), gradient = self.value_and_grad_func(params, func_state, batch)
        vector_cache = gradient
        if self.series_name == 'quasi-damped':
            coefficient_cache = 1
        elif self.series_name == 'damped':
            # This will be deferred until the new scale_factor is available and
            # has modified the damping appropriately
            pass
        update_sum = gradient
        cached_update_sums = []
        if (self.acceleration_order > 0 and
                self.num_update_steps == 2*self.acceleration_order + 1):
            cached_update_sums.append(update_sum)

        # Do HessianSeries calculation
        flat_params, params_treedef = jax.tree_util.tree_flatten(params)
        ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
        for update_step in range(1, self.num_update_steps):
            hessian2_vector_cache = vector_cache
            for _ in range(2):
                hessian2_vector_cache, hessian2_treedef = jax.tree_util.tree_flatten(hessian2_vector_cache)
                hessian2_vector_cache = jax.jvp(
                    lambda *p: self.value_and_grad_func(
                        jax.tree_util.tree_unflatten(params_treedef, p),
                        func_state,
                        batch)[1],
                    flat_params,
                    hessian2_vector_cache)[1]
            if update_step == 1:
                if self.scale_factor_method == 'lower_bound':
                    scale_factor = self.compute_scale_factor(
                        hessian2_vector_cache, gradient)
                elif self.scale_factor_method == 'power_series':
                    optimiser_state['rng'], sub_rng = jax.random.split(optimiser_state['rng'])
                    scale_factor = self.compute_max_eigenvalue(flat_params, params_treedef, func_state, batch, sub_rng)
                    scale_factor = scale_factor**2 + optimiser_state['damping']
                    optimiser_state['scale_factor'] = float('-inf')

                if self.series_name == 'quasi-damped':
                    optimiser_state['scale_factor'] = jnp.maximum(
                        scale_factor + optimiser_state['damping'],
                        optimiser_state['scale_factor'])
                elif self.series_name == 'damped':
                    optimiser_state['scale_factor'] = jnp.maximum(
                        scale_factor,
                        optimiser_state['scale_factor'])
                    # Update coefficient_cache with newly adjusted damping
                    coefficient_cache = 1 / (1 + optimiser_state['damping']/jnp.sqrt(optimiser_state['scale_factor']))
                    update_sum = jax.tree_util.tree_map(
                        lambda us: us * coefficient_cache,
                        update_sum)
                    if cached_update_sums:
                        cached_update_sums[0] = update_sum
            vector_cache = jax.tree_util.tree_map(
                partial(self.compute_new_vector_cache,
                        state=optimiser_state),
                vector_cache,
                hessian2_vector_cache)
            coefficient_cache = self.compute_new_coefficient_cache(
                coefficient_cache, optimiser_state, update_step)
            update_sum = jax.tree_util.tree_map(
                lambda us, vc: us + coefficient_cache*vc,
                update_sum,
                vector_cache)
            if (self.acceleration_order > 0
                    and update_step >= (self.num_update_steps - 2*self.acceleration_order - 1)):
                cached_update_sums.append(update_sum)
        update_sum = jax.tree_util.tree_map(
            lambda us: optimiser_state['scale_factor']**(-0.5) * us,
            update_sum)
        # Perform series acceleration
        if self.acceleration_order > 0:
            # Create a list of (flat_vector, unraveller_func) for each time step
            ravelled_sums = [jax.flatten_util.ravel_pytree(t)
                             for t in cached_update_sums]
            source_sequence = [t[0] for t in ravelled_sums]
            if self.acceleration_type == 'shanks':
                accelerated_sum = util.compute_epsilon_acceleration(
                    source_sequence,
                    num_applications=self.acceleration_order,
                    **self.acceleration_kwargs)
            elif self.acceleration_type == 'levin':
                accelerated_sum = util.compute_levin_acceleration(
                    source_sequence,
                    num_applications=self.acceleration_order,
                    **self.acceleration_kwargs)
            accelerated_sum = accelerated_sum * optimiser_state['scale_factor']**(-0.5)
            # Use the unraveller of the last time step
            update_sum = ravelled_sums[-1][1](accelerated_sum)

        # Calculate adaptive parameters if appropriate
        ravelled_update, _ = jax.flatten_util.ravel_pytree(update_sum)

        ravelled_last_update, _ = jax.flatten_util.ravel_pytree(optimiser_state['last_update']) #delta0, includes momentum term
        if self.adaptive_update:

            # SFN Hessian already has damping included
            grad_hess_grad = ravelled_update.T @ ravelled_gradient
            # Delta^T F delta_0 = delta_0^T F Delta
            grad_hess_last = ravelled_last_update.T @ ravelled_gradient
            # We assume the approximate SFN Hessian at this step will cancel
            # with the approximate SFN Hessian inverses at every previous step,
            # and trust that the repeated multiplication by momentum factors will
            # suppress the worst inaccuracies.
            # We still need to multiply with the previous iteration's learning rate:
            # delta_0 = alpha_0 (approx sfn H)_0 gradient_0 + mu_0 delta_{-1}
            # delta_n = alpha_n (approx sfn H)_n gradient_n + mu_n delta_{n-1}
            # so buffer_n = (approx sfn H)_n delta_n \approx mu_n buffer_{n-1} + alpha_n gradient_n
            last_hess_last = optimiser_state['last_learning_rate'] * ravelled_last_update.T @ optimiser_state['simplified_momentum_buffer']

            # If for any reason the last update is zero (either at initialisation or later,
            # because the gradient and momentum components are zero or cancel), we still
            # want to be able to follow the gradient, so this is fine.
            last_hess_last = jnp.where(last_hess_last == 0,
                                       1,
                                       last_hess_last)

            learning_rate, momentum = jnp.linalg.solve(
                jnp.array([[grad_hess_grad, grad_hess_last],
                           [grad_hess_last, last_hess_last]]),
                jnp.array([ravelled_gradient.T @ ravelled_update,
                           ravelled_gradient.T @ ravelled_last_update]))

            # Only save these out if we need them
            optimiser_state['last_learning_rate'] = learning_rate
            optimiser_state['simplified_momentum_buffer'] = (
                optimiser_state['simplified_momentum_buffer'] * momentum
                + learning_rate*ravelled_gradient)
        else:
            learning_rate = self.learning_rate
            momentum = self.momentum

        # used to be scaled_updates
        final_updates = jax.tree_util.tree_map(
            lambda u, lu: learning_rate*u + momentum*lu,
            update_sum,
            optimiser_state['last_update'])
        new_params = jax.tree_util.tree_map(
            lambda p, su: p - su,
            params,
            final_updates)
        optimiser_state['last_update'] = final_updates

        if self.hessian_damping_factor:
            last_loss = loss
            new_loss = self.value_and_grad_func(new_params, func_state, batch)[0][0]
            # ravelled_scaled_updates, _ = jax.flatten_util.ravel_pytree(scaled_updates)

            # `ravelled_scaled_update` doesn't account for the minus sign, so we add it here
            # We _don't_ want to use scaled updates because that already includes momentum.
            # That may have been alright when we were only calculating learning_rates a few 
            # versions ago, but not any more.
            # TODO: Think about damping cancellation - don't cancel, just add Delta (reg) Delta to denominator
            update_hess_update = ravelled_update.T @ ravelled_gradient
            update_hess_last = ravelled_last_update.T @ ravelled_gradient
            last_hess_last = optimiser_state['last_learning_rate'] * ravelled_last_update.T @ optimiser_state['simplified_momentum_buffer']

            quad_model_change = (0.5 * ((learning_rate**2) * update_hess_update
                                        + 2 * learning_rate * momentum * update_hess_last
                                        + (momentum**2) * last_hess_last)
                                 - learning_rate * update_hess_update
                                 - momentum * update_hess_last)
            rho = (new_loss - last_loss) / quad_model_change

            damping = jnp.where(
                rho > 3/4, self.hessian_damping_factor * optimiser_state['damping'],
                jnp.where(
                    rho < 1/4, optimiser_state['damping'] / self.hessian_damping_factor,
                    optimiser_state['damping']))
            if any(self.damping_clipping):
                damping = jnp.clip(damping, *self.damping_clipping)
            optimiser_state['damping'] = damping
            optimiser_state['rho'] = rho

        func_state = new_func_state
        statistics['loss'] = loss
        statistics['learning_rate'] = learning_rate
        statistics['momentum'] = momentum
        return new_params, optimiser_state, func_state, statistics


class SelectiveNewtonSGD():
    """Selective implementation of Newton's method with SGD."""

    def __init__(self,
                 value_and_grad_func,
                 eigenvalue_transform_method,
                 learning_rate=None,
                 momentum=0,
                 small_eigenvalue_threshold=None,
                 small_eig_learning_rate=None,
                 kfac_damping_factor=None,
                 initial_damping=0,
                 kfac_adaptivity=False,
                 kfac_damping_adaptation_interval=1,
                 kfac_prevent_bad_updates=False,
                 exp_scale=None,
                 exp_damping=None,
                 curvature_ema=None,
                 damping_min=None,
                 damping_max=None):
        self.value_and_grad_func = value_and_grad_func
        self.eigenvalue_transform_method = eigenvalue_transform_method
        self.hessian_func = jax.hessian(
            lambda params, func_state, batch, unraveller: self.value_and_grad_func(
                unraveller(params), func_state, batch)[0][0])
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.small_eigenvalue_threshold = small_eigenvalue_threshold
        self.small_eig_learning_rate = small_eig_learning_rate
        self.kfac_damping_factor = kfac_damping_factor
        self.initial_damping = initial_damping
        self.kfac_adaptivity = kfac_adaptivity
        self.kfac_prevent_bad_updates = kfac_prevent_bad_updates
        self.kfac_damping_adaptation_interval = kfac_damping_adaptation_interval
        self.exp_scale = exp_scale
        self.exp_damping = exp_damping
        self.curvature_ema = curvature_ema
        self.damping_clipping = (damping_min, damping_max)
        if self.kfac_prevent_bad_updates and not self.kfac_damping_factor:
            print("Error in AdaptiveSelectiveNewtonSGD configuration: Preventing bad updates only makes sense if the damping factor is being adjusted")

    def init(self, model_params, *args, **kwargs):
        flat_params, _ = jax.flatten_util.ravel_pytree(model_params)
        return {'damping': jnp.array(self.initial_damping, dtype=float),
                'eta': jnp.array(0.0, dtype=float),
                'damping_counter': jnp.array(0, dtype=int),
                'rho': jnp.array(0.0, dtype=float),
                'last_update': jnp.zeros_like(flat_params),
                'last_curvature': jnp.zeros((len(flat_params), len(flat_params)))}

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, optimiser_state, batch, func_state, global_step_int, *args, **kwargs):
        statistics = {}
        (loss, new_func_state), gradient = self.value_and_grad_func(params, func_state, batch)
        ravelled_params, param_unraveller = jax.flatten_util.ravel_pytree(params)
        original_hessian = self.hessian_func(ravelled_params, func_state, batch, param_unraveller)

        if self.curvature_ema:
            optimiser_state['last_curvature'] = jax.lax.cond(
                global_step_int == 0,
                lambda: original_hessian,
                lambda: optimiser_state['last_curvature'])

            hessian = (optimiser_state['last_curvature'] * self.curvature_ema
                       + original_hessian * (1-self.curvature_ema))
            optimiser_state['last_curvature'] = hessian
        else:
            hessian = original_hessian

        eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
        if self.eigenvalue_transform_method in ('abs', 'square'):
            if self.eigenvalue_transform_method == 'abs':
                eigenvalues = jnp.abs(eigenvalues)
            elif self.eigenvalue_transform_method == 'square':
                eigenvalues = eigenvalues**2
            if self.kfac_damping_factor:
                eigenvalues = eigenvalues + (optimiser_state['damping']
                                             + optimiser_state['eta'])

            if self.small_eigenvalue_threshold and self.small_eig_learning_rate:
                eigenvalues = jnp.where(eigenvalues <= self.small_eigenvalue_threshold,
                                        1/self.small_eig_learning_rate,
                                        eigenvalues)
            # statistics['sfn_eigenvalues'] = eigenvalues
            eigenvalues = 1/eigenvalues
            # statistics['sfn_inv_eigenvalues'] = eigenvalues
            if self.kfac_adaptivity:
                unaveraged_eigenvalues, unaveraged_eigenvectors = jnp.linalg.eigh(original_hessian)
                unaveraged_eigenvalues = jnp.abs(unaveraged_eigenvalues)
                if self.kfac_damping_factor:
                    unaveraged_eigenvalues = unaveraged_eigenvalues + (optimiser_state['damping']
                                                                       + optimiser_state['eta'])
                unaveraged_sfn_hessian = unaveraged_eigenvectors @ jnp.diag(unaveraged_eigenvalues) @ unaveraged_eigenvectors.T

        elif self.eigenvalue_transform_method == 'exp_interpolated':
            exp_factor = jnp.exp(-self.exp_scale * eigenvalues**2)
            eigenvalues = ((1 - exp_factor) / jnp.abs(eigenvalues)
                           +
                           (1 - eigenvalues**2) * exp_factor)
            # TODO: Do we want to use this or the true SFN Hessian to compute
            # learning rates etc.?
        elif self.eigenvalue_transform_method == 'exp_interpolated_damping':
            exp_factor = jnp.exp(-self.exp_scale * eigenvalues**2)
            eigenvalues = ((1 - exp_factor) / jnp.abs(eigenvalues)
                           +
                           exp_factor / (jnp.abs(eigenvalues) + self.exp_damping))
        else:
            raise ValueError(f'Invalid eigenvalue_transform_method {self.eigenvalue_transform_method}')

        ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
        updates = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T @ ravelled_gradient

        last_update = optimiser_state.get('last_update', jnp.zeros_like(updates))
        if self.kfac_adaptivity:
            # sfn_hessian already has damping included
            grad_hess_grad = updates.T @ unaveraged_sfn_hessian @ updates
            grad_hess_last = updates.T @ unaveraged_sfn_hessian @ last_update
            last_hess_last = last_update.T @ unaveraged_sfn_hessian @ last_update

            # If for any reason the last update is zero (either at initialisation or later,
            # because the gradient and momentum components are zero or cancel), we still
            # want to be able to follow the gradient, so this is fine.
            last_hess_last = jnp.where(last_hess_last == 0,
                                       1,
                                       last_hess_last)

            learning_rate, momentum = jnp.linalg.solve(
                jnp.array([[grad_hess_grad, grad_hess_last],
                           [grad_hess_last, last_hess_last]]),
                jnp.array([ravelled_gradient.T @ updates,
                           ravelled_gradient.T @ last_update]))
        else:
            learning_rate = self.learning_rate
            momentum = self.momentum

        final_update = learning_rate * updates + momentum * last_update
        optimiser_state['last_update'] = final_update
        new_params = param_unraveller(ravelled_params - final_update)

        if self.kfac_damping_factor:
            last_loss = loss
            new_loss = self.value_and_grad_func(new_params, func_state, batch)[0][0]

            def undo_step(optimiser_state):
                optimiser_state['damping'] = self.kfac_damping_factor * optimiser_state['damping']
                optimiser_state['last_update'] = last_update
                optimiser_state['damping_counter'] = 0
                return optimiser_state, params, last_loss

            def commit_step(optimiser_state):
                optimiser_state['rho'], optimiser_state['damping'] = lax.cond(
                    optimiser_state['damping_counter'] == 0,
                    self.update_damping,
                    lambda optimiser_state, *_: (optimiser_state['rho'], optimiser_state['damping']),
                    optimiser_state,
                    last_loss,
                    new_loss,
                    updates,
                    last_update,
                    learning_rate,
                    momentum,
                    unaveraged_sfn_hessian,
                    ravelled_gradient)

                optimiser_state['damping_counter'] = (optimiser_state['damping_counter'] + 1) % self.kfac_damping_adaptation_interval
                return optimiser_state, new_params, loss

            if self.kfac_prevent_bad_updates:
                optimiser_state, new_params, loss = lax.cond(new_loss > last_loss, undo_step, commit_step, optimiser_state)
            else:
                commit_step(optimiser_state)

        func_state = new_func_state
        statistics.update(
            dict(loss=loss,
                 learning_rate=learning_rate,
                 momentum=momentum))
        return new_params, optimiser_state, func_state, statistics

    def update_damping(self, optimiser_state, last_loss, new_loss, updates, last_update, learning_rate, momentum, unaveraged_sfn_hessian, ravelled_gradient):
        # `updates` doesn't account for the minus sign, so we add it here
        # Damping in the denominator has already been added to the hessian
        # sfn_hessian includes damping
        update_hess_update = updates.T @ unaveraged_sfn_hessian @ updates
        update_hess_last = updates.T @ unaveraged_sfn_hessian @ last_update
        last_hess_last = last_update.T @ unaveraged_sfn_hessian @ last_update

        quad_model_change = (0.5 * ((learning_rate**2) * update_hess_update
                                    + 2 * learning_rate * momentum * update_hess_last
                                    + (momentum**2) * last_hess_last)
                             - learning_rate * ravelled_gradient.T @ updates
                             - momentum * ravelled_gradient.T @ last_update)
        rho = (new_loss - last_loss) / quad_model_change

        # if we aren't adjusting damping now, keep it fixed.
        # Otherwise:
        # if damping > 3/4: damping = damping * factor
        # elif damping < 1/4: damping = damping / factor
        # else: damping = damping
        damping = jnp.where(
            rho > 3/4, self.kfac_damping_factor * optimiser_state['damping'],
            jnp.where(
                rho < 1/4, optimiser_state['damping'] / self.kfac_damping_factor,
                optimiser_state['damping']))
        if any(self.damping_clipping):
            damping = jnp.clip(damping, *self.damping_clipping)

        return rho, damping


class WrappedLBFGS():
    """Wrapper around the SciPy LBFGS implementation to match the
    OptaxWrapper format.
    """

    def __init__(self, value_and_grad_func, **lbfgs_kwargs):
        self.value_and_grad_func = value_and_grad_func
        self.lbfgs_kwargs = lbfgs_kwargs

    def init(self, params, _, batch, *args, **kwargs):
        # Copy logic from extern._lbfgs
        x0 = params
        (f_0, _), g_0 = self.value_and_grad_func(
            x0, kwargs['func_state'], batch)
        g_0, _ = jax.flatten_util.ravel_pytree(g_0)

        x0, _ = jax.flatten_util.ravel_pytree(x0)
        d = len(x0)
        dtype = jnp.dtype(x0)
        maxcor = self.lbfgs_kwargs.get('maxcor', 10)
        return LBFGSResults(
            converged=jnp.array(False, dtype=bool),
            failed=jnp.array(False, dtype=bool),
            k=0,
            nfev=1,
            ngev=1,
            x_k=x0,
            f_k=f_0,
            g_k=g_0,
            s_history=jnp.zeros((maxcor, d), dtype=dtype),
            y_history=jnp.zeros((maxcor, d), dtype=dtype),
            rho_history=jnp.zeros((maxcor,), dtype=dtype),
            gamma=jnp.array(1., dtype=float),
            status=0,
            ls_status=0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, optimiser_state, batch, func_state, *args, **kwargs):
        del args, kwargs
        ravelled_params, param_unraveller = jax.flatten_util.ravel_pytree(params)

        def value_func(params, *args):
            return self.value_and_grad_func(param_unraveller(params), *args)[0][0]

        optimiser_result, optimiser_state = jax_scipy_optimize_minimize(
            fun=value_func,
            x0=ravelled_params,
            args=(func_state, batch),
            method='l-bfgs-experimental-do-not-rely-on-this',
            options=dict(state_initial=optimiser_state,
                         **self.lbfgs_kwargs))

        params = param_unraveller(optimiser_result.x)
        return params, optimiser_state, func_state, dict(loss=optimiser_result.fun)


class NewtonRootSFN():
    """Selective implementation of Newton's method with SGD."""

    def __init__(self,
                 value_and_grad_func,
                 damping,
                 tolerance,
                 commutation_tolerance,
                 eigenvalue_kwargs={},
                 learning_rate=1,
                 max_num_steps=100):
        self.value_and_grad_func = value_and_grad_func
        self.hessian_func = jax.hessian(
            lambda params, func_state, batch, unraveller: self.value_and_grad_func(
                unraveller(params), func_state, batch)[0][0])
        self.learning_rate = learning_rate
        self.max_num_steps = max_num_steps
        self.damping = damping
        self.tolerance = tolerance
        self.commutation_tolerance = commutation_tolerance
        self.eigenvalue_kwargs = eigenvalue_kwargs

    def init(self, model_params, rng, batch, func_state, *args, **kwargs):
        ravelled_params, param_unraveller = jax.flatten_util.ravel_pytree(model_params)
        hessian = self.hessian_func(ravelled_params, func_state, batch, param_unraveller)
        hessian2 = hessian @ hessian
        alpha = -0.5
        z = 3 / (2 * jnp.linalg.norm(hessian2, ord='fro'))
        X = jnp.eye(hessian.shape[0]) / z**alpha
        return {'last_X': X,
                'rng': rng}

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, optimiser_state, batch, func_state, global_step_int, *args, **kwargs):
        statistics = {}
        (loss, new_func_state), gradient = self.value_and_grad_func(params, func_state, batch)
        ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
        ravelled_params, param_unraveller = jax.flatten_util.ravel_pytree(params)
        hessian = self.hessian_func(ravelled_params, func_state, batch, param_unraveller)
        hessian2 = hessian @ hessian
        identity = jnp.eye(hessian.shape[0])

        # Compute updates
        optimiser_state['rng'], sub_rng = jax.random.split(optimiser_state['rng'])
        max_eigenvalue = util.appox_max_eigenvalue(hessian2, sub_rng, **self.eigenvalue_kwargs)
        hessian2 = hessian2 + self.damping * max_eigenvalue * identity
        alpha = -0.5
        i = jnp.zeros(1)

        X = optimiser_state['last_X']
        M = X @ X @ hessian2

        def abort_hot_start(*_):
            z = 3 / (2 * jnp.linalg.norm(hessian2, ord='fro'))
            X = jnp.eye(hessian.shape[0]) / z**alpha
            M = X @ X @ hessian2
            return M, X

        M, X = jax.lax.cond(
            ((X @ M - M @ X) > self.commutation_tolerance).any(),
            abort_hot_start,
            lambda M, X: (M, X),
            M, X)

        def loop_body(state):
            i, M, X = state
            M1 = (1 - alpha)*identity + alpha*M
            X = X @ M1
            M = M1 @ M1 @ M
            i += 1
            return i, M, X

        i, _, X = jax.lax.while_loop(
            cond_fun=lambda state: jax.lax.bitwise_and(
                           state[0] < self.max_num_steps,
                           jnp.linalg.norm(state[1] - identity, ord=float('inf')) > self.tolerance)[0],
            body_fun=loop_body,
            init_val=(i, M, X))

        optimiser_state['last_X'] = X
        final_update = self.learning_rate * X @ ravelled_gradient
        new_params = param_unraveller(ravelled_params - final_update)

        func_state = new_func_state
        statistics.update(
            dict(loss=loss, num_steps=i))
        return new_params, optimiser_state, func_state, statistics


class ObservedKFAC():
    """Wrapper around KFAC and SelectiveNewtonSGD optimisers, using the
    former to compute a curvature matrix to pass to the latter.
    """

    def __init__(self, value_and_grad_func, kfac_kwargs, newton_kwargs):
        self.value_and_grad_func = value_and_grad_func
        self.kfac_optimiser = kfac_jax.Optimizer(
            **kfac_kwargs,
            value_and_grad_func=value_and_grad_func,
            value_func_has_aux=False,
            value_func_has_state=True,
            value_func_has_rng=False,
            multi_device=False)

        self.newton_optimiser = SelectiveNewtonSGD(
            value_and_grad_func=value_and_grad_func,
            **newton_kwargs)
        self.counter = 0

    def init(self, model_params, *args, **kwargs):
        _, self.params_unraveller = jax.flatten_util.ravel_pytree(model_params)
        return self.kfac_optimiser.init(model_params, *args, **kwargs)

    def step(self, model_params, kfac_state, batch, func_state, *args, **kwargs):
        kfac_estimator_state = jax.tree_util.tree_map(jnp.copy, kfac_state.estimator_state)
        kfac_func_args = jax.tree_util.tree_map(
            jnp.copy,
            self.kfac_optimiser._setup_func_args_and_rng(
                model_params, kwargs['rng'], batch, func_state)[0])
        kfac_outputs = self.kfac_optimiser.step(model_params,
                                                kfac_state,
                                                batch=batch,
                                                func_state=func_state, *args, **kwargs)

        if self.counter % 10 == 0:
            kfac_approx_fisher = self.kfac_optimiser._estimator.to_dense_matrix(
                kfac_estimator_state)

            identity = jnp.eye(kfac_approx_fisher.shape[0])
            kfac_exact_fisher = jnp.stack(
                [jax.flatten_util.ravel_pytree(
                    self.kfac_optimiser._implicit.multiply_fisher(
                        jax.tree_util.tree_map(jnp.copy, kfac_func_args),
                        self.params_unraveller(identity_row))
                    )[0]
                 for identity_row in identity])
            approx_inv_fisher = jnp.stack(
                [jax.flatten_util.ravel_pytree(
                    self.kfac_optimiser._estimator.multiply_inverse(
                        kfac_estimator_state,
                        self.params_unraveller(identity_row),
                        identity_weight=0,
                        exact_power=0,
                        use_cached=True)
                    )[0]
                 for identity_row in identity])

            fig = plt.gcf()
            grid = fig.add_gridspec(2, 4, width_ratios=(15, 15, 15, 1))
            exact_fisher_ax = fig.add_subplot(grid[0, 0])
            approx_fisher_ax = fig.add_subplot(grid[0, 1])
            vanilla_error_ax = fig.add_subplot(grid[0, 2])
            exact_inv_fisher_ax = fig.add_subplot(grid[1, 0])
            approx_inv_fisher_ax = fig.add_subplot(grid[1, 1])
            inv_error_ax = fig.add_subplot(grid[1, 2])
            colourbar_ax = fig.add_subplot(grid[:, 3])
            error_matrix = kfac_approx_fisher - kfac_exact_fisher
            # val_limit = max(jnp.abs(matrix).max()
            #                 for matrix in (kfac_exact_fisher,
            #                                kfac_clean_curvature,
            #                                error_matrix))
            # min_val = -val_limit
            # max_val = val_limit
            min_val = None
            max_val = None

            full_img = exact_fisher_ax.imshow(
                jnp.tanh(
                    kfac_exact_fisher / jnp.abs(kfac_exact_fisher).max()),
                vmin=min_val,
                vmax=max_val)
            exact_fisher_ax.set_title("tanh normed KFAC True Fisher")

            approx_fisher_ax.imshow(
                jnp.tanh(
                    kfac_approx_fisher / jnp.abs(kfac_approx_fisher).max()),
                vmin=min_val,
                vmax=max_val)
            approx_fisher_ax.set_title("tanh normed KFAC Approximate Fisher")

            vanilla_error_ax.imshow(
                jnp.tanh(
                    error_matrix / jnp.abs(error_matrix).max()),
                vmin=min_val,
                vmax=max_val)
            vanilla_error_ax.set_title("tanh normed Error in Approximation")

            exact_inv_fisher = jnp.linalg.inv(kfac_exact_fisher)
            exact_inv_fisher /= jnp.abs(exact_inv_fisher).max()
            exact_inv_fisher_ax.imshow(jnp.tanh(exact_inv_fisher), vmin=min_val, vmax=max_val)
            exact_inv_fisher_ax.set_title("tanh normed KFAC True Inverse Fisher")

            approx_inv_fisher /= jnp.abs(approx_inv_fisher).max()
            approx_inv_fisher_ax.imshow(jnp.tanh(approx_inv_fisher), vmin=min_val, vmax=max_val)
            approx_inv_fisher_ax.set_title("tanh normed KFAC Approximate Inverse Fisher")

            inv_error_matrix = exact_inv_fisher - approx_inv_fisher
            inv_error_matrix /= jnp.abs(inv_error_matrix).max()
            inv_error_ax.imshow(jnp.tanh(inv_error_matrix), vmin=min_val, vmax=max_val)
            inv_error_ax.set_title("tanh normed Error in Inverse Approximation")

            plt.colorbar(full_img, cax=colourbar_ax)
            fig.set_size_inches(16, 9)
            plt.savefig(f"./figs_kfac/{self.counter}.pdf", dpi=600)
            plt.close()
            # plt.draw()
            # plt.show()
        self.counter += 1

        return kfac_outputs
