"""Helper utilities and functions."""

import os

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import scipy.special
import torch as to
from tbparse import SummaryReader


def nested_update(source_dict, update_dict):
    """Recursively update each level of `source_dict` with the contents of
    `update_dict`.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in source_dict:
            nested_update(source_dict[key], value)
        else:
            source_dict[key] = value


def bootstrap_sample(data_length, num_datasets, num_samples=None):
    """Bootstrap sample, generating `num_datasets` sample sets of `num_samples`
    each, returning the indices of the sample.
    """
    if num_samples is None:
        num_samples = data_length
    return np.random.choice(data_length,
                            replace=True,
                            size=(num_datasets, num_samples))


def interpolate_timestamps(data_values, data_times, num_timestamps):
    """Interpolate the `data_values` over `data_times` so that the data can all
    be represented with one array of `num_timestamps` timestamps.
    """
    # Keep second dimension when indexing
    data_elapsed_times = data_times - data_times[:, 0:1]
    interp_timestamps = np.linspace(data_elapsed_times.min(),
                                    data_elapsed_times.max(),
                                    num_timestamps)

    all_values = np.empty((len(data_values), num_timestamps))
    for run_id, (run_values, run_timestamps) in enumerate(zip(
            data_values, data_elapsed_times)):
        assert np.all(np.diff(run_timestamps) > 0)
        all_values[run_id] = np.interp(interp_timestamps,
                                       run_timestamps,
                                       run_values)

    return all_values, interp_timestamps


def extract_old_uci_order(old_root, new_root):
    """Load the UCI dataset order from `old_raw_root` and compute the indices
    necessary to transform `new_raw_root` in the same way.
    """
    new_rows = np.loadtxt(os.path.join(new_root, 'raw', 'data_targets.txt'))
    old_rows = to.cat(
        (to.load(os.path.join(old_root, 'data.pt')),
         to.load(os.path.join(old_root, 'targets.pt'))),
        dim=1)
    indices = [(new_rows == old_row.numpy()).prod(axis=1).argmax()
               for old_row in old_rows]
    np.savetxt(os.path.join(new_root, 'permutation_indices.txt'),
               indices,
               fmt='%i')


def maybe_initialise_determinism(seed=None):
    """Initialiser function for main and worker threads, using the provided
    random `seed` if it is set.
    """
    if seed is None:
        return

    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        # or :16:8 if memory becomes an issue
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    to.manual_seed(seed)
    np.random.seed(seed)
    to.backends.cudnn.benchmark = False
    to.use_deterministic_algorithms(True)


def visualise_graph(*args, graph_id=1, **kwargs):
    import torchviz
    graph = torchviz.make_dot(*args, **kwargs)
    graph.render(f'graph_{graph_id}.gv')


def cosine_similarity(x, y):
    return x.dot(y) / (to.linalg.norm(x) * to.linalg.norm(y))


def compute_epsilon_acceleration(source_sequence,
                                 num_applications=1,
                                 inverse_func='samelson',
                                 modifier='sablonniere'):
    """Compute `num_applications` recursive Shanks transformation of
    `source_sequence` (preferring later elements) using `inverse_func` and the
    epsilon-algorithm.
    """
    def inverse(vector):
        if inverse_func == 'elementwise':
            return 1 / vector
        elif inverse_func == 'samelson':
            return vector / vector.dot(vector)
        else:
            raise ValueError(inverse_func)

    epsilon = {}
    for m, source_m in enumerate(source_sequence):
        epsilon[m+1, -1] = 0
        epsilon[m, 0] = source_m

    s = 1
    m = (len(source_sequence) - 1) - 2*num_applications
    initial_m = m
    while m < len(source_sequence) - 1:
        while m >= initial_m:
            if modifier == 'sabblonniere':
                inverse_scaling = np.floor(s/2) + 1
            else:
                inverse_scaling = 1

            epsilon[m, s] = epsilon[m+1, s-2] + inverse_scaling * inverse(epsilon[m+1, s-1] - epsilon[m, s-1])
            epsilon.pop((m+1, s-2))
            m -= 1
            s += 1
        m += 1
        s -= 1
        epsilon.pop((m, s-1))
        m = initial_m + s
        s = 1

    return epsilon[initial_m, 2*num_applications]


def compute_levin_acceleration(source_sequence,
                               num_applications=1,
                               transform_type='t',
                               d_transform_order=None):
    """Compute `num_applications` recursive Levin transformations of
    `source_sequence` using `transform_type`.
    """

    def R(n):
        if transform_type == 't':
            trial_vector = source_sequence[n+1] - source_sequence[n]
        elif transform_type == 'u':
            trial_vector = n * (source_sequence[n] - source_sequence[n-1])
        elif transform_type == 'v':
            diff_cur = source_sequence[n+1] - source_sequence[n]
            diff_last = source_sequence[n] - source_sequence[n-1]
            trial_vector = diff_last * diff_cur / (diff_cur - diff_last)
        else:
            raise ValueError(transform_type)
        chosen_index = jnp.argmax(jnp.abs(trial_vector))
        return trial_vector[chosen_index]

    def H(n, k):
        if k == 0:
            return source_sequence[n]
        else:
            return H(n, k-1) - g(n, k-1, k) * (H(n+1, k-1) - H(n, k-1))/(g(n+1, k-1, k) - g(n, k-1, k))

    def g(n, k, i):
        if k == 0:
            if d_transform_order:
                q = i % d_transform_order
                p = i // d_transform_order
                if q == 0:
                    q = d_transform_order
                    p -= 1
                trial_vector = sum(scipy.special.binom(q, z)
                                   * ((-1)**(q-z))
                                   * source_sequence[n-q + z]
                                   for z in range(0, q+1))
                chosen_index = trial_vector.abs().argmax()
                return (n**(q-p)) * trial_vector[chosen_index]
            else:
                return (n**(1-i)) * R(n)
        else:
            return g(n, k-1, i) - g(n, k-1, k) * (g(n+1, k-1, i) - g(n, k-1, i))/(g(n+1, k-1, k) - g(n, k-1, k))

    if d_transform_order:
        return H(len(source_sequence) - num_applications - 2, num_applications*d_transform_order)
    else:
        return H(len(source_sequence) - num_applications - 2, num_applications)


def compute_pade_acceleration(source_sequence,
                              num_applications=1,
                              transform_type='hybrid'):
    step_llast, step_last, step_cur = source_sequence[-3:]
    if transform_type == 'hybrid':
        scale_factor = ((step_last - step_llast).dot(step_cur - step_last)
                        / (step_cur - step_last).dot(step_cur - 2*step_last + step_llast))
    elif transform_type == 'aitken':
        scale_factor = ((step_last - step_llast).dot(step_last - step_llast)
                        / (step_last - step_llast).dot(step_cur - 2*step_last + step_llast))

    return step_last - (step_cur - step_last) * scale_factor


def pandas_from_tensorboard(root_directory):
    store_kwargs = dict(path=os.path.join(root_directory, 'pandas_store.h5'))
    if (os.access(store_kwargs['path'], os.F_OK)
            and not os.access(store_kwargs['path'], os.W_OK)):
        store_kwargs['mode'] = 'r'
    with pd.HDFStore(**store_kwargs) as store:
        if 'scalars' in store:
            return store['scalars']
        else:
            scalar_data = SummaryReader(root_directory,
                                        pivot=False,
                                        extra_columns={'dir_name', 'wall_time'}).scalars
            store['scalars'] = scalar_data
            return scalar_data


def bootstrap_aggregate(dataframe, aggregator, num_repetitions):
    rng = np.random.default_rng()
    aggregated_samples = pd.concat(
        (pd.DataFrame(
            data=aggregator(
                rng.choice(
                    dataframe.values,
                    axis=1,
                    size=dataframe.shape[1]),
                axis=1),
            index=dataframe.index)
         for _ in range(num_repetitions)),
        axis='columns')
    mean_statistic = aggregated_samples.agg('mean', axis='columns')
    std_statistic = aggregated_samples.agg('std', axis='columns')
    return mean_statistic, std_statistic


def compute_damped_coefficient(n, damping):
    summed_factors = 0
    for k in range(n+1):
        summed_factors += scipy.special.binom(0.5, k) * (damping**2 - 1)**k
    return ((-1)**n
            * (damping**2 - 1)**(-1-n)
            * (damping - summed_factors))


def appox_max_eigenvalue(matrix, rng, tolerance, max_num_steps):
    """Approximately compute the largest eigenvalue of `matrix`."""
    v = jax.random.uniform(rng, matrix.shape[0:1])
    error = jnp.array(float('inf'))
    eigenvalue = jnp.array(0.)
    i = jnp.zeros(1)

    def loop_body(state):
        i, error, eigenvalue, v = state
        v_unit = v / jnp.linalg.norm(v, ord=2)
        v = matrix @ v_unit
        last_eigenvalue = eigenvalue
        eigenvalue = v_unit.T @ v
        error = jnp.abs(eigenvalue - last_eigenvalue)
        i += 1
        return i, error, eigenvalue, v

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: jax.lax.bitwise_and(state[0] < max_num_steps,
                                              state[1] > tolerance)[0],
        body_fun=loop_body,
        init_val=(i, error, eigenvalue, v))
    return final_state[2]


def approx_max_eigenvalue_jvp(jvp_func, flat_params, rng, tolerance, max_num_steps):
    """Approximately compute the largest eigenvalue of a matrix defined by its
    product with an arbitrary vector, `matrix_product`.
    """
    v = [jax.random.uniform(rng, param.shape) for param in flat_params]
    error = jnp.array(float('inf'))
    eigenvalue = jnp.array(0.)
    i = jnp.zeros(1)

    def loop_body(state):
        i, error, eigenvalue, v = state
        ravelled_v, v_unraveller = jax.flatten_util.ravel_pytree(v)
        v_unit_ravelled = ravelled_v / jnp.linalg.norm(ravelled_v, ord=2)
        v_unit = v_unraveller(v_unit_ravelled)
        v = jax.jvp(jvp_func, flat_params, v_unit)[1]
        ravelled_v, v_unraveller = jax.flatten_util.ravel_pytree(v)
        last_eigenvalue = eigenvalue
        eigenvalue = v_unit_ravelled.T @ ravelled_v
        error = jnp.abs(eigenvalue - last_eigenvalue)
        i += 1
        flattened_v = jax.tree_util.tree_flatten(v)[0]
        return i, error, eigenvalue, flattened_v

    final_state = jax.lax.while_loop(
        cond_fun=lambda state: jax.lax.bitwise_and(state[0] < max_num_steps,
                                              state[1] > tolerance)[0],
        body_fun=loop_body,
        init_val=(i, error, eigenvalue, v))
    return final_state[2]


def flatten_keys(data, prefix=''):
    """Flatten all keys of the nested dict `data` into one tuple."""
    flat_keys = []
    for key, value in data.items():
        if isinstance(value, dict):
            flat_keys.extend(flatten_keys(value, prefix + key + '.'))
        else:
            flat_keys.append(prefix + key)
    return flat_keys


def pytorch_resnet18_to_haiku_key(pytorch_key, prefix='res_net18/~/'):
    """Translate `key` from the PyTorch ResNet-18 model into Haiku notation."""
    haiku_key = prefix
    split_pytorch_key = pytorch_key.split('.')

    if split_pytorch_key[0].startswith('conv'):
        if haiku_key == 'res_net18/~/':
            return haiku_key + 'initial_conv.w'
        else:
            haiku_index = int(split_pytorch_key[0][-1]) - 1
            return haiku_key + f'conv_{haiku_index}.w'
    elif split_pytorch_key[0] == 'fc':
        haiku_key += 'logits.'
        if split_pytorch_key[1] == 'weight':
            haiku_key += 'w'
        elif split_pytorch_key[1] == 'bias':
            haiku_key += 'b'
        return haiku_key
    elif split_pytorch_key[0].startswith('layer'):
        group_num = int(split_pytorch_key[0][5:]) - 1
        block_num = int(split_pytorch_key[1])
        haiku_key += f'block_group_{group_num}/~/block_{block_num}/~/'
        return pytorch_resnet18_to_haiku_key('.'.join(split_pytorch_key[2:]),
                                             prefix=haiku_key)

    if split_pytorch_key[0].startswith('bn'):
        if haiku_key == 'res_net18/~/':
            haiku_key += 'initial_batchnorm'
        else:
            haiku_index = int(split_pytorch_key[0][-1]) - 1
            haiku_key += f'batchnorm_{haiku_index}'
    elif split_pytorch_key[0] == 'downsample':
        haiku_key += 'shortcut_'
        if split_pytorch_key[1] == '0':
            return haiku_key + 'conv.w'
        else:
            haiku_key += 'batchnorm'

    if split_pytorch_key[-1] == 'running_mean':
        haiku_key += '/~/mean_ema.average'
    elif split_pytorch_key[-1] == 'running_var':
        haiku_key += '/~/var_ema.average'
    elif split_pytorch_key[-1] == 'weight':
        haiku_key += '.scale'
    elif split_pytorch_key[-1] == 'bias':
        haiku_key += '.offset'

    return haiku_key


def import_resnet18_state(haiku_state, pytorch_state):
    model_params = haiku_state.model_params
    model_state = haiku_state.model_state

    for pytorch_key, pytorch_value in pytorch_state.items():
        if pytorch_key.startswith('fc'):
            # This is the final fully-connected head trained on ImageNet, which
            # we need to replace with our own retrained layer
            continue

        haiku_key = pytorch_resnet18_to_haiku_key(pytorch_key)
        haiku_key_tuple = haiku_key.split('.')

        # Select correct Haiku state object
        haiku_dict = model_state
        for subkey in haiku_key_tuple:
            if subkey not in haiku_dict:
                haiku_dict = model_params
                break
            else:
                haiku_dict = haiku_dict[subkey]
        else:
            haiku_dict = model_state

        # Access correct Haiku sub-dictionary
        final_key = haiku_key_tuple[-1]
        for subkey in haiku_key_tuple[:-1]:
            haiku_dict = haiku_dict[subkey]
        # Update Retain Haiku object structure while importing PyTorch values
        haiku_dict[final_key] = (jnp.zeros_like(haiku_dict[final_key])
                                 + np.ascontiguousarray(pytorch_value.detach().T.numpy()))

    return haiku_state


def round_sig_figs(data, num_sig_figs):
    """Round `data` to `num_sig_figs.speed`."""
    return data.apply(
        lambda x: (
            round(x,
                  (num_sig_figs - np.floor(np.log10(np.abs(x))) - 1).astype(int))
            if x != 0 else x))
