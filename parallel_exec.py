import copy
import numpy as np
import os
import pathlib
import ray.tune as tune
import ray.tune.suggest.bayesopt

import config
import pickle
import local_config
import train


def ray_tune_trainable(config, checkpoint_dir=None):
    """Trainable function to start training one configuration under Ray Tune.
    """
    # Undo Ray's directory changing so our relative paths work
    os.chdir(
        pathlib.Path(__file__).parent.resolve())

    repetition_override = {
        '_ray_tune_config': True,
        'optimiser': {
            key[4:]: config[key] for key in config if key.startswith('opt.')}}
    repetition_override.update(
        {key[5:]: config[key] for key in config if key.startswith('root.')})
    for kwarg_key in ('eigenvalue_kwargs', 'scale_factor_kwargs'):
        for eigenvalue_kwarg in ('tolerance', 'max_num_steps'):
            initial_key = f'{kwarg_key}.{eigenvalue_kwarg}'
            if initial_key in repetition_override['optimiser']:
                eigenvalue_kwargs = repetition_override['optimiser'].setdefault(f'{kwarg_key}', {})
                eigenvalue_kwargs[eigenvalue_kwarg] = repetition_override['optimiser'].pop(initial_key)
    if 'shanks_acceleration_dict' in config:
        repetition_override['optimiser'].update(
            config['shanks_acceleration_dict'])
    for adam_beta in ('b1', 'b2'):
        temp_key = f'adam_one_minus_{adam_beta}'
        if temp_key in config:
            repetition_override['optimiser'][adam_beta] = 1 - config.pop(temp_key)
    if checkpoint_dir:
        repetition_override['load_state'] = os.path.join(checkpoint_dir,
                                                         'checkpoint.pt')

    master_config = copy.deepcopy(config['master_config'])
    train.main(config_dict=master_config,
               config_overrides=repetition_override)


def ray_tune_run(num_configurations,
                 tune_mode,
                 master_config,
                 local_dir=local_config.scratch_dir):
    """Main execution function for managing ASHA runs using Ray Tune."""
    search_space_spec = master_config.pop('ray_search_space_spec', None)
    runs_per_gpu = master_config.pop('runs_per_gpu', None)
    tuning_metric = master_config.pop('tuning_metric', None)
    if tuning_metric is None:
        print("WARNING: No tuning_metric specified, using validation loss which is default")
        tuning_metric = "validation_loss"
    if runs_per_gpu is None:
        runs_per_gpu = 6
    time_s = master_config.pop('time_s', None)
    if time_s is None:
        time_s = 60*60
    name = master_config.get('run_group_name', None)
    print("Search space spec " + search_space_spec)
    search_space = {'master_config': master_config}
    if search_space_spec.startswith('Ours'):
        def accelerator_sampler(spec):
            max_steps = 20
            while True:
                num_update_steps = np.random.randint(1, max_steps+1)
                shanks_acceleration_order = np.random.randint(0, ((num_update_steps - 1)/2)+1)
                if shanks_acceleration_order <= (num_update_steps - 1)/2:
                    return {'num_update_steps': num_update_steps,
                            'acceleration_order': shanks_acceleration_order}

        search_space.update({'shanks_acceleration_dict': tune.sample_from(accelerator_sampler),
                             'opt.initial_damping': tune.loguniform(1e-8, 1),
                            })

        if not search_space_spec.endswith("Adaptive"):
            search_space.update({'opt.learning_rate': tune.loguniform(1e-3, 5),
                                 'opt.momentum': tune.loguniform(1e-3, 0.95),
                                 'opt.initial_damping': tune.loguniform(1e-8, 1),
                                })

        # TODO: Both of the below blocks to be refactored
        if search_space_spec.split('_')[-1] == 'OptimiseScaleFactor':
            search_space.update({'initial_scale_factor': tune.loguniform(10, 1000)})
        if search_space_spec.endswith("MaxEigenvalue"):
            search_space.update({'opt.scale_factor_kwargs.tolerance': tune.loguniform(1e-7, 1e-1),
                                 'opt.scale_factor_kwargs.max_num_steps': tune.randint(1, 200)})
    elif search_space_spec == 'SFNUnadaptive':
        search_space.update({'opt.learning_rate': tune.loguniform(1e-6, 1e-1),
                             'opt.momentum': tune.loguniform(1e-3, 0.95),
                             'opt.small_eig_learning_rate': tune.loguniform(1e-3, 1e1),
                             'opt.small_eigenvalue_threshold': tune.loguniform(1e-10, 1e-3)})
    elif search_space_spec == 'SGD':
        search_space.update({'opt.learning_rate': tune.loguniform(1e-6, 1e-1)})
    elif search_space_spec.startswith('Adam'):
        search_space.update({'opt.learning_rate': tune.loguniform(1e-6, 1e-0)})
        if not search_space_spec.endswith('Simple'):
            search_space.update({'opt.eps': tune.loguniform(1e-10, 1e1),
                                      'adam_one_minus_b1': tune.loguniform(1e-3, 1e-0),
                                      'adam_one_minus_b2': tune.loguniform(1e-4, 1e-0)})
    elif search_space_spec.startswith('KFAC'):
        search_space.update({'opt.initial_damping': tune.loguniform(1e-8, 1)})
        if search_space_spec.endswith('Kazuki'):
            search_space.update({'opt.learning_rate': tune.loguniform(1e-6, 1e1),
                                 'opt.momentum': tune.loguniform(1e-3, 0.95)})

    elif search_space_spec.startswith('SFNAdaptive'):
        search_space.update({'opt.initial_damping': tune.loguniform(1e-8, 10)})
    else:
        raise ValueError(search_space_spec)

    print(search_space)

    dataset = master_config['dataset']['name']
    if dataset in ('FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'UCI_Protein'):
        search_space['root.batch_size'] = tune.choice(
            (50, 100, 200, 400, 800, 1600, 3200))
    elif dataset in ('Kin8nm'):
        search_space['root.batch_size'] = tune.choice(
            (50, 100, 200, 400, 800, 1600))
    elif dataset not in ('UCI_Energy'):
        raise ValueError(dataset)
    tune_kwargs = dict(
        run_or_experiment=ray_tune_trainable,
        name=f"{tune_mode}_Time_{name}_{search_space_spec}",
        metric=tuning_metric,
        mode='min',
        stop=lambda trial_id, result: not np.isfinite(result[tuning_metric]),
        num_samples=num_configurations,
        resources_per_trial={'cpu': 1, 'gpu': 1/runs_per_gpu},
        local_dir=local_dir,
        log_to_file=True,
        config=search_space,
        keep_checkpoints_num=1,
        checkpoint_score_attr='timestamp',
        # resume=True
    )
    if tune_mode == 'ASHA':
        scheduler = tune.schedulers.ASHAScheduler(
            time_attr='time_total_s',
            max_t=time_s,
            grace_period=30)
        tune_kwargs['scheduler'] = scheduler
    elif tune_mode == 'BO':
        search_algorithm = tune.suggest.bayesopt.BayesOptSearch()
        tune_kwargs['search_alg'] = search_algorithm
    else:
        raise ValueError(tune_mode)

    analysis = tune.run(**tune_kwargs)
    best_config = analysis.get_best_config(tuning_metric, 'min')
    output_dir = os.path.join(local_dir, f"{tune_mode}_Time_{name}_{search_space_spec}")
    with open(os.path.join(output_dir, 'best_result.pkl'), 'wb') as pickle_file:
        pickle.dump(analysis.best_result, pickle_file)
    with open(os.path.join(output_dir, 'best_config.pkl'), 'wb') as pickle_file:
        pickle.dump(best_config, pickle_file)


if __name__ == '__main__':
    master_config = config.load_config()
    ray_tune_run(num_configurations=100,
                 tune_mode='ASHA',
                 master_config=master_config)
