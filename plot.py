import collections
import json
import os
import shutil
import pandas as pd
import torch as to
import numpy as np
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from contextlib import contextmanager

import util
import play


SOLARIZED = dict(
    base03='#002b36',
    base02='#073642',
    base01='#586e75',
    base00='#657b83',
    base0='#839496',
    base1='#93a1a1',
    base2='#eee8d5',
    base3='#fdf6e3',
    yellow='#b58900',
    orange='#cb4b16',
    red='#dc322f',
    magenta='#d33682',
    violet='#6c71c4',
    blue='#268bd2',
    cyan='#2aa198',
    green='#859900')
SOLARIZED_CMAP = LinearSegmentedColormap.from_list(
    'solarized', (SOLARIZED['base1'], SOLARIZED['base2'], SOLARIZED['base3']))
PLOT_ORDER = ('SGD', 'Adam', 'KFAC (Kazuki)', 'KFAC (DeepMind)', 'LBFGS', 'SFN', 'Adaptive Exact SFN', 'Ours (Damped, Unadaptive)', 'Ours (SF)', 'Ours (Adaptive)', 'Fixed-Momentum Ours',
              '1 Step', '2 Steps', '3 Steps', '4 Steps', '5 Steps', '6 Steps', '7 Steps', '8 Steps', '9 Steps', '10 Steps', '15 Steps', '20 Steps', '25 Steps', '50 Steps', '100 Steps')

LOSS_LIMITS = dict(
    Optimal_UCI_Energy_FullBatch=(2e-4, 1),
    Optimal_FashionMNIST_TunedBatch=(2e-1, 3),
    Optimal_SVHN_TunedBatch=(2e-2, 1e2),
    Optimal_CIFAR10_TunedBatch=(5e-5, 1e2),
    Optimal_FashionMNIST_FixedBatch=(1.5e-2, 3),
    Optimal_SVHN_FixedBatch=(1e-6, 1e1),
    Optimal_CIFAR10_FixedBatch=(5e-8, 1e4))
ALGORITHM_COLOURS = collections.defaultdict(lambda: None)
ALGORITHM_COLOURS.update({
    'SGD': SOLARIZED['blue'],
    'Adam': SOLARIZED['cyan'],
    'KFAC (Kazuki)': SOLARIZED['green'],
    'KFAC (DeepMind)': SOLARIZED['yellow'],
    'Ours (Damped, Unadaptive)': SOLARIZED['orange'],
    'Ours (SF)': SOLARIZED['magenta'],
    'LBFGS': SOLARIZED['violet'],
    'SFN': SOLARIZED['orange'],
    'Adaptive Exact SFN': '#000000',
    'Ours (Adaptive)': SOLARIZED['magenta'],
    'Fixed-Momentum Ours': '#666666',
    'Newton': SOLARIZED['magenta']})


def paper_theme(wrapped_function):
    """Override matplotlib to use a standard theme."""
    def wrapper(*args, **kwargs):
        with plt.style.context('Solarize_Light2'):
            plt.gcf().set_facecolor('white')
            return wrapped_function(*args, **kwargs)
    return wrapper


@contextmanager
def no_plt_show():
    def close_all():
        plt.cla()
        plt.clf()
        plt.close()
    original_show = plt.show
    plt.show = close_all
    yield
    plt.show = original_show


def plot_series_approximation_evolution():
    exact_updates = to.load('exact_update.pt').detach().cpu()
    series_updates = [t.detach().cpu()
                      for t in to.load('approx_update_sums.pt')]

    exact_norm = to.linalg.norm(exact_updates)
    magnitude_errors = [exact_norm - to.linalg.norm(s)
                        for s in series_updates]
    cosine_similarities = [exact_updates.dot(s)
                           / (exact_norm * to.linalg.norm(s))
                           for s in series_updates]

    plt.plot(cosine_similarities)
    plt.show()

    plt.plot(magnitude_errors)
    plt.show()


def plot_hyperband_dashboard(log_directory,
                             metric='validation_loss'):
    analysis = ExperimentAnalysis(log_directory)
    dataframes = analysis.trial_dataframes
    metric_data = pd.concat(dataframes)[['training_iteration', metric]]
    metric_data.index.names = ('directory', 'row_id')
    all_configs = analysis.get_all_configs()
    for config in all_configs.values():
        config.pop('master_config')
        nested_values = {key: value
                         for key, value in config.items()
                         if isinstance(value, dict)}
        for subkey, subvalue in nested_values.items():
            config.pop(subkey)
            config.update(subvalue)
    hyperparameters = config.keys()
    all_configs = pd.DataFrame.from_dict(all_configs, orient='index')
    all_data = pd.merge(metric_data, all_configs, left_on='directory', right_index=True)
    max_iterations = all_data['training_iteration'].groupby('directory').max()
    all_data = pd.merge(all_data, max_iterations.rename('max_iterations'), left_on='directory', right_on='directory')

    fig = plt.gcf()
    master_grid = fig.add_gridspec(2, 1, height_ratios=(20, 1))
    scatter_grid = master_grid[0].subgridspec(1, len(hyperparameters))
    controls_grid = master_grid[1].subgridspec(1, 3, width_ratios=(15, 1, 1))

    scatter_axs = []
    for hyperparameter_id, hyperparameter in enumerate(hyperparameters):
        subplot_ax = fig.add_subplot(scatter_grid[0, hyperparameter_id])
        scatter_axs.append(subplot_ax)

    slider_ax = fig.add_subplot(controls_grid[0, 0])
    minus_button_ax = fig.add_subplot(controls_grid[0, 1])
    plus_button_ax = fig.add_subplot(controls_grid[0, 2])
    slider = Slider(slider_ax,
                    'training_iteration',
                    valmin=1,
                    valmax=all_data['training_iteration'].max(),
                    valinit=1,
                    valstep=1)
    plus_button = Button(plus_button_ax, '+')
    plus_button.on_clicked(lambda *_: slider.set_val(slider.val + 1))
    minus_button = Button(minus_button_ax, '–')
    minus_button.on_clicked(lambda *_: slider.set_val(slider.val - 1))

    def update_slider(value):
        filtered_data = all_data[all_data['training_iteration'] == value]
        finalised_mask = (filtered_data['max_iterations'] == value)
        for hyperparameter, scatter_ax in zip(hyperparameters, scatter_axs):
            scatter_ax.cla()
            scatter_ax.scatter(filtered_data[hyperparameter],
                               filtered_data[metric])
            scatter_ax.scatter(filtered_data[hyperparameter][finalised_mask],
                               filtered_data[metric][finalised_mask])
            scatter_ax.set_xlabel(hyperparameter)
            scatter_ax.set_ylabel(metric)
            scatter_ax.set_yscale('log')

    slider.on_changed(update_slider)
    update_slider(1)

    plt.show()


def plot_hyperband_evolution(root_directory,
                             metric='validation_loss'):
    for log_directory in os.scandir(root_directory):
        print(f"Parsing {log_directory.path}...")
        analysis = ExperimentAnalysis(log_directory)
        if not analysis.trials and not os.path.exists(
                os.path.join(log_directory, 'experiment_log_backup.json')):
            print(f"Editing log paths for {log_directory.path}")
            # Backup experiment file
            experiment_log_path = ExperimentAnalysis._get_latest_checkpoint(
                None, os.path.expanduser(log_directory))
            assert len(experiment_log_path) == 1
            experiment_log_path = experiment_log_path[0]
            shutil.copy(experiment_log_path,
                        os.path.join(log_directory,
                                     'experiment_log_backup.json'))
            with open(experiment_log_path, 'r') as experiment_log_file:
                experiment_log_data = json.load(experiment_log_file)
            # Change all checkpoints['local_dir'] to log_directory
            # Change all checkpoints['logdir'] to log_directory/<existing_name>
            original_directory = json.loads(
                experiment_log_data['checkpoints'][0])['local_dir']
            for checkpoint_id, checkpoint_str in enumerate(
                    experiment_log_data['checkpoints']):
                experiment_log_data['checkpoints'][checkpoint_id] = checkpoint_str.replace(
                    original_directory, log_directory.path)
            experiment_log_data['runner_data']['_local_checkpoint_dir'] = log_directory.path
            experiment_log_data['runner_data']['checkpoint_file'] = (
                experiment_log_data['runner_data']['checkpoint_file'].replace(
                    original_directory, log_directory.path))
            with open(experiment_log_path, 'w') as experiment_log_file:
                json.dump(experiment_log_data, experiment_log_file)
            analysis = ExperimentAnalysis(log_directory)

        dataframes = analysis.trial_dataframes
        metric_data = pd.concat(dataframes)[['timestamp', metric]].sort_values(by='timestamp')

        runtimes = metric_data['timestamp'] - metric_data['timestamp'][0]
        best_evolution = metric_data[metric].cummin()
        plt.plot(runtimes, best_evolution, label=log_directory.name)
    plt.yscale('log')
    plt.xlabel('Runtime (s)')
    plt.ylabel(f'Best {metric}')
    plt.legend()
    plt.show()


def plot_optimal_results(root_directory, metric='Loss/Test', multirun=True):
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_kwargs = {'logy': True,
                   'alpha': 0.4}
    if not multirun:
        colours = [None]
        directory_iter = [root_directory]
    else:
        directory_iter = os.scandir(root_directory)
    for algorithm_directory, colour in zip(directory_iter, colours):
        algorithm_directory = Path(algorithm_directory)
        if not algorithm_directory.is_dir():
            continue
        if colour:
            plot_kwargs['color'] = colour
        print(f"Parsing {algorithm_directory}...")
        data = util.pandas_from_tensorboard(algorithm_directory)
        (data
         [data['tag'] == metric]
         .set_index('step')
         .groupby('dir_name')
         ['value']
         .plot(**plot_kwargs))
    plt.ylabel(metric)
    plt.title(root_directory)
    plt.show()


@paper_theme
def plot_optimal_envelopes(root_directory,
                           metric='Loss/Test',
                           aggregation=np.median,
                           num_repetitions=100,
                           wall_time=False,
                           log_x_axis=True,
                           num_sig_figs=3):
    try:
        subdirectories = sorted(os.scandir(root_directory),
                                key=lambda directory: PLOT_ORDER.index(directory.name))
    except ValueError as exception:
        print(f"Some keys weren't able to be ordered: {exception}")
        subdirectories = os.scandir(root_directory)

    lines = []
    envelopes = []
    for algorithm_directory in subdirectories:
        if not algorithm_directory.is_dir():
            continue
        print(f"Parsing {algorithm_directory.path}...")
        data = util.pandas_from_tensorboard(algorithm_directory)
        if wall_time:
            pruned_data = data[data['tag'] == metric]
            # Remove largest wall_time entry, which is a post-optimiser step loss
            pruned_data = pruned_data[
                ~(pruned_data['wall_time'] ==
                  pruned_data.groupby('dir_name')['wall_time'].transform('max'))]
            time_deltas = pd.to_timedelta(
                pruned_data
                .groupby('dir_name')
                ['wall_time']
                .apply(
                    lambda group: util.round_sig_figs(group - group.min(), num_sig_figs)),
                unit='s')
            data_series = (pruned_data
                           .assign(wall_time=time_deltas)
                           .pivot_table(
                               index='wall_time',
                               columns='dir_name',
                               values='value')
                           .interpolate(
                               method='time'))
            data_series = data_series.set_index(
                data_series.index.total_seconds())
        else:
            data_series = data[data['tag'] == metric].pivot(
                index='step', columns='dir_name', values='value')
        median_metric, median_error = util.bootstrap_aggregate(
            data_series, aggregation, num_repetitions)
        colour_kwargs = dict(color=ALGORITHM_COLOURS[algorithm_directory.name])
        if colour_kwargs['color'] is None:
            colour_kwargs.pop('color')
        lines.append(
            median_metric.plot(label=algorithm_directory.name,
                               **colour_kwargs)
            .get_lines()[-1])
        envelopes.append(
            plt.fill_between(median_error.index,
                             median_metric - median_error,
                             median_metric + median_error,
                             alpha=0.4,
                             **colour_kwargs))
    legend = plt.legend()
    legend_mapping = dict()
    for legend_line, plot_line, plot_envelope in zip(legend.get_lines(), lines, envelopes):
        legend_line.set_picker(5)
        legend_mapping[legend_line] = (plot_line, plot_envelope)

    def legend_toggle(event):
        legend_line = event.artist
        plot_line, plot_envelope = legend_mapping[legend_line]
        new_visibility = not plot_line.get_visible()
        plot_line.set_visible(new_visibility)
        plot_envelope.set_visible(new_visibility)
        if new_visibility:
            legend_line.set_alpha(1.0)
        else:
            legend_line.set_alpha(0.2)
        plt.gcf().canvas.draw()

    plt.gcf().canvas.mpl_connect('pick_event', legend_toggle)

    plt.yscale('log')
    if wall_time:
        plt.xlabel('Runtime (s)')
    if metric == 'Loss/Training':
        plt.ylabel('Training Loss')
    elif metric == 'Loss/Test':
        plt.ylabel('Test Loss')
    else:
        plt.ylabel(metric)
    plt.title(root_directory)
    # plt.show()
    if log_x_axis:
        plt.xscale('log')
        plt.xlim(1e-2, None)
    save_name = Path(root_directory).name
    if save_name in LOSS_LIMITS:
        plt.ylim(*LOSS_LIMITS[save_name])
    plt.savefig(f'./plots_jax/{save_name}{"_Time" if wall_time else ""}_{metric.replace("/", "_")}.pdf',
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


@paper_theme
def plot_optimiser_comparison():
    """Plot a summary graphic showing the updates computed by various
    optimisers on a sample function.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, 50),
                       np.linspace(-1, 1, 50))
    t = to.linspace(-1, 1, 50)
    half_t = to.linspace(0, 1, 50)
    z = play.f(np.stack((x, y)))

    ax = plt.axes(projection='3d', computed_zorder=False)
    ax.elev = 45
    ax.plot_surface(x, y, z, alpha=1.0, zorder=-1, cmap=SOLARIZED_CMAP)

    start_pt = to.tensor([-0.75, -0.75])
    grad_update_pt = play.grad_update(*start_pt, lr=0.1)
    newton_update_pt = play.newton_update(*start_pt)
    flip_update_pt = play.flip_update(*start_pt)
    series_update_pts = play.series_updates(*start_pt, num_steps=5)

    update_chain = (start_pt, *series_update_pts)
    ax.text(*(series_update_pts[-1] + to.tensor([0.05, 0.04])),
            play.f(series_update_pts[-1]),
            r"$\mathbf{x}^{*}_{\sf Ours}$")
    for last_pt, new_pt in zip(update_chain[:-1], update_chain[1:]):
        update_vector = new_pt - last_pt
        update_x = last_pt[0] + half_t*update_vector[0]
        update_y = last_pt[1] + half_t*update_vector[1]
        update_z = play.f(to.stack((update_x, update_y)))
        ax.plot(update_x, update_y, update_z, c=ALGORITHM_COLOURS['Ours'])
        ax.scatter(*new_pt, play.f(new_pt), c=ALGORITHM_COLOURS['Ours'], zorder=3)
        # Manually plot arrow projected onto surface
        mid_index = len(update_x) // 2
        midpoint = to.stack((update_x[mid_index],
                             update_y[mid_index],
                             update_z[mid_index]))
        mid_vector = to.stack(
            [dimension[mid_index+1] - dimension[mid_index-1]
             for dimension in (update_x, update_y, update_z)])
        mid_vector /= mid_vector.norm()
        mid_grad = play.grad_f(*midpoint[:2])
        mid_ortho = to.tensor([mid_grad[1], -mid_grad[0], 0])
        mid_ortho /= mid_ortho.norm()
        # mid_ortho = mid_vector @ to.tensor([[0, 1],
        #                                     [-1, 0]])
        half_width = 0.03
        half_length = 0.045
        arrow_points = [
            midpoint + half_length*mid_vector,
            midpoint - half_length*mid_vector + half_width*mid_ortho,
            midpoint - half_length*mid_vector - half_width*mid_ortho]
        ax.add_collection3d(
            Poly3DCollection(arrow_points,
                             color=ALGORITHM_COLOURS['Ours']))

    const_hess = play.hess_f(0, 0)
    eigenvectors = to.linalg.eigh(const_hess).eigenvectors

    ax.scatter(*start_pt, play.f(start_pt), c=SOLARIZED['base02'], zorder=3)
    ax.text(*(start_pt + to.tensor([0.05, 0.05])), play.f(start_pt), r"$\mathbf{x}$")

    point_labels = (r"$\mathbf{x}^{*}_{\sf SGD}$",
                    r"$\mathbf{x}^{*}_{\sf Newton}$",
                    r"$\mathbf{x}^{*}_{\sf Exact\ SFN}$")

    for vector, colour in zip(
            eigenvectors.T,
            (SOLARIZED['violet'], SOLARIZED['cyan'])):
        eig_x = t*vector[0]
        eig_y = t*vector[1]
        eig_z = play.f(to.stack((eig_x, eig_y)))
        ax.plot(eig_x, eig_y, eig_z, c=colour, linewidth=0.75)
        half_width = 0.03
        half_length = 0.045
        parallel_vector = to.tensor([*vector, 0])
        perpendicular_vector = to.tensor([*(vector @ to.tensor([[0., 1.],
                                                                [-1., 0.]])),
                                          0])
        base_point = to.stack((eig_x[0], eig_y[0], eig_z[0]))
        endpoint = base_point + 0.2*parallel_vector
        if colour == SOLARIZED['violet']:
            label = r"$\lambda < 0$"
        else:
            label = r"$\lambda > 0$"
        ax.text(*(endpoint + to.tensor([0.05, 0.05, 0])), label)

    for update_pt, colour, label, offset in zip(
            (grad_update_pt, newton_update_pt, flip_update_pt),
            (ALGORITHM_COLOURS['SGD'], ALGORITHM_COLOURS['Newton'], ALGORITHM_COLOURS['Exact SFN']),
            point_labels,
            (to.tensor([0.05, 0.05]), to.tensor([0.05, 0.05]), to.tensor([0.1, -0.12]))):
        update_vector = update_pt - start_pt
        update_x = start_pt[0] + half_t*update_vector[0]
        update_y = start_pt[1] + half_t*update_vector[1]
        update_z = play.f(to.stack((update_x, update_y)))
        ax.plot(update_x, update_y, update_z, c=colour)
        ax.scatter(*update_pt, play.f(update_pt), c=colour, zorder=3)
        ax.text(*(update_pt + offset), play.f(update_pt), label)
        # Manually plot arrow projected onto surface
        mid_index = len(update_x) // 2
        midpoint = to.stack((update_x[mid_index],
                             update_y[mid_index],
                             update_z[mid_index]))
        mid_vector = to.stack(
            [dimension[mid_index+1] - dimension[mid_index-1]
             for dimension in (update_x, update_y, update_z)])
        mid_vector /= mid_vector.norm()
        mid_grad = play.grad_f(*midpoint[:2])
        mid_ortho = to.tensor([mid_grad[1], -mid_grad[0], 0])
        mid_ortho /= mid_ortho.norm()
        # mid_ortho = mid_vector @ to.tensor([[0, 1],
        #                                     [-1, 0]])
        half_width = 0.03
        half_length = 0.045
        arrow_points = [
            midpoint + half_length*mid_vector,
            midpoint - half_length*mid_vector + half_width*mid_ortho,
            midpoint - half_length*mid_vector - half_width*mid_ortho]
        ax.add_collection3d(
            Poly3DCollection(arrow_points,
                             color=colour))

    axis_crop = 0.9
    # ax.set_xlim(-axis_crop, axis_crop)
    # ax.set_ylim(-axis_crop, axis_crop)

    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Loss', rotation=90)
    ax.set_facecolor('white')
    # ax.xaxis.set_pane_color((1, 1, 1, 0))
    # ax.yaxis.set_pane_color((1, 1, 1, 0))
    # ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig('./plots_jax/comparison_3d.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    ax = plt.axes(aspect='equal')
    ax.contour(x, y, z, levels=30, zorder=-1, cmap=SOLARIZED_CMAP)

    ax.text(*(series_update_pts[-1] + to.tensor([0.03, 0.02])),
            r"$\mathbf{x}^{*}_{\sf Ours}$")
    for last_pt, new_pt in zip(update_chain[:-1], update_chain[1:]):
        ax.plot(*to.stack((last_pt, new_pt)).T, c=ALGORITHM_COLOURS['Ours'])
        ax.scatter(*new_pt, c=ALGORITHM_COLOURS['Ours'], zorder=3)
        midpoint = (last_pt + new_pt) / 2
        update_vector = new_pt - last_pt
        ax.arrow(*(midpoint - 0.1*update_vector),
                 *(0.085*update_vector),
                 color=ALGORITHM_COLOURS['Ours'],
                 head_width=0.03)

    for update_pt, colour, label, offset in zip(
            (grad_update_pt, newton_update_pt, flip_update_pt),
            (ALGORITHM_COLOURS['SGD'], ALGORITHM_COLOURS['Newton'], ALGORITHM_COLOURS['Exact SFN']),
            point_labels,
            (to.tensor([0.03, 0.03]), to.tensor([0.03, 0.03]), to.tensor([0.03, -0.05]))):
        ax.plot(*to.stack((start_pt, update_pt)).T, c=colour)
        ax.scatter(*update_pt, c=colour, zorder=3)
        ax.text(*(update_pt + offset), label)
        midpoint = (start_pt + update_pt) / 2
        update_vector = update_pt - start_pt
        ax.arrow(*(midpoint - 0.1*update_vector),
                 *(0.085*update_vector),
                 color=colour,
                 head_width=0.03)

    ax.scatter(*start_pt, c=SOLARIZED['base02'], zorder=3)
    ax.text(*(start_pt + to.tensor([-0.1, 0])),
            r"$\mathbf{x}$")

    for vector, colour in zip(
            eigenvectors.T,
            (SOLARIZED['violet'], SOLARIZED['cyan'])):
        ax.plot([-vector[0], vector[0]],
                [-vector[1], vector[1]],
                c=colour,
                linewidth=0.75)
        ax.arrow(*(-vector),
                 *(0.2*vector),
                 color=colour,
                 linewidth=0.75,
                 head_width=0.03)
        if colour == SOLARIZED['violet']:
            label = r"$\lambda < 0$"
        else:
            label = r"$\lambda > 0$"
        ax.text(*(-0.8*vector + 0.08*to.tensor([-vector[1], vector[0]])), label)

    projection_pt = start_pt.dot(eigenvectors.T[0]) * eigenvectors.T[0]
    ax.plot([start_pt[0], projection_pt[0]],
            [start_pt[1], projection_pt[1]],
            c=SOLARIZED['cyan'], linewidth=0.75, linestyle='--')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')
    plt.savefig('./plots_jax/comparison_2d.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


@paper_theme
def plot_accelerator_comparison():
    nums_accelerations = (1, 2)
    num_applications = max(nums_accelerations)
    null_start_lengths = {'Shanks': 2*num_applications,
                          'Shanks (Sablonnière)': 2*num_applications,
                          'VectorRho': 2*num_applications,
                          'TopologicalRho': 2*num_applications,
                          'Levin': num_applications+2,
                          'D2Transform': num_applications+2,
                          'Padé': 2}
    alphas = {1: 0.5,
              2: 1.0}
    colours = {'original': SOLARIZED['violet'],
               'Shanks': SOLARIZED['cyan'],
               'Levin': SOLARIZED['yellow'],
               'Padé': SOLARIZED['red'],
               'Shanks (Sablonnière)': SOLARIZED['green']}

    with no_plt_show():
        similarities, magnitudes, target_norm = play.hessian_investigation(
            dimension=100,
            mean=0,
            stdev=1,
            scale_factor=1000,
            num_update_steps=1000,
            nums_accelerations=nums_accelerations)
        for num_applications in nums_accelerations:
            for data_dict in (similarities, magnitudes):
                data_dict[f'Padé{num_applications}'] = data_dict.pop(f'Pade{num_applications}')
                data_dict[f'Shanks (Sablonnière-Modified){num_applications}'] = data_dict.pop(f'ShanksM{num_applications}')

    plt.axhline(1.0, c='k', label='Target')
    plt.plot(similarities['original'],
             label='Original',
             color=colours['original'])
    for accelerator in ('Shanks',
                        'Levin',
                        'Padé',
                        'Shanks (Sablonnière)'):
        for num_applications in nums_accelerations:
            plt.plot([*[float('nan')]*null_start_lengths[accelerator],
                      *similarities[f'{accelerator}{num_applications}']],
                     label=f'{num_applications}×{accelerator}',
                     color=colours[accelerator],
                     alpha=alphas[num_applications])
    plt.legend()
    plt.ylim(0, None)
    plt.xlabel('Series Steps')
    plt.ylabel('Cosine Similarity')
    plt.gcf().set_facecolor('white')
    plt.savefig('./plots/accelerator_similarity.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.axhline(target_norm, c='k', label='Target')
    plt.plot(magnitudes['original'],
             label='Original',
             color=colours['original'])
    for accelerator in ('Shanks',
                        'Levin',
                        'Padé',
                        'Shanks (Sablonnière)'):
        for num_applications in nums_accelerations:
            plt.plot([*[float('nan')]*null_start_lengths[accelerator],
                      *magnitudes[f'{accelerator}{num_applications}']],
                     label=f'{num_applications}×{accelerator}',
                     color=colours[accelerator],
                     alpha=alphas[num_applications])
    plt.legend()
    plt.ylim(0, target_norm+5)
    plt.xlabel('Series Steps')
    plt.ylabel('2-Norm')
    plt.gcf().set_facecolor('white')
    plt.savefig('./plots/accelerator_magnitude.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
