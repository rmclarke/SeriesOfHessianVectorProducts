from collections import defaultdict

import jax
import jax.numpy as jnp
import torch as to
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy

import util


def f(inputs):
    x = inputs[0]
    y = inputs[1]
    return (2*x + y) * (x - y) + 3*x*y


def grad_f(x, y):
    inputs = to.tensor([x, y], dtype=to.float)
    return to.autograd.functional.jacobian(f, inputs)


def hess_f(x, y):
    inputs = to.tensor([x, y], dtype=to.float)
    return to.autograd.functional.hessian(f, inputs)


def grad_update(x, y, lr=1):
    inputs = to.tensor([x, y], dtype=to.float)
    return inputs - lr*grad_f(*inputs)


def jacobian_approx_hess(grads, eps=1e-6):
    approx_hessian = 2 * to.outer(grads, grads)
    return approx_hessian + eps * to.eye(approx_hessian.shape[0])


def jacobian_update(x, y):
    inputs = to.tensor([x, y], dtype=to.float)
    grads = grad_f(*inputs)
    approx_hessian = jacobian_approx_hess(grads)
    return inputs - approx_hessian.inverse() @ grads


def newton_update(x, y):
    inputs = to.tensor([x, y], dtype=to.float)
    return inputs - hess_f(*inputs).inverse() @ grad_f(*inputs)


def unit_update(x, y):
    inputs = to.tensor([x, y], dtype=to.float)
    eigenvalues, eigenvectors = to.linalg.eigh(hess_f(x, y))
    eigenvalues[eigenvalues < 0] = 1
    unit_hessian = eigenvectors @ to.diag(eigenvalues) @ eigenvectors.T
    return inputs - unit_hessian.inverse() @ grad_f(*inputs)


def flip_update(x, y):
    inputs = to.tensor([x, y], dtype=to.float)
    eigenvalues, eigenvectors = to.linalg.eigh(hess_f(x, y))
    eigenvalues = eigenvalues.abs()
    flip_hessian = eigenvectors @ to.diag(eigenvalues) @ eigenvectors.T
    return inputs - flip_hessian.inverse() @ grad_f(*inputs)


def series_updates(x, y, num_steps):
    inputs = to.tensor([x, y], dtype=to.float)
    terms = [grad_f(x, y)]
    partial_sums = [grad_f(x, y)]
    scale_factor = 25.0
    hessian = hess_f(x, y)
    for step in range(1, num_steps):
        terms.append(
            (2*step * (2*step - 1) / (4 * step**2))
            * (terms[-1] - hessian @ hessian @ terms[-1] / scale_factor))
        partial_sums.append(partial_sums[-1] + terms[-1])
    return [inputs - scale_factor**(-0.5) * partial_sum
            for partial_sum in partial_sums]


def plot_3d():
    x, y = np.meshgrid(np.linspace(-1, 1, 50),
                       np.linspace(-1, 1, 50))
    z = f(np.stack((x, y)))

    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, alpha=0.5)

    start_pt = to.tensor([0.75, 0.75])
    grad_update_pt = grad_update(*start_pt, lr=0.1)
    newton_update_pt = newton_update(*start_pt)
    # unit_update_pt = unit_update(*start_pt)
    flip_update_pt = flip_update(*start_pt)

    ax.scatter(*start_pt, f(start_pt), c='k', label='Start')
    ax.scatter(*grad_update_pt, f(grad_update_pt), c='r', label='Grad')
    ax.scatter(*newton_update_pt, f(newton_update_pt), c='orange', label='Newton')
    # ax.scatter(*unit_update_pt, f(unit_update_pt), c='green', label='Unit')
    ax.scatter(*flip_update_pt, f(flip_update_pt), c='cyan', label='Flip')

    # plt.legend()
    plt.show()


def plot_2d():
    x, y = np.meshgrid(np.linspace(-5, 3, 100),
                       np.linspace(-3, 3, 100))
    z = f(np.stack((x, y)))

    ax = plt.axes()
    ax.contour(x, y, z, alpha=0.5, levels=30)

    start_pt = to.tensor([0.75, 1])
    grad_update_pt = grad_update(*start_pt)
    # jacobian_update_pt = jacobian_update(*start_pt)
    newton_update_pt = newton_update(*start_pt)
    # unit_update_pt = unit_update(*start_pt)
    flip_update_pt = flip_update(*start_pt)

    ax.scatter(*start_pt, c='k', label='Start')
    ax.scatter(*grad_update_pt, c='r', label='Grad')
    # ax.scatter(*jacobian_update_pt, c='blue', label='Jacobian')
    ax.scatter(*newton_update_pt, c='orange', label='Newton')
    # ax.scatter(*unit_update_pt, c='green', label='Unit')
    ax.scatter(*flip_update_pt, c='cyan', label='Flip')

    const_hess = hess_f(0, 0)
    eigenvectors = to.linalg.eigh(const_hess).eigenvectors
    # jacobian_eigenvectors = to.linalg.eigh(
    #     jacobian_approx_hess(
    #         grad_f(*start_pt))).eigenvectors
    for eigenvector_set in (eigenvectors, ): #jacobian_eigenvectors):
        for vector in eigenvector_set.T:
            ax.plot([0, -vector[0]],
                    [0, -vector[1]])
    ax.plot([grad_update_pt[0], grad_update_pt[0] - 4*eigenvectors[1][0]],
            [grad_update_pt[1], grad_update_pt[1] - 4*eigenvectors[1][1]],
            c='orange')
    ax.plot([start_pt[0], grad_update_pt[0]],
            [start_pt[1], grad_update_pt[1]],
            c='r')

    plt.legend()
    plt.show()


def hessian_investigation(dimension, mean, stdev, scale_factor, num_update_steps, nums_accelerations):
    # Construct a real, symmetric Hessian
    hessian = to.randn(dimension, dimension) * stdev + mean
    triangle_indices = to.triu_indices(dimension, dimension)
    hessian.T[triangle_indices[0],
              triangle_indices[1]] = hessian[triangle_indices[0],
                                             triangle_indices[1]]

    eigenvalues, eigenvectors = to.linalg.eigh(hessian)
    modified_eigenvalues = (eigenvalues**2) ** -0.5
    target_matrix = eigenvectors @ to.diag(modified_eigenvalues) @ eigenvectors.T

    matrix_sum = to.eye(dimension)
    matrix_cache = to.eye(dimension)
    for update_step in range(1, num_update_steps):
        combinatronic_factor = 2*update_step * (2*update_step - 1) / (update_step**2)
        matrix_cache = ((1/4)
                        * combinatronic_factor
                        * (to.eye(dimension)
                           - (1/scale_factor) * hessian @ hessian)) @ matrix_cache
        matrix_sum += matrix_cache
    matrix_sum *= scale_factor**(-0.5)

    gradient = to.randn(dimension) * stdev + mean
    target_vector = target_matrix @ gradient
    target_norm = to.linalg.norm(target_vector)

    vector_sum = gradient.clone()
    vector_cache = gradient.clone()
    vector_similarities = defaultdict(list)
    vector_magnitudes = defaultdict(list)
    old_sums = [vector_sum.clone()]
    for update_step in range(1, num_update_steps):
        combinatronic_factor = 2*update_step * (2*update_step - 1) / (update_step**2)
        vector_cache = ((1/4)
                        * combinatronic_factor
                        * (vector_cache
                           - (1/scale_factor) * hessian @ hessian @ vector_cache))
        vector_sum += vector_cache
        old_sums.append(vector_sum.clone())

        for num_applications in nums_accelerations:
            if update_step >= 2*num_applications:
                accelerated_vector = util.compute_epsilon_acceleration(old_sums, num_applications)
                vector_similarities[f'Shanks{num_applications}'].append(
                    util.cosine_similarity(target_vector,
                                           accelerated_vector * scale_factor**(-0.5)))
                vector_magnitudes[f'Shanks{num_applications}'].append(
                    to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))
            if update_step >= 2*num_applications:
                accelerated_vector = util.compute_epsilon_acceleration(old_sums, num_applications, modifier='sablonniere')
                vector_similarities[f'ShanksM{num_applications}'].append(
                    util.cosine_similarity(target_vector,
                                           accelerated_vector * scale_factor**(-0.5)))
                vector_magnitudes[f'ShanksM{num_applications}'].append(
                    to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))
            if update_step >= 2*num_applications:
                accelerated_vector = util.compute_epsilon_acceleration(old_sums, num_applications, modifier='vector_rho')
                vector_similarities[f'VectorRho{num_applications}'].append(
                    util.cosine_similarity(target_vector,
                                           accelerated_vector * scale_factor**(-0.5)))
                vector_magnitudes[f'VectorRho{num_applications}'].append(
                    to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))
            # if update_step >= 2*num_applications:
            #     accelerated_vector = util.compute_epsilon_acceleration(old_sums, num_applications, modifier='topological_rho')
            #     vector_similarities[f'TopologicalRho{num_applications}'].append(
            #         util.cosine_similarity(target_vector,
            #                                accelerated_vector * scale_factor**(-0.5)))
            #     vector_magnitudes[f'TopologicalRho{num_applications}'].append(
            #         to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))
            if update_step >= (num_applications + 2):
                accelerated_vector = util.compute_levin_acceleration(old_sums, num_applications)
                vector_similarities[f'Levin{num_applications}'].append(
                    util.cosine_similarity(target_vector,
                                           accelerated_vector * scale_factor**(-0.5)))
                vector_magnitudes[f'Levin{num_applications}'].append(
                    to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))
            # if update_step >= (num_applications + 2):
            #     accelerated_vector = util.compute_levin_acceleration(old_sums, num_applications, d_transform_order=2)
            #     vector_similarities[f'D2Transform{num_applications}'].append(
            #         util.cosine_similarity(target_vector,
            #                                accelerated_vector * scale_factor**(-0.5)))
            #     vector_magnitudes[f'D2Transform{num_applications}'].append(
            #         to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))
            if update_step >= 2:
                accelerated_vector = util.compute_pade_acceleration(old_sums)
                vector_similarities[f'Pade{num_applications}'].append(
                    util.cosine_similarity(target_vector,
                                           accelerated_vector * scale_factor**(-0.5)))
                vector_magnitudes[f'Pade{num_applications}'].append(
                    to.linalg.norm(accelerated_vector * scale_factor**(-0.5)))

        vector_similarities['original'].append(
            util.cosine_similarity(target_vector,
                                   vector_sum * scale_factor**(-0.5)))
        vector_magnitudes['original'].append(
            to.linalg.norm(vector_sum * scale_factor**(-0.5)))
    vector_sum *= scale_factor**(-0.5)

    null_start_lengths = {'Shanks': 2*num_applications,
                          'ShanksM': 2*num_applications,
                          'VectorRho': 2*num_applications,
                          'TopologicalRho': 2*num_applications,
                          'Levin': num_applications+2,
                          'D2Transform': num_applications+2,
                          'Pade': 2}

    _, axes = plt.subplots(1, 2)
    axes[0].set(title="Cosine Similarities")
    axes[0].axhline(1.0, c='k')
    axes[0].plot(vector_similarities['original'])

    axes[1].set(title="Magnitudes")
    axes[1].axhline(target_norm, c='k', label='Target')
    axes[1].plot(vector_magnitudes['original'], label='Original')

    for accelerator in ('Shanks',
                        'Levin',
                        'Pade',
                        'ShanksM',
                        # 'D2Transform',
                        # 'VectorRho',
                        # 'TopologicalRho',
                        ):
        for num_applications in nums_accelerations:
            axes[0].plot([*[float('nan')]*null_start_lengths[accelerator],
                          *vector_similarities[f'{accelerator}{num_applications}']])
            axes[1].plot([*[float('nan')]*null_start_lengths[accelerator],
                          *vector_magnitudes[f'{accelerator}{num_applications}']],
                         label=f'{num_applications}-{accelerator}')

    axes[1].legend()
    plt.show()
    # return vector_similarities, vector_magnitudes, target_norm
    # return target_matrix, matrix_sum


def precision_eigenvalue_coefficient(n, exp_scaling, centre=1):
    term_1 = ((-1/4)**n) * (np.math.factorial(2*n)/np.math.factorial(n)) * (centre**(-n-0.5))
    term_3 = (1 - centre) * ((-exp_scaling)**n) * np.exp(-exp_scaling*centre)
    term_4 = -n * ((-exp_scaling)**(n-1)) * np.exp(-exp_scaling*centre)

    term_2 = 0
    for k in range(0, n+1):
        term_2 += (- scipy.special.binom(n, k)
                   * ((-1/4)**k)
                   * (np.math.factorial(2*k)/np.math.factorial(k))
                   * (centre**(-k-0.5))
                   * ((-exp_scaling)**(n-k))
                   * np.exp(-exp_scaling*centre))

    return (term_1 + term_2 + term_3 + term_4) / np.math.factorial(n)


def kronecker_product_sum_plot(dimension=10, num_samples=100):

    fig = plt.gcf()
    grid = fig.add_gridspec(3, 3, width_ratios=(15, 15, 1), height_ratios=(15, 15, 1))
    avg_full_ax = fig.add_subplot(grid[0, 0])
    prod_avg_ax = fig.add_subplot(grid[0, 1])
    error_ax = fig.add_subplot(grid[1, 0])
    colourbar_ax = fig.add_subplot(grid[0:2, 2])

    redraw_button_ax = fig.add_subplot(grid[2, 2])
    redraw_button = Button(redraw_button_ax, 'R')
    redraw_button.on_clicked(lambda *_: redraw())

    def redraw():
        avg_full_ax.clear()
        prod_avg_ax.clear()
        error_ax.clear()
        colourbar_ax.clear()

        # left_factors = 10 * np.random.randn(num_samples, dimension, dimension)
        # right_factors = 10 * np.random.randn(num_samples, dimension, dimension)
        left_factors = np.stack(
            [np.outer(first, second)
             for first, second in zip(np.random.randn(num_samples, dimension),
                                      np.random.randn(num_samples, dimension))])
        right_factors = np.stack(
            [np.outer(first, second)
             for first, second in zip(np.random.randn(num_samples, dimension),
                                      np.random.randn(num_samples, dimension))])
        full_matrices = np.stack(
            [np.kron(left, right)
             for left, right in zip(left_factors, right_factors)])

        avg_full_matrices = full_matrices.mean(axis=0)
        avg_left_factors = left_factors.mean(axis=0)
        avg_right_factors = right_factors.mean(axis=0)

        prod_avg_factors = np.kron(avg_left_factors, avg_right_factors)

        error_matrix = prod_avg_factors - avg_full_matrices

        val_limit = max(np.abs(matrix).max()
                        for matrix in (avg_full_matrices,
                                       prod_avg_factors,
                                       error_matrix))
        min_val = -val_limit
        max_val = val_limit

        full_img = avg_full_ax.imshow(avg_full_matrices, vmin=min_val, vmax=max_val)
        avg_full_ax.set_title("Expectation of Kronecker Product")

        prod_avg_ax.imshow(prod_avg_factors, vmin=min_val, vmax=max_val)
        prod_avg_ax.set_title("Kronecker Product of Expectations")

        error_ax.imshow(error_matrix, vmin=min_val, vmax=max_val)
        error_ax.set_title("Error in Approximation")

        plt.colorbar(full_img, cax=colourbar_ax)

        plt.draw()

    redraw()
    plt.show()


def learn_kronecker_factor_corrections(dimension=10,
                                       num_samples=150,
                                       num_correction_iterations=1,
                                       num_optimisation_iterations=200,
                                       lr=0.001,
                                       damping=1e-4):
    left_vectors = np.random.randn(num_samples, dimension)
    right_vectors = np.random.randn(num_samples, dimension)

    left_factors = np.stack(
        [np.outer(vector, vector)
         for vector in left_vectors])
    right_factors = np.stack(
        [np.outer(vector, vector)
            for vector in right_vectors])

    full_matrices = np.stack(
        [np.kron(left, right)
            for left, right in zip(left_factors, right_factors)])

    avg_full_matrices = full_matrices.mean(axis=0) + damping*jnp.eye(dimension**2)
    avg_left_factors = left_factors.mean(axis=0)
    avg_right_factors = right_factors.mean(axis=0)

    def approx_inv_full_loss(full_matrix, left_factor, right_factor, left_corr, right_corr, gradient):
        approx_matrix = jnp.kron(left_factor + left_corr,
                                 right_factor + right_corr) + damping*jnp.eye(full_matrix.shape[0])
        return jnp.linalg.norm(
            jnp.linalg.solve(approx_matrix, full_matrix) @ gradient
            - gradient,
            ord=2)

    def mat_inv_full_loss(full_matrix, approx_matrix, gradient):
        approx_matrix = approx_matrix + damping*jnp.eye(full_matrix.shape[0])
        return jnp.linalg.norm(
            jnp.linalg.solve(approx_matrix, full_matrix) @ gradient
            - gradient,
            ord=2)

    def full_approx_inv_loss(full_matrix, left_factor, right_factor, left_corr, right_corr, gradient):
        approx_matrix = jnp.kron(left_factor + left_corr,
                                 right_factor + right_corr) + damping*jnp.eye(full_matrix.shape[0])
        return jnp.linalg.norm(
            full_matrix @ jnp.linalg.solve(approx_matrix, gradient)
            - gradient,
            ord=2)

    def full_approx_mat_loss(full_matrix, approx_matrix, gradient):
        approx_matrix = approx_matrix + damping*jnp.eye(full_matrix.shape[0])
        return jnp.linalg.norm(
            full_matrix @ jnp.linalg.solve(approx_matrix, gradient)
            - gradient,
            ord=2)

    def fullg_approxg_loss(full_matrix, left_factor, right_factor, left_corr, right_corr, gradient):
        approx_matrix = jnp.kron(left_factor + left_corr,
                                 right_factor + right_corr) + damping*jnp.eye(full_matrix.shape[0])
        return jnp.linalg.norm(
            full_matrix @ gradient - approx_matrix @ gradient,
            ord=2)

    def fullg_matg_loss(full_matrix, approx_matrix, gradient):
        approx_matrix = approx_matrix + damping*jnp.eye(full_matrix.shape[0])
        return jnp.linalg.norm(
            full_matrix @ gradient - approx_matrix @ gradient,
            ord=2)

    hessian = avg_full_matrices
    linear_coeff = np.random.randn(dimension**2)
    def model(data):
        return (0.5 * data.T @ hessian @ data) + (data.T @ linear_coeff)

    model_gradient = jax.grad(model)

    initial_point = np.random.randn(dimension**2)
    initial_value = model(initial_point)
    initial_grad = model_gradient(initial_point)

    correction_grad_functions = dict(
        Ainv_F_g=jax.grad(approx_inv_full_loss, argnums=(3, 4)),
        F_Ainv_g=jax.grad(full_approx_inv_loss, argnums=(3, 4)),
        Fg_Ag=jax.grad(fullg_approxg_loss, argnums=(3, 4)))
    matrix_grad_functions = dict(
        Minv_F_g=jax.grad(mat_inv_full_loss, argnums=(1,)),
        F_Minv_g=jax.grad(full_approx_mat_loss, argnums=(1,)),
        Fg_Mg=jax.grad(fullg_matg_loss, argnums=(1,)))

    fixed_corrections = {}
    for loss_name, grad_function in correction_grad_functions.items():
        fixed_corrections[loss_name] = [jnp.zeros_like(avg_left_factors),
                                        jnp.zeros_like(avg_right_factors)]
        for iteration in range(num_correction_iterations * num_optimisation_iterations):
            gradient = grad_function(avg_full_matrices,
                                     avg_left_factors,
                                     avg_right_factors,
                                     *fixed_corrections[loss_name],
                                     initial_grad)
            for (correction_id, correction), subgradient in zip(
                    enumerate(fixed_corrections[loss_name]), gradient):
                fixed_corrections[loss_name][correction_id] = correction - lr * subgradient

    fixed_matrices = {}
    for loss_name, grad_function in matrix_grad_functions.items():
        fixed_matrices[loss_name] = jnp.eye(avg_full_matrices.shape[0])
        for iteration in range(num_correction_iterations * num_optimisation_iterations):
            gradient = grad_function(avg_full_matrices,
                                     fixed_matrices[loss_name],
                                     initial_grad)
            fixed_matrices[loss_name] = fixed_matrices[loss_name] - lr * gradient[0]

    model_values = {}
    for algorithm in ('SGD (LR=0.001)',
                      # 'Gradient Descent + Line Search',
                      'Adam (defaults)',
                      # 'Diagonal Curvature + Line Search',
                      'KFAC Curvature',
                      'KFAC Curvature (constant correction, Ainv_F_g)',
                      'KFAC Curvature (constant correction, F_Ainv_g)',
                      'KFAC Curvature (constant correction, Fg_Ag)',
                      'KFAC Curvature (dynamic correction, Ainv_F_g)',
                      'KFAC Curvature (dynamic correction, F_Ainv_g)',
                      'KFAC Curvature (dynamic correction, Fg_Ag)',
                      # 'Learned Matrix LR (constant, Minv_F_g)',
                      # 'Learned Matrix LR (constant, F_Minv_g)',
                      # 'Learned Matrix LR (constant, Fg_Mg)',
                      'Learned Matrix LR (dynamic, Minv_F_g)',
                      'Learned Matrix LR (dynamic, F_Minv_g)',
                      'Learned Matrix LR (dynamic, Fg_Mg)',
                      # 'Low-Rank Moore-Penrose',
                      # 'Low-Rank SGD-Fallback',
                      # 'Newton + Line Search',
                      'Newton',
                      ):
        point = initial_point
        model_values = [initial_value]
        left_correction = jnp.zeros_like(avg_left_factors)
        right_correction = jnp.zeros_like(avg_right_factors)
        matrix_lr = jnp.eye(avg_full_matrices.shape[0])

        for iteration in range(num_optimisation_iterations):
            gradient = model_gradient(point)
            step_size = 'line_search'
            if algorithm == 'KFAC Curvature':
                direction = -jnp.linalg.lstsq(
                    jnp.kron(avg_left_factors, avg_right_factors),
                    gradient)[0]
            elif algorithm.startswith('KFAC Curvature (constant correction'):
                loss_key = algorithm.split(' ')[4][:-1]
                direction = -jnp.linalg.lstsq(
                    jnp.kron(avg_left_factors + fixed_corrections[loss_key][0],
                             avg_right_factors + fixed_corrections[loss_key][1]),
                    gradient)[0]
            elif algorithm.startswith('KFAC Curvature (dynamic correction'):
                loss_key = algorithm.split(' ')[4][:-1]
                for correction_iteration in range(num_correction_iterations):
                    correction_gradient = correction_grad_functions[loss_key](
                        avg_full_matrices,
                        avg_left_factors,
                        avg_right_factors,
                        left_correction,
                        right_correction,
                        gradient)
                    left_correction -= lr * correction_gradient[0]
                    right_correction -= lr * correction_gradient[1]
                direction = -jnp.linalg.lstsq(
                    jnp.kron(avg_left_factors + left_correction,
                             avg_right_factors + right_correction),
                    gradient)[0]
            elif algorithm.startswith('Learned Matrix LR (constant'):
                loss_key = algorithm.split(' ')[4][:-1]
                direction = -jnp.linalg.lstsq(fixed_matrices[loss_key], gradient)[0]
                step_size = 1
            elif algorithm.startswith('Learned Matrix LR (dynamic'):
                loss_key = algorithm.split(' ')[4][:-1]
                for correction_iteration in range(num_correction_iterations):
                    correction_gradient = matrix_grad_functions[loss_key](
                        avg_full_matrices,
                        matrix_lr,
                        gradient)
                    matrix_lr -= lr * correction_gradient[0]
                direction = -jnp.linalg.lstsq(matrix_lr, gradient)[0]
                step_size = 1
            elif algorithm == 'Diagonal Curvature + Line Search':
                direction = -gradient / jnp.diag(hessian)
            elif algorithm == 'Gradient Descent + Line Search':
                direction = -gradient
            elif algorithm == 'Newton + Line Search':
                direction = -jnp.linalg.lstsq(hessian, gradient)[0]
            elif algorithm == 'Newton':
                direction = -jnp.linalg.lstsq(hessian, gradient)[0]
                step_size = 1
            elif algorithm == 'SGD (LR=0.001)':
                direction = -gradient
                step_size = 0.001
            elif algorithm == 'Adam (defaults)':
                if iteration == 0:
                    eps = 1e-8
                    b1 = 0.9
                    b2 = 0.999
                    m = 0
                    v = 0
                m = b1*m + (1-b1)*gradient
                v = b2*v + (1-b2)*gradient**2
                m_hat = m / (1 - b1**(iteration+1))
                v_hat = v / (1 - b2**(iteration+1))
                direction = -m_hat / (jnp.sqrt(v_hat) + eps)
                step_size = 0.001
            elif algorithm.startswith('Low-Rank'):
                base_vectors = [
                    np.kron(left_vector, right_vector)
                    for left_vector, right_vector in zip(left_vectors, right_vectors)]
                if algorithm == 'Low-Rank Moore-Penrose':
                    approx_inverse_curvature = np.sum(
                        [np.outer(vector, vector) / np.linalg.norm(vector, ord=2)**4
                         for vector in base_vectors],
                        axis=0)
                elif algorithm == 'Low-Rank SGD-Fallback':
                    lr_lr = 0.001
                    approx_inverse_curvature = lr_lr * np.eye(dimension**2) + np.sum(
                        [(1 / np.linalg.norm(vector, ord=2)**2 - lr_lr)
                         * (1 / np.linalg.norm(vector, ord=2)**2)
                         * np.outer(vector, vector)
                         for vector in base_vectors],
                        axis=0)
                else:
                    raise ValueError(f"Algorithm {algorithm} not recognised")
                direction = -approx_inverse_curvature @ gradient
                step_size = num_samples
            else:
                raise ValueError(f'Unknown algorithm {algorithm}')

            if step_size == 'line_search':
                step_size = ((-gradient.T @ direction)
                             / (direction.T @ hessian @ direction))
            point = point + step_size*direction
            model_values.append(
                model(point))
        plt.plot(model_values, label=algorithm)
    plt.legend()
    plt.xlabel('Optimisation Iteration')
    plt.ylabel('Function Value')
    plt.show()
