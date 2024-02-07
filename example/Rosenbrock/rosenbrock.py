import numpy as np
from dwmgrad.dwmg import DynamicWinMoAdaGradv2


def rosenbrock_function(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_gradient(x, y, a=1, b=100):
    grad_x = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    grad_y = 2 * b * (y - x ** 2)
    return np.array([grad_x, grad_y])


def train_optimizer_with_output(optimizer, optimizer_name, initial_params, max_steps=1000):
    params = np.array(initial_params, copy=True)
    losses = []
    params_history = [params.copy()]

    if optimizer_name == "dynamic_adagrad":
        prev_loss = None
        for step in range(max_steps):
            x, y = params[0], params[1]
            grads = rosenbrock_gradient(x, y)
            current_loss = rosenbrock_function(x, y)
            losses.append(current_loss)

            loss_diff = 0
            if prev_loss is not None:
                loss_diff = prev_loss - current_loss
            prev_loss = current_loss

            optimizer.update([params], [grads], loss_diff)
            params_history.append(params.copy())
    else:
        for step in range(max_steps):
            x, y = params[0], params[1]
            grads = rosenbrock_gradient(x, y)
            current_loss = rosenbrock_function(x, y)
            losses.append(current_loss)

            optimizer.update([params], [grads])
            params_history.append(params.copy())

    return params, losses, params_history


def plot_optimization_path(ax, params_history, label, color):
    # Unpack parameters
    xs = [p[0] for p in params_history]
    ys = [p[1] for p in params_history]
    ax.plot(xs, ys, label=label, color=color, marker='o')


def plot_rosenbrock_contours(ax, xlim, ylim):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_function(X, Y)
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray')
    ax.plot(1, 1, 'go', label='Optimal Solution (1,1)', markersize=15)  # 'go'表示绿色圆点


final_params_dynamic, losses_dynamic, params_history_dynamic = train_optimizer_with_output(
    DynamicWinMoAdaGradv2(lr=0.001), "dynamic_adagrad", [1.0, 2.0])

print("Final parameters with our method:", final_params_dynamic)
