import numpy as np
import matplotlib.pyplot as plt


def plot_function_and_derivative(x, func, derivative):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    ax.plot(x, func(x), color="#d35400", linewidth=3, label=func.__name__)
    ax.plot(x, derivative(x), color="#1abd15", linewidth=3, label=derivative.__name__)

    ax.legend(loc="upper left", frameon=False)
    plt.show()


def prelu(x, alpha=0.25):
    return np.where(x <= 0, alpha * x, x)


def autograd(func, x):
    epsilon = 1e-6
    delta_y = func(x + epsilon) - func(x)
    derivative = delta_y / epsilon
    return derivative


def prelu_list(x_data):
    fx = [prelu(x) for x in x_data]
    return fx


def autograd_prelu_list(x_data):
    func = prelu
    dfx = [autograd(func, x) for x in x_data]
    return dfx


x_data = list(range(-100, 100 + 1, 1))
x_data = [x / 10 for x in x_data]
plot_function_and_derivative(x_data, prelu_list, autograd_prelu_list)

x_list = list([-1, 0, 1])
x_prelu_list = prelu_list(x_list)
print([np.round(num, 2) for num in x_prelu_list])

x_list = list([-1, 0, 1])
x_gradient_list = autograd_prelu_list(prelu_list(x_list))
print([np.round(num, 2) for num in x_gradient_list])
