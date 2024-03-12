# Example 2.1
import random


def f(x):
    return 3 * x ** 4 - 4 * x ** 2 - 6 * x - 3


def find_minimum(fx, x, num_iteration, step):
    for i in range(num_iteration):
        dx = (fx(x + 0.05) - fx(x - 0.05)) / (2 * 0.05)
        if dx > 0:
            x = x - step
        elif dx < 0:
            x = x + step
    return x


x = random.uniform(-10, 10)
print("initial x: ", x)

x = 8.033107966229196
x = find_minimum(f, x, 100, 0.1)
print(x)

x = 3.0
x = find_minimum(f, x, 1, 0.1)
print(round(x, 2))

x = 3.0
x = find_minimum(f, x, 5, 0.1)
print(round(x, 2))

x = 3.0
x = find_minimum(f, x, 100, 0.2)
print(round(x, 2))
