import cv2
import math
from matplotlib import pyplot as plt


def computeXDerivative(image):
    w = len(image[0])
    h = len(image)
    x_derivative = [[0] * w for _ in range(h)]

    for y in range(h):
        for x in range(1, w - 1):
            x_derivative[y][x] = (image[y][x + 1] - image[y][x - 1])

    return x_derivative


def computeYDerivative(image):
    w = len(image[0])
    h = len(image)
    y_derivative = [[0] * w for _ in range(h)]

    for y in range(1, h - 1):
        for x in range(w):
            y_derivative[y][x] = (image[y + 1][x] - image[y - 1][x])

    return y_derivative


def computeMagnitudeXY(image):
    w = len(image[0])
    h = len(image)
    gradient_magnitude = [[0] * w for _ in range(h)]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gradient_magnitude[y][x] = math.sqrt(
                (image[y + 1][x] - image[y - 1][x]) ** 2
                +
                (image[y][x + 1] - image[y][x - 1]) ** 2
            )

    return gradient_magnitude


cat_image = cv2.imread('cat.jpeg', 0)
cat_image = cv2.resize(cat_image, (400, 400), cv2.INTER_AREA)

image = cat_image.tolist()
x_derivative = computeXDerivative(image)
y_derivative = computeYDerivative(image)
gradient_magnitude = computeMagnitudeXY(image)

fig = plt.figure(figsize=(10, 7))

fig.add_subplot(2, 2, 1)
plt.imshow(cat_image, cmap='gray')
plt.axis('off')
plt.title('Input image')

fig.add_subplot(2, 2, 2)
plt.imshow(x_derivative, cmap='gray')
plt.axis('off')
plt.title('Gradient in X- direction')

fig.add_subplot(2, 2, 3)
plt.imshow(y_derivative, cmap='gray')
plt.axis('off')
plt.title('Gradient in Y- direction')

fig.add_subplot(2, 2, 4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.axis('off')
plt.title('Gradient Magnitude')

plt.show()
