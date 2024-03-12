import math

import numpy as np


def create_train_data():
    data = [
        ['1.4', '0'],
        ['1', '0'],
        ['1.3', '0'],
        ['1.9', '0'],
        ['2.0', '0'],
        ['1.8', '0'],
        ['3.0', '1'],
        ['3.8', '1'],
        ['4.1', '1'],
        ['3.9', '1'],
        ['4.2', '1'],
        ['3.4', '1'],
    ]
    return np.array(data)


def compute_prior_probablity(data):
    y_unique = ['0', '1']
    prior_probability = np.zeros(len(y_unique))
    row = len(data)

    for i in range(0, 2):
        prior_probability[i] = sum(row[-1] == y_unique[i] for row in data) / row

    return prior_probability


def cal_mean(data):
    result = []

    result.append(sum([float(item[0]) for item in data if item[-1] == '0']) / sum(row[-1] == '0' for row in data))
    result.append(sum([float(item[0]) for item in data if item[-1] == '1']) / sum(row[-1] == '1' for row in data))

    return result


def cal_sigma(data, means):
    result = []

    result.append(sum([((float(item[0]) for item in data if item[-1] == '0') - means[0]) ** 2]) / sum(
        row[-1] == '0' for row in data))
    result.append(sum([((float(item[0]) for item in data if item[-1] == '1') - means[1]) ** 2]) / sum(
        row[-1] == '1' for row in data))

    return result


def gauss(x, mean, sigma):
    result = (1.0 / (np.sqrt(2 * math.pi * sigma))) \
             * (np.exp(-(float(x) - mean) ** 2 / (2 * sigma)))
    return result


data = create_train_data()
means = cal_mean(data)
sigmas = cal_sigma(data, means)
print(means, sigmas)
