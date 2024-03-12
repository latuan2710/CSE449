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


train_data = create_train_data()
print(train_data)


def compute_prior_probablity(train_data):
    y_unique = ['0', '1']
    prior_probability = np.zeros(len(y_unique))

    for row in train_data:
        label = row[-1]
        if label == '0':
            prior_probability[0] += 1
        elif label == '1':
            prior_probability[1] += 1

    prior_probability /= len(train_data)

    return prior_probability


prior_probablity = compute_prior_probablity(train_data)


def cal_mean_and_varience(train_data):
    y_unique = ['0', '1']
    mean = np.zeros(len(y_unique))
    variance = np.zeros(len(y_unique))

    for i, label in enumerate(y_unique):
        lengths = [float(item[0]) for item in train_data if item[1] == label]

        mean_value = sum(lengths) / len(lengths)

        variance_value = sum((x - mean_value) ** 2 for x in lengths) / len(lengths)

        mean[i] = int(mean_value * 1000) / 1000
        variance[i] = int(variance_value * 1000) / 1000

    return mean, variance


train_data = create_train_data()
cal_mean_and_varience(train_data)


def train_naive_bayes(train_data):
    prior_probability = compute_prior_probablity(train_data)

    mean, variance = cal_mean_and_varience(train_data)

    return prior_probability, mean, variance


data = create_train_data()
prior_probability, mean, variance = train_naive_bayes(data)
print(prior_probability, "\nMean", mean, "\nVariance", variance)


def prediction_Iris(X, prior_probability, mean, variance):
    y_unique = ['0', '1']
    rs = np.zeros(len(y_unique))
    p = np.zeros(len(y_unique))
    total = 0
    for i, label in enumerate(y_unique):
        mean_value = float(mean[i])
        variance_value = float(variance[i])

        p_value = float((1 / math.sqrt(2 * math.pi * variance_value)) * math.exp(
            -((float(X[0]) - mean_value) ** 2 / (2 * variance_value)))) * prior_probability[i]

        p[i] = p_value
        total += p_value

    for i in range(len(y_unique)):
        p_value = p[i]
        rs_value = p_value / total

        rs[i] = rs_value
    return p


X = ['3.4', '0']
data = create_train_data()
prior_probability, mean, variance = train_naive_bayes(data)
pred = prediction_Iris(X, prior_probability, mean, variance)
print(pred)
