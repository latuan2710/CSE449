import random
import numpy as np
import pandas as pd


def prepare_data(file_name_dataset):
    data = pd.read_csv(file_name_dataset, sep=";")

    g1 = data.loc[:, 'G1']
    g2 = data.loc[:, 'G2']
    studyTime = data.loc[:, "studytime"]
    failures = data.loc[:, "failures"]
    absences = data.loc[:, 'absences']

    predict = "G3"

    x = np.column_stack((g1, g2, studyTime, failures, absences))
    y = np.array(data[predict])

    return x, y

def initialize_params():
    bias = 0
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)
    w4 = random.gauss(mu=0.0, sigma=0.01)
    w5 = random.gauss(mu=0.0, sigma=0.01)
    return [w1, w2, w3, w4, w5, bias]


def predict(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, b):
    result = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + b
    return result


def compute_loss_mse(y_hat, y):
    return (y_hat - y) ** 2


def compute_gradient_wi(xi, y, y_hat):
    dl_dwi = 2 * xi * (y_hat - y)
    return dl_dwi


def compute_gradient_b(y, y_hat):
    dl_db = 2 * (y_hat - y)
    return dl_db


def update_weight_wi(wi, dl_dwi, lr):
    wi = wi - lr * dl_dwi
    return wi


def update_weight_b(b, dl_db, lr):
    b = b - lr * dl_db
    return b


def implement_linear_regression(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []

    w1, w2, w3, w4, w5, b = initialize_params()

    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
            # get a sample
            x1 = X_data[i][0]
            x2 = X_data[i][1]
            x3 = X_data[i][2]
            x4 = X_data[i][3]
            x5 = X_data[i][4]

            y = y_data[i]

            # print(y)
            # compute output
            y_hat = predict(x1, x2, x3, x4, x5, w1, w2, w3, w4, w5, b)

            # compute loss
            loss = compute_loss_mse(y, y_hat)

            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_dw4 = compute_gradient_wi(x4, y, y_hat)
            dl_dw5 = compute_gradient_wi(x5, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            # update parameters
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            w4 = update_weight_wi(w4, dl_dw4, lr)
            w5 = update_weight_wi(w5, dl_dw5, lr)
            b = update_weight_b(b, dl_db, lr)

            # logging
            losses.append(loss)
    return (w1, w2, w3, w4, w5, b, losses)


x, y = prepare_data('student-mat.csv')
print(x,y)
(w1, w2, w3, w4, w5, b, losses) = implement_linear_regression(x, y)
print(predict(10, 9, 3, 0, 2, w1, w2, w3, w4, w5, b))
print(predict(13, 13, 2, 0, 0, w1, w2, w3, w4, w5, b))
