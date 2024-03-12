import random
import numpy as np
import pandas as pd


def create_train_data_area():
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=",")

    x = data.loc[:, ["age", "anaemia", "creatinine_phosphokinase", "diabetes",
                     "ejection_fraction", "high_blood_pressure", "platelets",
                     "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]].values

    return x, data.loc[:, 'DEATH_EVENT'].values


def initialize_params():
    bias = 0
    weights = []
    for _ in range(12):
        weight = random.gauss(mu=0.0, sigma=0.01)
        weights.append(weight)

    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12 = weights
    return [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, bias]


def predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b):
    result = (
            w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 +
            w6 * x6 + w7 * x7 + w8 * x8 + w9 * x9 + w10 * x10 +
            w11 * x11 + w12 * x12 + b
    )
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

    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b = initialize_params()

    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
            # get a sample
            x1 = X_data[i][0]
            x2 = X_data[i][1]
            x3 = X_data[i][2]
            x4 = X_data[i][3]
            x5 = X_data[i][4]
            x6 = X_data[i][5]
            x7 = X_data[i][6]
            x8 = X_data[i][7]
            x9 = X_data[i][8]
            x10 = X_data[i][9]
            x11 = X_data[i][10]
            x12 = X_data[i][11]

            y = y_data[i]

            # print(y)
            # compute output
            y_hat = predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10,
                            w11, w12, b)

            # compute loss
            loss = compute_loss_mse(y, y_hat)

            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_dw4 = compute_gradient_wi(x4, y, y_hat)
            dl_dw5 = compute_gradient_wi(x5, y, y_hat)
            dl_dw6 = compute_gradient_wi(x6, y, y_hat)
            dl_dw7 = compute_gradient_wi(x7, y, y_hat)
            dl_dw8 = compute_gradient_wi(x8, y, y_hat)
            dl_dw9 = compute_gradient_wi(x9, y, y_hat)
            dl_dw10 = compute_gradient_wi(x10, y, y_hat)
            dl_dw11 = compute_gradient_wi(x11, y, y_hat)
            dl_dw12 = compute_gradient_wi(x12, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            # update parameters
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            w4 = update_weight_wi(w4, dl_dw4, lr)
            w5 = update_weight_wi(w5, dl_dw5, lr)
            w6 = update_weight_wi(w6, dl_dw6, lr)
            w7 = update_weight_wi(w7, dl_dw7, lr)
            w8 = update_weight_wi(w8, dl_dw8, lr)
            w9 = update_weight_wi(w9, dl_dw9, lr)
            w10 = update_weight_wi(w10, dl_dw10, lr)
            w11 = update_weight_wi(w11, dl_dw11, lr)
            w12 = update_weight_wi(w12, dl_dw12, lr)
            b = update_weight_b(b, dl_db, lr)

            # logging
            losses.append(loss)
    return (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b, losses)


x, y = create_train_data_area()
(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b, losses) = implement_linear_regression(x, y)
print(predict(50, 1, 111, 0, 20, 0, 210000, 1.9, 137, 1, 0, 7,
              w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b))
