import numpy as np


def create_train_data():
    data = [
        ["Yes", "Yes", "No", "No"],
        ["Yes", "No", "No", "No"],
        ["No", "Yes", "Yes", "Yes"],
        ["No", "Yes", "Yes", "Yes"],
        ["Yes", "Yes", "Yes", "Yes"],
        ["Yes", "No", "Yes", "No"],
        ["No", "No", "Yes", "No"]
    ]
    return np.array(data)


def compute_prior_probablity(data, y_unique):
    prior_probability = np.zeros(len(y_unique))
    row = len(data)

    for i in range(0, 2):
        prior_probability[i] = sum(row[-1] == y_unique[i] for row in data) / row

    return prior_probability


def getIndexFromValue(featureName, listFeatures):
    return np.where(listFeatures == featureName)[0][0]


def compute_conditional_probability(train_data, y_unique):
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(0, len(y_unique)):
            for k in range(0, len(x_unique)):
                x_conditional_probability[j, k] = len(
                    np.where((train_data[:, i] == x_unique[k]) & (train_data[:, -1] == y_unique[j]))[0]) / len(
                    np.where(train_data[:, -1] == y_unique[j])[0])

        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name


def prediction_love_AI(X, list_x_name, prior_probability, conditional_probability):
    xs = []
    for i in range(len(list_x_name)):
        xs.append(getIndexFromValue(X[i], list_x_name[i]))

    p0 = prior_probability[0]
    p1 = prior_probability[1]

    for i, x_index in enumerate(xs):
        p0 *= conditional_probability[i][0, x_index]

    for i, x_index in enumerate(xs):
        p1 *= conditional_probability[i][1, x_index]

    print(p0, p1)

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


def train_naive_bayes(train_data):
    y_unique = ['No', 'Yes']
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probablity(train_data, y_unique)

    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name = compute_conditional_probability(train_data, y_unique)

    return prior_probability, conditional_probability, list_x_name


# 4.6.1
X = ['Yes', 'No', 'Yes']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(data)
pred = prediction_love_AI(X, list_x_name, prior_probability, conditional_probability)

if (pred):
    print("Ad love AI!")
else:
    print("Ad not love AI!")
