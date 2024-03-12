import numpy as np


def create_train_data():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'no'],
        ['Sunny', 'Hot', 'High', 'Strong', 'no'],
        ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'no'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'yes']
    ]
    return np.array(data)


def compute_prior_probablity(data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    row = len(data)

    for i in range(0, 2):
        prior_probability[i] = sum(row[-1] == y_unique[i] for row in data) / row

    return prior_probability


# train_data = create_train_data()
# prior_probablity = compute_prior_probablity(train_data)
# print("P(play tennis = No) = ", prior_probablity[0])
# print("P(play tennis = Yes) = ", prior_probablity[1])
def getIndexFromValue(featureName, listFeatures):
    return np.where(listFeatures == featureName)[0][0]


def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(0, len(y_unique)):
            for k in range(0, len(x_unique)):
                x_conditional_probability[j, k] = len(
                    np.where((train_data[:, i] == x_unique[k]) & (train_data[:, 4] == y_unique[j]))[0]) / len(
                    np.where(train_data[:, 4] == y_unique[j])[0])

        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name


# # 4.3.1
# train_data = create_train_data()
# _, list_x_name = compute_conditional_probability(train_data)
# print("x1 = ", list_x_name[0])
# print("x2 = ", list_x_name[1])
# print("x3 = ", list_x_name[2])
# print("x4 = ", list_x_name[3])


# Question: 4.4.1
# train_data = create_train_data()
# _, list_x_name = compute_conditional_probability(train_data)
# outlook = list_x_name[0]
# i1 = getIndexFromValue("Overcast", outlook)
# i2 = getIndexFromValue("Rain", outlook)
# i3 = getIndexFromValue("Sunny", outlook)
# print(i1, i2, i3)

# # Question: 4.4.2
# train_data = create_train_data()
# conditional_probability, list_x_name = compute_conditional_probability(train_data)
# x1 = getIndexFromValue("Sunny", list_x_name[0])
# print("P('Outlook'='Sunny'|Play Tennis'='Yes') = ", np.round(conditional_probability[0][1, x1], 2))

# # Question: 4.4.3
# train_data = create_train_data()
# conditional_probability, list_x_name = compute_conditional_probability(train_data)
# # Compute P("Outlook"="Sunny"|Play Tennis"="No")
# x1 = getIndexFromValue("Sunny", list_x_name[0])
# print("P('Outlook'='Sunny'|Play Tennis'='No') = ", np.round(conditional_probability[0][0, x1], 2))

####################
# Prediction
####################
def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):
    x1 = getIndexFromValue(X[0], list_x_name[0])
    x2 = getIndexFromValue(X[1], list_x_name[1])
    x3 = getIndexFromValue(X[2], list_x_name[2])
    x4 = getIndexFromValue(X[3], list_x_name[3])

    p0 = conditional_probability[0][0, x1] * conditional_probability[1][0, x2] * conditional_probability[2][0, x3] * \
         conditional_probability[3][0, x4] * prior_probability[0]

    p1 = conditional_probability[0][1, x1] * conditional_probability[1][1, x2] * conditional_probability[2][1, x3] * \
         conditional_probability[3][1, x4] * prior_probability[1]

    print(p0, p1)

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


def train_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probablity(train_data)

    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name = compute_conditional_probability(train_data)

    return prior_probability, conditional_probability, list_x_name


# 4.6.1
X = ['Sunny', 'Cool', 'High', 'Strong']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(data)
pred = prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability)

if (pred):
    print("Ad should go!")
else:
    print("Ad should not go!")
