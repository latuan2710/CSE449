import math

import pandas as pd

data = {
    'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
    'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
    'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
    'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no', ]}
df = pd.DataFrame(data, columns=data.keys())


def gini_impurity(value_counts):
    n = value_counts.sum()
    p_sum = 0

    for i in value_counts.keys():
        p_sum = p_sum + ((value_counts[i] / n) ** 2)

    gini = 1 - p_sum
    return gini


def gini_split_a(attribute_name):
    attribute_values = df[attribute_name].value_counts()
    gini_A = 0
    for key in attribute_values.keys():
        df_k = df[class_name][df[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = df.shape[0]
        gini_A = gini_A + ((n_k / n) * gini_impurity(df_k))
    return gini_A


min = 1
root = ''
for key1 in list(data.keys()):
    for key2 in list(data.keys()):
        if (key1 != key2):
            class_name = key1
            gini = gini_split_a(key2)
            if (gini < min):
                min = gini
                root = key1
print(min, root)
