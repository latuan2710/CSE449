import math

import pandas as pd

attribute_names = ['love_math', 'love_art', 'love_english']
class_name = 'love_ai'
data = {
    'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
    'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
    'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
    'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no', ]}
df = pd.DataFrame(data, columns=data.keys())


def entropy(value_counts):
    n = value_counts.sum()
    entropy_sum = 0

    for i in value_counts.keys():
        p_i = value_counts[i] / n
        if p_i != 0:
            entropy_sum -= p_i * math.log2(p_i)

    return entropy_sum


def entropy_split_a(attribute_name):
    attribute_values = df[attribute_name].value_counts()
    entropi_A = 0
    for key in attribute_values.keys():
        df_k = df[class_name][df[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = df.shape[0]
        entropi_A = entropi_A + ((n_k / n) * entropy(df_k))
    return entropi_A


class_value_count = df[class_name].value_counts()
entropy_class = entropy(class_value_count)
print("Entropy for class:", entropy_class)
entropy_attribute = {}
for key in attribute_names:
    entropy_attribute[key] = entropy_split_a(key)
    print(f'Entropy for {key} is {entropy_attribute[key]:.3f}')

min_value = min(entropy_attribute.values())
print('\nThe minimum value of Entropy:{0:.3}'.format(min_value))
print('The maximum value of Information Gain:{0:.3}'.format(entropy_class - min_value))

selected_attribute = min(entropy_attribute, key=entropy_attribute.get)
print('\nThe selected attribute is: ', selected_attribute)
