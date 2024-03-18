import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

data = {
    'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
    'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
    'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
    'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no', ],
}

df = pd.DataFrame(data, columns=data.keys())

classifier = tree.DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=1)

one_hot_data = pd.get_dummies(df[['love_math', 'love_art', 'love_english']], drop_first=True)
print(one_hot_data)

X = one_hot_data.iloc[:, :].values
y = df['love_ai'].values

classifier.fit(X, y)

fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(classifier, ax=ax, feature_names=['love_math', 'love_art', 'love_english'], filled=True)

plt.show()
