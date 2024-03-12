import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

data = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',')

df = pd.DataFrame(data, columns=data.keys())

classifier = tree.DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

classifier.fit(X, y)

fig, ax = plt.subplots(figsize=(50, 50))
tree.plot_tree(classifier, ax=ax, feature_names=["age", "anaemia", "creatinine_phosphokinase", "diabetes",
                                                 "ejection_fraction", "high_blood_pressure", "platelets",
                                                 "serum_creatinine", "serum_sodium", "sex", "smoking", "time"],
               filled=True)

plt.show()
