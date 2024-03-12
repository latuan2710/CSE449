import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('heart_failure_clinical_records_dataset.csv', sep=',')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = LinearRegression()

regr.fit(X_train, y_train)

input = [50, 1, 111, 0, 20, 0, 210000, 1.9, 137, 1, 0, 7]
print(regr.predict([input]))
