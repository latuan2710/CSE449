import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('forestfires.csv', sep=',')

df['month'].replace({'jan': 1}, inplace=True)
df['month'].replace({'feb': 2}, inplace=True)
df['month'].replace({'mar': 3}, inplace=True)
df['month'].replace({'apr': 4}, inplace=True)
df['month'].replace({'may': 5}, inplace=True)
df['month'].replace({'jun': 6}, inplace=True)
df['month'].replace({'jul': 7}, inplace=True)
df['month'].replace({'aug': 8}, inplace=True)
df['month'].replace({'sep': 9}, inplace=True)
df['month'].replace({'oct': 10}, inplace=True)
df['month'].replace({'nov': 11}, inplace=True)
df['month'].replace({'dec': 12}, inplace=True)

df['day'].replace({'mon': 2}, inplace=True)
df['day'].replace({'tue': 3}, inplace=True)
df['day'].replace({'wed': 4}, inplace=True)
df['day'].replace({'thu': 5}, inplace=True)
df['day'].replace({'fri': 6}, inplace=True)
df['day'].replace({'sat': 7}, inplace=True)
df['day'].replace({'sun': 8}, inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = LinearRegression()

regr.fit(X_train, y_train)

input = [2, 4, 8, 8, 93.6, 235.1, 723.1, 10.1, 20.9, 66, 4.9, 0]
print(regr.predict([input]))
