import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('winequality-red.csv')
# print(df.head())
# print(df.shape)
# print(df.isnull().sum())
# print(df['quality'].value_counts())
# print(df.groupby('quality').mean())

x = df.drop(columns = 'quality', axis = 1)
y = df['quality']

#Label Binarisation

y = y.apply(lambda value: 1 if value >= 7 else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)

model = RandomForestClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)
score = accuracy_score(y_test, predict)
print("Test Prediction Accuracy : ", score)

input = np.asarray([7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0])
input = input.reshape(1, -1)
predicted = model.predict(input)

if predicted[0] == 1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")
