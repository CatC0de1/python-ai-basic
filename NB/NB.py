import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_excel('parkinsons.xlsx')

data['Class'] = data['Class'].fillna('Healthy')
data = data.replace({',': '.'}, regex=True)

data_train = data.iloc[:100]
data_test = data.iloc[100:]

x_train = data_train.drop(columns=['Class'])
y_train = data_train['Class']
x_test = data_test.drop(columns=['Class'])
y_test = data_test['Class']

y_train.hist()
plt.show()

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy} || Accuracy %: {accuracy * 100}%')
print(f'Error Rate: {1 - accuracy} || Error Rate %: {(1 - accuracy) * 100}%')