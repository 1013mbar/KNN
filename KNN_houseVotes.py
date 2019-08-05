# -*- coding: utf-8 -*-
"""
@author: lucas
"""
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('house-votes-84.data')
df.replace('?',-9999,inplace=True)
df.replace('republican',0,inplace=True)
df.replace('democrat',1,inplace=True)
df.replace('y',1,inplace=True)
df.replace('n',1,inplace=True)
#df.drop(['id'],1,inplace=True)
print(df)
X = np.array(df.drop(['party'],1))
y = np.array(df['party'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measure = np.array([0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,1])
example_measure = example_measure.reshape(1,-1)

prediction = clf.predict(example_measure)
print(prediction)