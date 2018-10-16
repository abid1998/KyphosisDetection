import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('kyphosis.csv')
print(df.head())


#sns.pairplot(df,hue='Kyphosis')
#plt.show()

X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

print("--------------For RFC---------------")
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred1 = rfc.predict(X_test)
print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
