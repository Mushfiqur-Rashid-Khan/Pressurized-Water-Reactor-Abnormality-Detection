import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

data=pd.read_csv("PWR_Data.csv")
data.head()

data.drop(['Number'], axis=1)

values=data.values

imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputedData=imputer.fit_transform(values)

scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(imputedData)

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X = normalizedData[:,0:11]
Y = normalizedData[:,13]

kfold= model_selection.KFold(n_splits=9, random_state=7, shuffle=True)

cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
logisticRegr.predict(X_test[0:12])
predictions = logisticRegr.predict(X_test)
predictions = logisticRegr.predict(X_test)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, predictions)

import matplotlib.pyplot as plt
cm_display.plot()
plt.show()

Accuracy = metrics.accuracy_score(y_test, predictions)
acc=Accuracy*100
print(acc)

Precision = metrics.precision_score(y_test, predictions)
pre=Precision*100
print(pre)

Recall = metrics.recall_score(y_test, predictions)
rec=Recall*100
print(rec)

F1 = metrics.f1_score(y_test, predictions)
f1=F1*100
print(f1)

import time
st=time.time()*1000000
end=time.time()*1000000
print(end-st)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Faultless', 'Faulty'])
