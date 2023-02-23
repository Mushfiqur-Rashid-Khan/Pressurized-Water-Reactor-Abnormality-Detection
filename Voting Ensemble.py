import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
lr=LogisticRegression()
svc=LinearSVC(C=1.0)
rfc=RandomForestClassifier(n_estimators=100)

df=pd.read_csv("PWR_Data.csv")

from sklearn.model_selection import train_test_split

train,test= train_test_split(df,test_size=0.3)

train_feat=train.iloc[:,:12]
train_targ=train["Status"]

print("{0:0.2f}% in training set".format((len(train_feat)/len(df.index)) * 100))
print("{0:0.2f}% in testing set".format((1-len(train_feat)/len(df.index)) * 100))

seed = 7
kfold = model_selection.KFold(n_splits=10)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, train_feat, train_targ, cv=kfold)
acc=results.mean()
acc1=acc*100
print("The accuracy is: ",acc1,'%')

import time
start_time = time.time()
print("--- %s seconds ---" % (time.time()*1000000 - start_time*1000000))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( train_feat, train_targ, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
logisticRegr.predict(X_test[0:2])
predictions = logisticRegr.predict(X_test)
predictions = logisticRegr.predict(X_test)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Faultless', 'Faulty'])

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
