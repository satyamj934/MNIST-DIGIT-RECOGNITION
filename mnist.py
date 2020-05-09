import os

os.chdir('D:\data frames\mnist-in-csv')

import pandas as pd

train_data=pd.read_csv('mnist_train.csv')

test_data=pd.read_csv('mnist_test.csv')

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV


from sklearn.metrics import accuracy_score,confusion_matrix


features=list(set(list(train_data.columns))-{'label'})

trainx=train_data[features]/255

trainy=train_data['label']

testx=test_data[features]/255

testy=test_data['label']

parameters = {'C':[0.1,1,10,100,1000],'kernel':['rbf','linear']}

grid=GridSearchCV(SVC(),parameters)

grid.fit(trainx,trainy)

clf=SVC(kernel='linear')

clf.fit(trainx,trainy)


pred=clf.predict(testx)

acc=accuracy_score(testy,pred)

conf=confusion_matrix(testy,pred)