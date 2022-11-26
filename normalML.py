import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from random import randrange
import numpy as np
import xgboost as xg

df = pd.read_csv("eyeAdgFinalFeatures.csv")

Target = df["Label"]
features = df.drop("Label",axis=1)

feature_names=tuple(features.columns)
X_train = features
y_train = Target

accuracy_matrix =[]


model=RandomForestClassifier(n_estimators=100)
acc = []
for i in range(20):
   # Cross validation
   cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
   scores = cross_val_score(model, X_train, y_train, cv=cv)
   acc.append(scores.mean())
acc = np.array(acc)
accuracy_matrix.append(acc.mean())



model = xg.XGBClassifier(n_estimators=900,random_state=40, learning_rate=0.01)
acc = []
for i in range(20):
   # Cross validation
   cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
   scores = cross_val_score(model, X_train, y_train, cv=cv)
   acc.append(scores.mean())
acc = np.array(acc)
accuracy_matrix.append(acc.mean())


model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
acc = []
for i in range(20):
   # Cross validation
   cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
   scores = cross_val_score(model, X_train, y_train, cv=cv)
   acc.append(scores.mean())
acc = np.array(acc)
accuracy_matrix.append(acc.mean())



model = LogisticRegression()
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
acc = []
for i in range(20):
   # Cross validation
   cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
   scores = cross_val_score(model, X_train, y_train, cv=cv)
   acc.append(scores.mean())
acc = np.array(acc)
accuracy_matrix.append(acc.mean())



model=DecisionTreeClassifier()
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
acc = []
for i in range(20):
   # Cross validation
   cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
   scores = cross_val_score(model, X_train, y_train, cv=cv)
   acc.append(scores.mean())
acc = np.array(acc)
accuracy_matrix.append(acc.mean())


model=KNeighborsClassifier() 
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
acc = []
for i in range(20):
   # Cross validation
   cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
   scores = cross_val_score(model, X_train, y_train, cv=cv)
   acc.append(scores.mean())
acc = np.array(acc)
accuracy_matrix.append(acc.mean())





print(accuracy_matrix)
