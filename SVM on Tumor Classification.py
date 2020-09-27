#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from itertools import islice
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score

def plotLCurve(model, X, y):
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    train_sizes = [5, 10, 50, 100, 200, 250, 319]
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=splitter, train_sizes=train_sizes, scoring='accuracy')
    train_scores = np.mean(train_scores, axis=1)
    test_scores = np.mean(test_scores, axis=1)
    pyplot.figure()
    pyplot.xlabel("Training examples")
    pyplot.ylabel("Score")
    pyplot.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    pyplot.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score")
    
def main():
    scaler = StandardScaler()
    data = pd.read_csv('cancer.csv')
    X = np.array(data.iloc[:, 2:32])
    X = scaler.fit_transform(X)
    y = np.array(data['diagnosis'])
    y[y=='M'] = 1
    y[y=='B'] = 0
    y = y.astype(np.int8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    
    #UNREGULARIZED LINEAR SVM
    model_linear = SVC(kernel='linear')
    plotLCurve(model_linear, X_train, y_train)
    model_linear.fit(X_train, y_train)
    predict = model_linear.predict(X_test)
    print(classification_report(y_test,predict))
    scores = cross_val_score(model_linear, X_train, y_train, cv=5, scoring='accuracy')
    print (np.mean(scores))
    
    #REGULARIZED LINEAR SVM
    grid_values = {'C': [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 5, 10]}
    temp = SVC(kernel='linear')
    model_linear_reg = GridSearchCV(estimator=temp, param_grid=grid_values, scoring='accuracy', refit=True)
    model_linear_reg.fit(X_train, y_train)
    plotLCurve(model_linear_reg.best_estimator_, X_train, y_train)
    predict = model_linear_reg.predict(X_test)
    print(classification_report(y_test,predict))
    scores = cross_val_score(model_linear_reg.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
    print (np.mean(scores))
    
    #UNREGULARIZED SVM WITH GAUSSIAN KERNEL
    model_gauss = SVC(kernel='rbf')
    plotLCurve(model_gauss, X_train, y_train)
    model_gauss.fit(X_train, y_train)
    predict = model_gauss.predict(X_test)
    print(classification_report(y_test,predict))
    scores = cross_val_score(model_gauss, X_train, y_train, cv=5, scoring='accuracy')
    print (np.mean(scores))
    
    #REGULARIZED SVM WITH GAUSSIAN KERNEL
    grid_values_two = {'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 5, 10],
                       'gamma': [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 1, 3, 5, 10, 20]}
    temp = SVC(kernel='rbf')
    model_gauss_reg = GridSearchCV(estimator=temp, param_grid=grid_values_two, scoring='accuracy', refit=True)
    model_gauss_reg.fit(X_train, y_train)
    plotLCurve(model_gauss_reg.best_estimator_, X_train, y_train)
    predict = model_gauss_reg.predict(X_test)
    print(classification_report(y_test,predict))
    scores = cross_val_score(model_gauss_reg.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
    print (np.mean(scores))

if __name__ == "__main__":
    main()


# In[ ]:




