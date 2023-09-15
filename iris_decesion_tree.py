#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:54:10 2022

@author: xuezhao
"""

from sklearn import datasets
import pandas as pd

#load datasets
iris=datasets.load_iris()
dir(iris)#see what datasets methods provide
print(iris.DESCR) 

#create dataframes with column names
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=pd.Series(iris.target)
#maps target values with target names
df['target_names']=df['target'].apply(lambda y: iris.target_names[y])

print(df.sample(n=6, random_state=42))

#splitting data
from sklearn.model_selection import train_test_split

df_train, df_test=train_test_split(df,test_size=0.25)
x_train=df_train[iris.feature_names]
x_test=df_test[iris.feature_names]

y_train=df_train['target']
y_test=df_test['target']

#tuning hyperparameter values
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

for max_depth in [1, 2, 3, 4]:
    #initialize a decision tree classifier for each iteration with different max_Depth
    clf=DecisionTreeClassifier(max_depth=max_depth)
    
    #also initialize shuffle splitter
    rs=ShuffleSplit(n_splits=20, test_size=0.25)
    
    cv_results=cross_validate(clf, x_train, y_train, cv=rs, scoring='accuracy')
    accuracy_score=pd.Series(cv_results['test_score']) 
    print('@ max_depth={}:accuracy_scores:{}-{}'.format(max_depth,
                                                   accuracy_score.quantile(.1).round(3),
                                                   accuracy_score.quantile(.9).round(3)))

#visualizing boundaries
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, x, y):
    
    feature_names = x.columns
    x, y= x.values, y.values
    
    x_min, x_max= x[:,0].min(), x[:,0].max()
    y_min, y_max= x[:,1].min(), x[:,1].max()
    
    step=0.02
    
    xx, yy=np.meshgrid(
    np.arange(x_min, x_max, step), 
    np.arange(y_min, y_max, step)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12,8))
    plt.contourf(xx, yy, Z, cmap='Paired_r', alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(x[:,0], x[:,1], c=y, edgecolors='k')
    plt.title("Tree's Decision boundaries")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    
    x=df[['petal width (cm)', 'petal length (cm)']]#use [[]], because they are arrays
    y=df['target']
    
    clf=DecisionTreeClassifier(max_depth=3)
    clf.fit(x,y)
    
    plot_decision_boundary(clf, x, y)
 
#feature engineering
df['petal length x width (cm)'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal length x width (cm)'] = df['sepal length (cm)'] * df['sepal width (cm)']

fig, ax =plt.subplots(1, 1, figsize=(12, 6))

h_label = 'petal length x width (cm)'
v_label = 'sepal length x width (cm)'

for c in df['target'].value_counts().index.to_list():
    df[df['target']==c].plot(title='Class distribution vs newly derived features',
                             kind='scatter',
                             x=h_label, y=v_label,
                             color=['r','g','b'][c],
                             marker=f'${c}$',
                             s=64, alpha=0.5, ax=ax,)
    fig

from sklearn.metrics import accuracy_score

features_orig=iris.feature_names
features_new = ['petal length x width (cm)', 'sepal length x width (cm)']

accuracy_scores_orig = []
accuracy_scores_new = []

for _ in range(500):
    
    df_train, df_test = train_test_split(df, test_size=0.3)

    x_train_orig = df_train[features_orig]
    x_train_new = df_train[features_new]

    x_test_orig = df_test[features_orig]
    x_test_new = df_test[features_new]

    y_train = df_train['target']
    y_test = df_test['target']

    clf_orig = DecisionTreeClassifier(max_depth=2)
    clf_new = DecisionTreeClassifier(max_depth=2)

    clf_orig.fit(x_train_orig, y_train)
    clf_new.fit(x_train_new, y_train)

    y_pred_orig = clf_orig.predict(x_test_orig)
    y_pred_new = clf_new.predict(x_test_new)

    accuracy_scores_orig.append(round(accuracy_score(y_test, y_pred_orig), 3))
    accuracy_scores_new.append(round(accuracy_score(y_test, y_pred_new), 3))
    
 
accuracy_scores_orig = pd.Series(accuracy_scores_orig)
accuracy_scores_new = pd.Series(accuracy_scores_new)
    
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True);

accuracy_scores_orig.plot(title='Distribution of Classifier accuracy [Originial feature]',
                          kind='box',
                          grid=True,
                          ax=axs[0])

accuracy_scores_new.plot(
    title='Distribution of classifier accuracy [New Features]',
    kind='box',
grid=True,
ax=axs[1]
)

fig
