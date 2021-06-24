# -*- coding: utf-8 -*-
"""ML_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19prXM_4R54HKfpar8kqpqUKQD8wcF5w-
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import io
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

#pip install --upgrade --quiet neptune-client
#pip install neptune-notebooks

import neptune.new as neptune
#run = neptune.init(project='opopiol/ML-project')
run = neptune.init(project='opopiol/ML-project',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YjIyYjU2ZS0wMTc5LTQ3NWUtOWZkZC05OTg2YzI1M2VkNDUifQ==') # your credentials

run["JIRA"] = "NPT-952"
run["parameters"] = {"learning_rate": 0.001,
                     "optimizer": "Adam"}

for epoch in range(100):
   run["train/loss"].log(epoch * 0.4)
run["eval/f1_score"] = 0.66

from google.colab import drive
drive.mount('/content/drive')

#import train dataset
train_path = '/content/drive/MyDrive/project/train_data.csv'
train_data = pd.read_csv(train_path, header=None)

#import labels
labels_path = '/content/drive/MyDrive/project/train_labels.csv'
labels = pd.read_csv(labels_path, header=None,  names=['y'])

#import test dataset
test_path = '/content/drive/MyDrive/project/test_data.csv'
test_data = pd.read_csv(test_path, header=None)

X, y = train_data, labels
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

"""### EDA"""

y.value_counts()



y = y['y'].apply(lambda x: 0 if x == -1 else 1)

y.value_counts()

X_train.shape

y_train.shape

#scaling data

#StandardScaler
scaler = StandardScaler()

X_train_scale = StandardScaler().fit_transform(X_train)
y_train_scale = scaler.fit_transform(y_train)

#MinMaxScaler
minmaxscaler = MinMaxScaler()
X_train_minmaxscaler = MinMaxScaler().fit_transform(X_train)

X_train_scale.mean(axis=0)

#X_train_scale has now unit variance and zero mean

n_components = [2 , 0.99] 

for i in n_components:
    pca = PCA(n_components=i, whiten=True)
    X_pca = pca.fit_transform(X_train_scale)
    print(pca.explained_variance_ratio_)
    plt.bar([i], pca.explained_variance_ratio_)

#PCA

pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X_train_scale)

X_pca.shape

X_pca

y_train = y_train['y'].apply(lambda x: 0 if x == -1 else 1)

#final_pca_df = pd.DataFrame(data = X_pca)
#final_pca = pd.concat([final_pca_df, y_train], axis = 1)
#final_pca

sns.set_palette('Set1')

plt.figure(figsize=(8,8))
sns.scatterplot(X_pca[:,0],
           X_pca[:,1], 
           s =40, hue=y_train);

#PCA

pca = PCA(n_components=0.99, whiten=True)
X_pca = pca.fit_transform(X_train_scale)

plt.figure(figsize=(8,8))
sns.scatterplot(X_pca[:,0],
           X_pca[:,1], 
           s =40, hue=y_train);

pca = PCA()
tsne = TSNE()

pipeline = Pipeline([('pca', PCA(n_components=2)), ('tsne', TSNE(n_components=2))])
X_p = pipeline.fit_transform(X)

sns.scatterplot(X_p[:,0],
           X_p[:,1], c=y);

pipeline = Pipeline([('pca', PCA(n_components=0.99)), ('tsne', TSNE(n_components=2))])
X_p = pipeline.fit_transform(X)

sns.scatterplot(X_p[:,0],
           X_p[:,1], c=y);

pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.99)), ('tsne', TSNE(n_components=2))])
X_p = pipeline.fit_transform(X)

sns.scatterplot(X_p[:,0],
           X_p[:,1], c=y);

pipeline = Pipeline([('minmaxscaler', MinMaxScaler()), ('pca', PCA(n_components=0.99)), ('tsne', TSNE(n_components=2))])
X_p = pipeline.fit_transform(X)

sns.scatterplot(X_p[:,0],
           X_p[:,1], c=y);

pipeline = Pipeline([('pca', PCA(n_components=2)), ('tsne', TSNE(n_components=2))])
X_p = pipeline.fit_transform(X)

sns.scatterplot(X_p[:,0],
           X_p[:,1], c=y);


