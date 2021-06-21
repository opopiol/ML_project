import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import svm
from imblearn.over_sampling import SMOTE


#import train dataset
X = pd.read_csv('train_data.csv', header=None)

#import labels
y = pd.read_csv('train_labels.csv', header=None, names=['y'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y = y['y'].apply(lambda x: 1 if x == -1 else 0)

#scale data

standardizer = StandardScaler()

X_train_scale = standardizer.fit_transform(X_train)
y_train_scale = standardizer.fit_transform(y_train)

#PCA
pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X_train_scale)

clf = LogisticRegression(random_state=0).fit(X, y)

svc = svm.SVC()
svc.fit(X, y)


pipe = Pipeline([('pca', pca),
                ('classifier', LogisticRegression())
            ])
search_space = [{'classifier': [LogisticRegression()],
                 'classifier__solver': ["warn"],
                 'classifier__penalty': ['l1'],
                 'classifier__class_weight': [None, "balanced"],
                 'classifier__C': np.logspace(1,4,10)},
                {'classifier': [SVC()],
                 'classifier__kernel': ["linear", "rbf", "poly"],
                 'classifier__class_weight': [None, "balanced"],
                 'classifier__gamma': ["scale"],
                'classifier__C': np.logspace(1,4,10)}]


grid_search = GridSearchCV(pipe,
                           search_space,
                           cv=5,
                           verbose=2,
                           n_jobs=-2,
                           scoring='balanced_accuracy')


best_model = grid_search.fit(X, y.values)


print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(features, target))
