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


#import train dataset
X = pd.read_csv('train_data.csv', header=None)

#import labels
y = pd.read_csv('train_labels.csv', header=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y = y['y'].apply(lambda x: 1 if x == -1 else 0)

#baseline
strategies = ['stratified', 'most_frequent', 'prior', 'uniform']

for i in strategies:
    dummy_clf = DummyClassifier(strategy=i)
    dummy_clf.fit(X_train, y_train)
    score = dummy_clf.score(X_train, y_train)
    dum_labels = np.unique(dummy_clf.predict(X))
    print(f'For {i} strategy score is {score}')
    plt.bar(i, score)

#scale data
scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)
y_train_scale = scaler.fit_transform(y_train)

#PCA
pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X_train_scale)

#LogisticRegression
clf = LogisticRegression(random_state=0).fit(X, y)

#SVC
svc = svm.SVC()
svc.fit(X, y)

#KNeighborsClassifier
knn = KNeighborsClassifier().fit(X, y)




#creating pipeline

pipe = Pipeline([('scaler', scaler), ('pca', pca),
                ('classifier', LogisticRegression())
            ])


search_space = [{'classifier': [LogisticRegression()],
                 'classifier__solver': ['warn', 'liblinear'],
                 'classifier__penalty': ['l1', 'l2'],
                 'classifier__class_weight': [None, 'balanced'],
                 'classifier__C': np.logspace(1,4,10)},
                {'classifier': [KNeighborsClassifier()],
                 'classifier__n_neighbors': [2, 4, 6, 8, 10, 20],
                 'classifier__algorithm': ['auto']},
                {'classifier': [SVC()],
                 'classifier__kernel': ['linear', 'rbf', 'poly'],
                 'classifier__class_weight': [None, 'balanced'],
                 'classifier__gamma': ['scale', 'auto'],
                 'classifier__C': np.logspace(1,4,10)},
                {'classifier': [RandomForestClassifier()],
                 'classifier__n_estimators': [10,20,50,100,200],
                 'classifier__criterion': ['gini', 'entropy'],
                 'classifier__max_depth': [2,5,10,None],
                 'classifier__max_features': ['auto', 'sqrt', 'log2'],
                 'classifier__class_weight': ['balanced', 'balanced_subsample']},
                {'estimator': [DecisionTreeClassifier()],
                 'estimator__criterion': ['gini', 'entropy'],
                 'estimator__class_weight': ['balanced'],
                 'estimator__splitter': ['best', 'random'],
                 'estimator__max_depth': [2,5,10,None]}]


grid_search = GridSearchCV(pipe,
                           search_space,
                           cv=5,
                           verbose=2,
                           n_jobs=-2,
                           scoring='balanced_accuracy')


best_model = grid_search.fit(X, y.values)


print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(X, y))
