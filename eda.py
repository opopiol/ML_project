import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv('train_data.csv', header=None)
y = pd.read_csv('train_labels.csv', header=None, names=['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X.head()
X.shape
X_train.shape
X_train.isnull().sum().sum()
X.isnull().sum().sum()
X.describe()

y.head()
y.shape
y_train.shape
y.value_counts()

y = y['y'].apply(lambda x: 1 if x == -1 else 0)

y.value_counts()
y.head()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = y);

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()

standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)