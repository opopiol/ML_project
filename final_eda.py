import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

#----------------------------------------------------------------------------------
#DATA IMPORT

from google.colab import drive
drive.mount('/content/drive')

#import train dataset
def i_train_d():
  train_path = '/content/drive/MyDrive/project/train_data.csv'
  train_data = pd.read_csv(train_path, header=None)

i_train_d()

#import labels
def i_labels():
  labels_path = '/content/drive/MyDrive/project/train_labels.csv'
  labels = pd.read_csv(labels_path, header=None,  names=['y'])

i_labels()

#import test dataset
def i_test_d():
  test_path = '/content/drive/MyDrive/project/test_data.csv'
  test_data = pd.read_csv(test_path, header=None)

i_test_d()

def x_y():
  X, y = train_data, labels
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=1)

x_y()

#----------------------------------------------------------------------------------
#EDA


#METRICS
y.value_counts()
#The distribution of the labels is very uneven. They appear in proportions of about 1:10.
#That’s why we decided to choose f1-score.

y_train = y_train['y'].apply(lambda x: 0 if x == -1 else 1)

y_train.value_counts()

X_train.shape

y_train.shape

X_train.describe()

X_train.isnull().sum().sum()


#----------------------------------------------------------------------------------
#DISTRIBUTION
def distribution(x_1: np.array):
    """This function returns plot with distribution
    :param x_1: np.array: train data after splitting
    :return: sns.kdeplot(pd.DataFrame(x_1)[i]): plot with distribution
    """
  for i in [x_1.skew().idxmin(), x_1.skew().idxmax()]:
    sns.kdeplot(pd.DataFrame(x_1)[i])
    plt.legend(labels=['x_1.skew().idxmin()', 'x_1.skew().idxmax()'])

distribution(X_train)

#We can see a normal distribution.


#----------------------------------------------------------------------------------
#SCALING DATA

#StandardScaler
scaler = StandardScaler()

X_train_scale = StandardScaler().fit_transform(X_train)

#MinMaxScaler
minmaxscaler = MinMaxScaler()
X_train_minmaxscaler = MinMaxScaler().fit_transform(X_train)

#----------------------------------------------------------------------------------
#PCA

sns.set_palette('Set1')

def pca_scatterplot(n):
  """This function returns scatterplot
  :param n: number of components in PCA()
  :return: sns.scatterplot(X_p[:,0], X_p[:,1], c=y_train: scatterplot
  """
  pca = PCA(n_components=n, whiten=True)
  X_pca = pca.fit_transform(X_train_scale)

  plt.figure(figsize=(6,6))
  plt.title(f'PCA(n_components={n})')
  sns.scatterplot(X_pca[:,0],
            X_pca[:,1],
            s=30, hue=y_train);

pca_scatterplot(2)

pca_scatterplot(0.99)


pca = PCA()
tsne = TSNE()

def clustering_scatterplot(a, b):
  """This function returns scatterplot with clusters from seaborn's library
    :param a: n_components for PCA
    :param b: n_components for TSNE
    :return: sns.scatterplot(X_p[:,0], X_p[:,1], c=y: scatterplot with clusters
    """
  pipeline = Pipeline([('pca', PCA(n_components = a)), ('tsne', TSNE(n_components = b))])
  X_p = pipeline.fit_transform(X)

  sns.scatterplot(X_p[:,0], X_p[:,1], c=y)
  plt.title(f'PCA(n_components={a})&TSNE(n_components={b})');

clustering_scatterplot(2, 2)

#best for our project with n_components=0.99 for PCA and n_components=2 for TSNE
clustering_scatterplot(0.99, 2)

clustering_scatterplot(0.95, 2)
