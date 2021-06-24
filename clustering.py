import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
X = pd.read_csv(train_path, header=None)

#import labels
labels = '/content/drive/MyDrive/project/train_labels.csv'
y = pd.read_csv(labels, header=None,  names=['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

"""### EDA"""

y.value_counts()

y = y['y'].apply(lambda x: 0 if x == -1 else 1)

y.value_counts()

y_train = y_train['y'].apply(lambda x: 0 if x == -1 else 1)

X_train.shape

y_train.shape

#scaling data

scaler = StandardScaler()

X_train_scale = StandardScaler().fit_transform(X_train)
y_train_scale = scaler.fit_transform(y_train)

X_train_scale.mean(axis=0)

#X_train_scale has now unit variance and zero mean

#PCA

pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X_train_scale)

X_pca.shape

X_pca

#final_pca_df = pd.DataFrame(data = X_pca)
#final_pca = pd.concat([final_pca_df, y_train], axis = 1)
#final_pca

sns.set_palette('Set1')

plt.figure(figsize=(8,8))
sns.scatterplot(X_pca[:,0],
           X_pca[:,1], hue=y_train);