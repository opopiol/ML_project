#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


#import neptune.new as neptune
#run = neptune.init(project='opopiol/ML-project')


# In[3]:


X = pd.read_csv('train_data.csv', header=None)
y = pd.read_csv('train_labels.csv', header=None, names=['y'])


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[5]:


X.head()


# In[6]:


X.shape


# In[7]:


X_train.shape


# In[8]:


X_train.isnull().sum().sum()


# In[9]:


X.isnull().sum().sum()


# In[22]:


X.describe()


# In[10]:


y.head()


# In[11]:


y.shape


# In[12]:


y_train.shape


# In[13]:


y.value_counts()


# In[15]:


y = y['y'].apply(lambda x: 1 if x == -1 else 0)


# In[16]:


y.value_counts()


# In[17]:


y.head()


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[28]:


sns.countplot(x = y);


# In[18]:


from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()

standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)


# In[ ]:




