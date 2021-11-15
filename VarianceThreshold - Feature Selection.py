#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import os
from os import path
import matplotlib.pyplot as plt


# In[2]:


#Feature Selection: Variance Threshold - omit all columns with variance <= 0.1 (or 0.01, 0.05 etc.)


# In[3]:


import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# In[4]:


df = pd.read_csv (r'C:\Users\shara\Documents\UZH\Master\HS21\Finance & Machine Learning\Project\cleaned\cleaned_indicators-01-04-2013-31-12-2020.csv')


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


#Separate dependent variable ("bitcoin-price_raw") from independent variables

X=df.drop(labels=['bitcoin-price_raw','Date'], axis=1)
y=df['bitcoin-price_raw']


# In[8]:


# separate dataset into train and test (70/30 split) to avoid overfitting when features are removed

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['bitcoin-price_raw','Date'], axis=1),
    df['bitcoin-price_raw'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# In[9]:


#Continue with X_train only - perform feature selection on X_train, then later omit the undesirable features from X_test manually.
X_train.head()


# In[10]:


# we need to normalize data so that variance threshold is comparable across all features

X_norm = (X_train-X_train.min())/(X_train.max()-X_train.min())
X_norm.head()


# In[11]:


# VARIANCE THRESHOLD
#
#We might have to discuss what the appropriate threshold should be. Is a variance threshold of 0.01 reasonable?

var_thres=VarianceThreshold(threshold=0.01)
var_thres.fit(X_norm)


# In[12]:


var_thres.get_support()


# In[13]:


#How many features don't have low variance? (Number of TRUEs)
len(X_norm.columns[var_thres.get_support()])


# In[14]:


#How many features do have low variance?
low_variance = [column for column in X_norm.columns
                    if column not in X_norm.columns[var_thres.get_support()]]

print(len(low_variance))


# In[15]:


#Which are the features with low variance?
for column in low_variance:
    print(column)


# In[20]:


low_variance


# In[23]:


low_variance.type


# In[24]:


print(type(low_variance))


# In[30]:


X_norm1 = X_norm.drop(low_variance,axis=1)
X_norm1.head()


# In[31]:


X2 = X_train.drop(columns=low_variance)


# In[32]:


X2


# In[33]:


#Is there a more elegant way to reverse the normalization? Do we even need to reverse it?
X_clean1 = X_norm1*(X2.max()-X2.min())+X2.min()
X_clean1


# In[ ]:




