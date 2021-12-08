#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import os
from os import path
import matplotlib.pyplot as plt


# In[2]:


#
#
#Feature Selection: Variance Threshold - omit all columns with variance <= 0.1 (or 0.01, 0.05 etc.)
#
#
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# In[3]:


df = pd.read_csv (r'C:\Users\shara\Documents\UZH\Master\HS21\Finance & Machine Learning\Project\cleaned\cleaned_indicators-01-04-2013-31-12-2020.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


#Separate dependent variable ("bitcoin-price_raw") from independent variables

X=df.drop(labels=['bitcoin-price_raw','Date'], axis=1)
y=df['bitcoin-price_raw']


# In[7]:


# separate dataset into train and test (70/30 split) to avoid overfitting when features are removed

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['bitcoin-price_raw','Date'], axis=1),
    df['bitcoin-price_raw'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# In[8]:


#Continue with X_train only - perform feature selection on X_train, then later omit the undesirable features from X_test manually.
X_train.head()


# In[9]:


# we need to normalize data so that variance threshold is comparable across all features

X_norm = (X_train-X_train.min())/(X_train.max()-X_train.min())
X_norm.head()


# In[10]:


# VARIANCE THRESHOLD
#
#We might have to discuss what the appropriate threshold should be. Is a variance threshold of 0.01 reasonable?

var_thres=VarianceThreshold(threshold=0.01)
var_thres.fit(X_norm)


# In[11]:


var_thres.get_support()


# In[12]:


#How many features don't have low variance? (Number of TRUEs)
len(X_norm.columns[var_thres.get_support()])


# In[13]:


#How many features do have low variance?
low_variance = [column for column in X_norm.columns
                    if column not in X_norm.columns[var_thres.get_support()]]

print(len(low_variance))


# In[14]:


#Which are the features with low variance?
for column in low_variance:
    print(column)


# In[15]:


X_norm1 = X_norm.drop(low_variance,axis=1)
X_norm1.head()


# In[17]:


X2 = X_train.drop(columns=low_variance)


# In[18]:


#Is there a more elegant way to reverse the normalization? Do we even need to reverse it?
X_clean1 = X_norm1*(X2.max()-X2.min())+X2.min()
X_clean1


# In[19]:


#
#
# PEARSON CORRELATION
#
#


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


X_clean1.corr()


# In[22]:


# (function syntax copied from a scikit-learn documentation)
# function selects highly correlated features and remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[23]:


# We can discuss what a reasonable correlation level is 
correlated_features = correlation(X_clean1, 0.9)
len(set(correlated_features))


# In[24]:


# So, thee are 214 features that are strongly correlated with other features
correlated_features


# In[25]:


X_clean2 = X_clean1.drop(correlated_features,axis=1)


# In[26]:


X_clean2.head()


# In[27]:


# above a datafram that has undergone two successive feature selection methods (variance & correlation)
# How many variables are left?
X_clean2.shape


# In[ ]:


# The two feature selection methods have reduced the variable count from 439 to 89. 

