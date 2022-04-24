#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats.mstats import normaltest
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Salary.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# To see unique values in the columns
print(df['rank'].unique())
print(df['discipline'].unique())
print(df['sex'].unique())


# In[7]:


print(df['rank'].value_counts())
print(df['discipline'].value_counts())
print(df['sex'].value_counts())


# In[9]:


# to convert catagorical data in numders
oe = OrdinalEncoder()

def ordinal_encode(df, column = ['rank','discipline','sex']):
    df[column] = oe.fit_transform(df[column])
    return df

oe_col = df.columns
df=ordinal_encode(df, oe_col)
df.head()


# In[12]:


df.columns


# In[13]:


df.shape


# Visualize Data

# In[14]:


fig = plt.figure(figsize = (20,20))
sns.pairplot(df[['yrs.since.phd', 'yrs.service', 'salary']])


# In[15]:


fig = plt.figure(figsize = (15,7))
sns.countplot(data = df)


# In[18]:


columns = ['rank', 'discipline', 'yrs.since.phd', 'yrs.service', 'sex', 'salary']
plt.figure(figsize = (20,50))
for i in range(len(columns)):
    plt.subplot(8,2, i+1)
    sns.distplot(df[columns[i]],color = 'r');
    plt.title(columns[i])
plt.tight_layout()


# In[20]:


corr_matrix = df.corr()
corr_matrix


# In[21]:


plt.figure(figsize = (10,10))
sns.heatmap(corr_matrix,annot = True)


# In[23]:


# splitting the data into independent and dependent datasets
x = df.drop(['salary'],axis = 1)
y = df['salary']


# In[24]:


# testing to accpect null hypothsis or not
normaltest(df.salary.values)


# In[25]:


#spliting dataset into training(70%) and testing(30%)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=45)


# In[26]:


xtrain.shape


# In[27]:


ytrain.shape


# In[30]:


error_df = list()

lr = LinearRegression()
lr = lr.fit(xtrain,ytrain)
ytrain_pred = lr.predict(xtrain)
ytest_pred = lr.predict(xtest)

error_df.append(pd.Series({'train': mean_squared_error(ytrain,ytrain_pred),
                          'test':mean_squared_error(ytest,ytest_pred)},name='ordinal_encode'))


error_df = pd.concat(error_df,axis=1)
error_df


# In[33]:


# scaling the data(feature scaling)
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)


# Model-Making

# In[35]:


lr = LinearRegression()
lr.fit(xtrain,ytrain)
lrpred = lr.predict(xtest)
r2_score(lrpred,ytest)


# In[36]:


dtr = DecisionTreeRegressor()
dtr.fit(xtrain,ytrain)
dtrpred = dtr.predict(xtest)
r2_score(dtrpred,ytest)


# In[37]:


rfr = RandomForestRegressor()
rfr.fit(xtrain,ytrain)
rfrpred = rfr.predict(xtest)
r2_score(rfrpred,ytest)


# In[ ]:




