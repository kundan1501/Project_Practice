#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv")
df.head()


# In[6]:


df.keys


# In[7]:


len(df)


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


sns.scatterplot(x="Country",y="Happiness Rank",data=df)


# In[12]:


sns.scatterplot(x='Happiness Score',y='Family',data=df)


# In[13]:


sns.pairplot(df)
plt.savefig("pairplot.png")
plt.show()


# In[14]:


df.corr()


# In[15]:


df.corr()["Happiness Score"].sort_values()


# In[16]:


df.corr()["Dystopia Residual"]


# In[17]:


df["Family"].plot(kind="hist")


# In[18]:


df["Happiness Score"].plot(kind="hist")


# In[19]:


sns.boxplot(x="Happiness Score",y="Country",data=df)
plt.title=("dystopia leve")


# In[20]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,linewidth=.5,linecolor="green",fmt='.2f')


# In[21]:


plt.figure(figsize=(15,8))
sns.heatmap(df.describe().transpose(),annot=True,linewidth=.5,linecolor="orange",fmt=".2f")
plt.xticks(fontsize=18)
plt.yticks(fontsize=15)
plt.show()


# In[22]:


import warnings
warnings.filterwarnings('ignore')

sns.relplot("Happiness Score","Country",data=df)


# In[23]:


sns.lineplot("Happiness Score","Country",data=df)


# In[24]:


sns.boxplot("Happiness Score",data=df)


# In[25]:


sns.boxplot(x="Country",y="Happiness Score",data=df)


# In[26]:


df.plot.bar()


# In[27]:


df.plot.bar(stacked=True)


# In[28]:


df.plot.box()


# In[29]:


df.skew()


# In[30]:


sns.distplot(df["Happiness Score"])


# Removing Outlier

# In[48]:


df.isnull().sum()


# In[49]:


df.info()


# In[31]:


# finding outliers
df.plot(kind='box',subplots=True,layout=(4,5),figsize=(15,7))


# In[32]:


df.corr()["Happiness Score"]


# In[33]:


df.skew()


# In[34]:


sns.distplot(df["Standard Error"])


# In[35]:


sns.distplot(df["Economy (GDP per Capita)"])


# In[36]:


sns.distplot(df["Family"])


# In[37]:


sns.distplot(df["Health (Life Expectancy)"])


# In[39]:


sns.distplot(df["Freedom"])


# In[40]:


sns.distplot(df["Trust (Government Corruption)"])


# In[41]:


sns.distplot(df["Generosity"])


# In[42]:


sns.distplot(df["Dystopia Residual"])


# In[50]:


df = df.drop(np.where(df['Standard Error'] > 0.06)[0])


# In[51]:


df = df.drop(np.where(df['Trust (Government Corruption)'] > 0.5)[0])


# In[52]:


df.info()


# In[53]:


df['Country'].value_counts()


# In[54]:


df['Region'].value_counts()


# In[55]:


df = pd.get_dummies(df, columns = ['Region', 'Country'])
print(df)


# In[56]:


from scipy.stats import zscore
from scipy import stats
import numpy as np
z=np.abs(zscore(df))
z


# In[57]:


df.skew()


# In[58]:


threshold=3
print(np.where(z>3))


# In[59]:


df_new = df[(z < 3).all(axis=1)]


# In[60]:


print("old dataframe",df.shape)
print("newdataframe",df_new.shape)
print("drop totall no of row droped",df.shape[0]-df_new.shape[0])


# In[62]:


total_loss=(129-0)/129*100
total_loss


# In[63]:


df_new


# In[64]:


df.isnull().sum()


# In[65]:


df_new.isnull().sum()


# In[ ]:




