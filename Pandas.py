#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


data = {'apple': [3,1,4,5],
       'orange': [1,5,6,8]}


# In[5]:


data


# In[6]:


type(data)


# In[8]:


df = pd.DataFrame(data)


# In[10]:


df.head()


# In[12]:


df = pd.read_csv('nba.csv')


# In[14]:





# In[15]:


df = pd.read_csv('nba.csv', index_col = 'Name')


# In[16]:


df.head()


# In[21]:


df = pd.read_csv('IMDB-Movie-Data.csv', index_col = 'Rank')


# In[23]:


df.head()


# In[25]:


df.info()


# In[27]:


df.shape


# In[30]:


#checking duplicates with sum method
sum(df.duplicated())


# In[32]:


df1 = df.append(df)


# In[33]:


df1.shape


# In[34]:


df1.duplicated().sum()


# In[35]:


df.duplicated().sum()


# In[36]:


df2 = df1.drop_duplicates()


# In[38]:


df2.shape


# In[39]:


df1.duplicated().sum()


# In[40]:


df1.shape


# In[45]:


#inplace option must be true untill than its wasn't remove the duplicates permanently from dataset
df1.drop_duplicates(inplace = True)


# In[46]:


df1.shape


# In[53]:


#cleaning up columns

col = df.columns


# In[49]:


len(df.columns)


# In[50]:


df.describe()


# In[55]:


len(col)


# In[56]:


type(list(col))


# In[57]:


type(col)


# In[58]:


#changing the column names
col1 = ['a','b','c','d','e','f','g','h','i','j','k']


# In[59]:


#changing column names
df.columns = col1


# In[60]:


df.head()


# In[61]:


df.columns = col


# In[62]:


df.head()


# In[63]:


df.rename(columns = {'Runtime (Minutes)':'Runtime',
                    'Revenue (Millions)': 'Revenue'}, inplace = True)


# In[65]:


df.columns


# In[66]:


col


# In[67]:


import numpy as np


# In[68]:


np.nan


# In[69]:


df.isnull()


# In[70]:


df.isnull().sum()


# In[76]:


#dropping rows
df1 = df.dropna()
df1


# In[72]:


df1.shape


# In[73]:


df.shape


# In[75]:


#dropping columns
df2 = df.dropna(axis =1)
df2.shape


# In[78]:


#filling the missing values with zero by fillna method
df3 = df.fillna(0)


# In[79]:


df3.isna().sum()


# # impution

# In[80]:


df.isnull().sum()


# In[81]:


revenue = df['Revenue']


# In[82]:


type(revenue)


# In[84]:


revenue_mean = revenue.mean()
revenue_mean


# In[88]:


revenue.fillna(revenue_mean, inplace = True)


# In[91]:


df['Revenue'] = revenue


# In[92]:


df.isnull().sum()


# In[93]:


metascore  = df['Metascore']


# In[94]:


metascore_mean = metascore.mean()
metascore_mean


# In[95]:


metascore.fillna(metascore_mean, inplace = True)


# In[96]:


metascore.isnull().sum()


# In[97]:


df['Metascore'] = metascore


# In[98]:


df.isnull().sum()


# In[99]:


df.describe()


# In[100]:


df.info()


# In[102]:


df['Genre'].describe()


# In[104]:


df['Genre'].value_counts().head()


# In[106]:


len(df['Genre'].unique())


# # Corr Method

# In[107]:


corrmat = df.corr()


# In[108]:


corrmat


# In[109]:


import seaborn as sns


# In[110]:


sns.heatmap(corrmat)


# In[112]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.inline', '')


# In[115]:


df.plot(kind = 'scatter', x = 'Rating', y = 'Revenue', title = 'Revenue vs Rating')


# In[116]:


df['Rating'].plot(kind = 'hist', title = 'Rating')


# In[117]:


df['Rating'].value_counts()


# In[118]:


df['Rating'].plot(kind = 'box')


# In[119]:


df['Rating'].describe()


# In[120]:


rating_cat= []
for rate in df['Rating']:
    if rate > 6.5:
        rating_cat.append('Good')
    else:
        rating_cat.append('Bad')


# In[121]:


rating_cat


# In[122]:


df['Rating Category'] = rating_cat


# In[123]:


df.head()


# In[124]:


df.boxplot(column = 'Revenue', by = 'Rating Category')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




