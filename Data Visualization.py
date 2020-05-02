#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from numpy.random import randn, randint, sample, uniform


# In[13]:


df = pd.DataFrame(randn(1000), index = pd.date_range('2020-04-25', periods = 1000), columns = ['value'])
ts = pd.Series(randn(1000), index = pd.date_range('2020-04-25', periods = 1000))


# In[14]:


df['value'] = df['value'].cumsum()
df.head()


# In[15]:


ts = ts.cumsum()
ts.head()


# In[17]:


type(df), type(ts)


# In[20]:


ts.plot(figsize = (10,5))


# In[19]:


plt.plot(ts)


# In[21]:


df.plot()


# In[22]:


iris = sns.load_dataset('iris')
iris.head()


# In[27]:


ax = iris.plot(figsize = (12,4), title = 'iris Dataset')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')


# In[28]:


ts.plot(kind = 'bar')


# In[29]:


ts.plot(kind = 'hist')


# In[31]:


df.iloc[0].plot(kind = 'bar')


# In[35]:


df1 = iris.drop(['species'], axis = 1)


# In[37]:


df1.head()


# In[39]:


df1.iloc[0].plot(kind = 'bar')


# In[41]:


#this is another method to plot histogram plots
df1.iloc[0].plot.bar()


# In[42]:


titanic = sns.load_dataset('titanic')
titanic.head()


# In[47]:


titanic['pclass'].plot(kind = 'hist')


# In[44]:


df2 = pd.DataFrame(randn(10, 4), columns = ['a', 'b', 'c', 'd'])
df2.head()


# In[46]:


df2.plot.bar(stacked = True)


# In[50]:


df2.plot(kind = 'bar', stacked = True)


# In[51]:


#for horizontal plot
df2.plot(kind = 'barh', stacked = True)


# In[52]:


iris.plot.hist()


# In[53]:


#another method
iris.plot(kind = 'hist')


# In[55]:


iris.plot(kind = 'hist', stacked = True, bins = 50)


# In[56]:


iris.plot(kind = 'hist', stacked = True, bins = 50, orientation = 'horizontal')


# In[58]:


iris['sepal_width'].diff().plot(kind = 'hist', stacked = True, bins = 50)


# In[60]:


df1.head()


# In[66]:


#alpha is transperance
df1.diff().hist(color = 'r', alpha = 0.5, figsize = (10,10))


# In[68]:


color = {'boxes' : 'DarkGreen', 'whiskers': 'r'}


# In[70]:


df1.plot(kind = 'box',figsize = (6,6), color = color)


# In[71]:


df1.plot(kind = 'box',figsize = (6,6), color = color, vert = False)


# In[74]:


df1.plot(kind = 'area', stacked = False)


# In[73]:


df1.plot.area()


# In[77]:


df1.plot.scatter(x = 'sepal_length', y = 'petal_length')


# In[76]:


df1.head()


# In[78]:


df1.plot.scatter(x = 'sepal_length', y = 'petal_length', c ='sepal_width')


# In[85]:


ax1 = df1.plot.scatter(x = 'sepal_length', y = 'petal_length', label = 'length');
df1.plot.scatter(x = 'sepal_width', y = 'petal_width', label = 'width', ax = ax1, color = 'r')


# In[86]:


df1.plot.scatter(x = 'sepal_length', y = 'petal_length', c ='sepal_width', s = df1['petal_width']*200)


# In[89]:


df1.plot.hexbin(x = 'sepal_length', y = 'petal_length', gridsize = 15, C = 'sepal_width')


# In[90]:


d = df1.iloc[0]
d


# In[94]:


d.plot.pie(figsize = (5,5))
plt.show()


# In[103]:


d1 = df1.head(3).T
d1


# In[104]:


d1.plot.pie(subplots = True, figsize = (20,20))


# In[105]:


d1.plot.pie(subplots = True, figsize = (20,20), fontsize = 15, autopct = '%.2f')


# In[106]:


[0.1]*4


# In[107]:


series = pd.Series([0.1]*4, index = ['a', 'b', 'c', 'd'])
series.plot.pie()


# In[108]:


series = pd.Series([0.2]*4, index = ['a', 'b', 'c', 'd'], name = 'Pie plot')
series.plot.pie()


# In[109]:


from pandas.plotting import scatter_matrix


# In[114]:


scatter_matrix(df1, figsize = (10,10), diagonal = 'kde', color = 'r')
plt.show()


# In[113]:


scatter_matrix(df1, figsize = (10,10))
plt.show()


# In[116]:


ts.plot.kde()


# In[117]:


from pandas.plotting import andrews_curves


# In[119]:


andrews_curves(df1, 'sepal_length')


# In[127]:


ts.plot(style  = 'rx-', label = 'series', legend = True)


# In[131]:


df1.plot(legend = False, figsize= (10,5), logy = True)


# In[132]:


x = df1.drop(['sepal_width', 'petal_width'], axis = 1)
x.head()


# In[133]:


y = df1.drop(['sepal_length', 'petal_length'], axis = 1)
y.head()


# In[144]:


x.plot(figsize = (10,3))
y.plot(figsize = (10,3), secondary_y = True)


# In[146]:


ax2 = x.plot()
y.plot(figsize = (16,3), secondary_y = True, ax = ax2)


# In[151]:


x.plot(figsize = (10,5), x_compat = True)


# In[157]:


df1.plot(subplots = True, figsize =(10,5), sharex = False)
plt.tight_layout()
plt.show()


# In[160]:


df1.plot(subplots = True, figsize =(10,5), sharex = False, layout = (2,2), )
plt.tight_layout()
plt.show()


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




