#!/usr/bin/env python
# coding: utf-8

# # using cufflinks and iplot()
# `line`
# `scatter`
# `bar`
# `box`
# `spread`
# `ratio`
# `heatmap`
# `surface`
# `histogram`
# `bubble`

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib.inline', '')


# In[10]:


from plotly.offline import iplot
import plotly as py
import plotly.tools as tls


# In[11]:


import cufflinks as cf


# In[12]:


print(py.__version__)


# In[13]:


df = pd.DataFrame()
help(df.iplot)


# In[14]:


tls.embed('https://plot.ly/~cufflinks/8')


# In[17]:


py.offline.init_notebook_mode(connected = True)


# In[15]:


cf.go_offline()


# In[54]:


df = pd.DataFrame(np.random.randn(100, 3), columns = ['A', 'B', 'C'])
df['A'] = df['A'].cumsum()+20
df['B'] = df['B'].cumsum()+20
df['C'] = df['C'].cumsum()+20


# In[55]:


df.head()


# In[56]:


df.shape


# In[57]:


df.iplot()


# In[27]:


plt.plot(df)


# In[28]:


df.plot()


# In[35]:


df.iplot(x ='A', y = 'B', mode = 'markers', size = 15)


# In[36]:


titanic = sns.load_dataset('titanic')
titanic.head()


# In[39]:


titanic.iplot(x ='sex', y = 'survived', kind = 'bar', title = 'Survived', xTitle = 'Sex', yTitle = '#Survived')


# In[47]:


titanic['sex'].value_counts()


# In[50]:


cf.getThemes()


# In[53]:


cf.set_config_file(theme = 'white')
df.iplot(kind = 'bar', bargap =0.4, barmode = 'stack')


# In[45]:


df.iplot(kind = 'bar', bargap =0.4, barmode = 'stack')


# In[46]:


df.iplot(kind = 'barh', bargap =0.4, barmode = 'stack')


# In[58]:


df.iplot(kind = 'box')


# In[59]:


df.iplot()


# In[61]:


df.iplot(kind = 'area', fill = True)


# In[64]:


df3 = pd.DataFrame({'X':[10,20,30,20,10], 'Y':[10,20,30,20,10], 'Z':[10,20,30,20,10]})
df3.head()


# In[66]:


df3.iplot(kind = 'surface', colorscale ='rdylbu')


# In[67]:


help(cf.datagen)


# In[70]:


cf.datagen.sinwave(10, inc = 0.4).iplot(kind = 'surface')


# In[81]:


cf.datagen.scatter3d(2, 150, mode ='stocks').iplot(kind = 'scatter3d', x='x', y ='y', z='y')


# In[82]:


df[['A', 'B']].iplot(kind = 'spread')


# In[84]:


df.iplot(kind = 'hist', bins = 50, barmode = 'group', bargap = 0.5)


# In[85]:


cf.datagen.bubble3d(5, 4, mode = 'stocks').iplot(kind = 'bubble3d', x='x', y ='y', z='z', size = 'size')


# In[87]:


cf.datagen.heatmap(8,8).iplot(kind = 'heatmap', title = 'cufflinks - heatmap', colorscale = 'spectral')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




