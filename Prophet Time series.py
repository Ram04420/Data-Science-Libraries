#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


from fbprophet import Prophet


# In[5]:


import plotly


# In[7]:


df = pd.read_csv('C:/Users/Admin/Desktop/example_wp_log_peyton_manning.csv')


# In[8]:


df.head()


# In[11]:


m = Prophet(daily_seasonality=True)
m.fit(df)


# In[12]:


future =m.make_future_dataframe(periods = 365)
future.tail()


# In[13]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[14]:


fig1 = m.plot(forecast)


# In[15]:


fig2 = m.plot_components(forecast)


# In[16]:


from fbprophet.plot import plot_plotly, go
import plotly.offline as py
py.init_notebook_mode()
import plotly.graph_objs as go


fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)

