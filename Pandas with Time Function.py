#!/usr/bin/env python
# coding: utf-8

# # Working with Date and Time function

# In[1]:


import pandas as pd
import datetime as dt


# # Review Python's `datetime` Module

# In[4]:


today = dt.date(2020, 5, 26)


# In[8]:


today.day
today.month
today.year


# In[10]:


today = dt.datetime(2020, 5, 26, 9,29,30)


# In[11]:


print(today)


# In[14]:


today.hour
today.minute
today.second


# ### `Timestamp`

# In[22]:


pd.Timestamp('2020-05-26')
pd.Timestamp('2020/05/26')
pd.Timestamp('2020 05 26')
pd.Timestamp('2020-05-26 7:25:16 PM')


# ### `DateTimeIndex`

# In[23]:


dates = ['2020-05-26','2020-05-27','2020-05-28',]


# In[26]:


x=pd.DatetimeIndex(dates)


# In[27]:


type(x)


# In[28]:


dates = [dt.date(2020,5,26), dt.date(2020,5,27),dt.date(2020,5,28)]


# In[29]:


dates


# In[30]:


dateindex = pd.DatetimeIndex(dates)
dateindex


# In[33]:


values = [12,13,10]
pd.Series(data = values, index = dateindex)


# ### `pd.to_datetime()`

# In[34]:


pd.to_datetime('2020-05-26')


# In[35]:


pd.to_datetime(dt.date(2020, 5, 26))


# In[38]:


pd.to_datetime(dt.datetime(2020, 5, 26, 10, 13, 25))


# In[39]:


dates = ['2020-05-26','2020-05-27','2020-05-28',]
pd.to_datetime(dates)


# In[54]:


dates = pd.Series(['May 26th, 2020', '2020, 12, 26', 'This is the Date', 'May 23rd, 2020', '22nd dec, 2020'])


# In[55]:


dates


# In[56]:


pd.to_datetime(dates, errors ='coerce')


# In[57]:


unixtime = [12345678, 87654321,25863147,87412365]


# In[58]:


pd.to_datetime(unixtime, unit = 's')


# # creating Range in the form Date `pd.date_range()`

# In[61]:


times = pd.date_range(start = '2020-05-26', end = '2020-08-26', freq = 'D' )


# In[62]:


times


# In[65]:


times = pd.date_range(start = '2020-05-26', end = '2020-08-26', freq = 'H' )
times


# ## import Stock Data using datareader

# In[68]:


from pandas_datareader import data


# In[71]:


company = 'MSFT'
start = '2020-01-01'
end = '2020-05-26'


# In[72]:


stock = data.DataReader(name = company, data_source = 'yahoo', start = start, end = end)


# In[74]:


stock.head(10)


# In[75]:


stock.loc['2020-01-06']


# In[76]:


stock.iloc[50]


# In[77]:


stock.loc['2020-01-05' : '2020-02-05']


# In[79]:


stock.truncate(before = '2020-04-01', after = '2020-04-10')


# # `Timedelta`

# In[84]:


timeA = pd.Timestamp('2020-05-26 12:02:40')
timeB = pd.Timestamp('2020-05-27 15:10:50')


# In[85]:


timeB-timeA


# In[87]:


pd.Timedelta(weeks = 10, days = 2, hours =10, minutes = 50, seconds = 12)


# In[91]:


df = pd.read_csv('ecommerce.csv', index_col = 'ID',  parse_dates =['order_date', 'delivery_date'])


# In[92]:


df.head()


# In[93]:


df['Delivery Time'] = df['delivery_date'] - df['order_date']


# In[94]:


df.head()


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




