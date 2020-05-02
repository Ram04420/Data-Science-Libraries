#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


x = np.linspace(0, 10, 20)
x


# In[11]:


y = randint(1, 50, 20)
y


# In[12]:


y.size


# In[13]:


dir(plt)


# In[14]:


plt.plot(y)


# In[15]:


print(np.sort(y))


# In[16]:


y = np.sort(y)


# In[17]:


plt.plot(y)


# In[26]:


plt.plot(x,y, 'b')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Random Numbers')
plt.show()


# In[37]:


plt.subplot(1,2,1)
plt.plot(x, y, 'ro-', markersize = 8)
plt.subplot(1,2,2)
plt.plot(x, y, 'b*-')


# # Matlab vs Matplotlib

# In[41]:


fig = plt.figure()
axes = fig.add_axes([0.1,0.1,1,1])
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('set the title of Random')
plt.show()


# In[42]:


dir(axes)


# In[54]:


y2 = y*x


# In[55]:


fig = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.1, 0.5, 0.4, 0.3])

ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('set the title of Random')


ax2.plot(x, y2, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y2')
ax2.set_title('set the title of Random Y2')
plt.show()


# In[56]:


#difference between
#plt.subplot and plt.subplots


# In[61]:


fig, ax = plt.subplots(1,2)

ax[0].plot(x,y,'r')
ax[1].plot(x,y,'b')


# In[68]:


fig, ax = plt.subplots(1,2)
col = ['r', 'b']
data = [y, y2]

for i, axes in enumerate(ax):
    axes.plot(x,data[i], col[i])
fig.tight_layout()


# In[78]:


fig, ax = plt.subplots(figsize = (8,4), dpi = 100 )

ax.plot(x, y, 'r' , label = 'y')
ax.plot(x, y2, 'b' , label = 'y*x')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Random Numbers')
ax.legend()
plt.show()
fig.savefig('Randomfile.png', dpi = 100)


# In[85]:


fig, ax = plt.subplots()
ax.plot(x, y, 'bo-', markersize = 12, linewidth = 3)


# In[91]:


fig,ax = plt.subplots(1,3, figsize = (12,4))
ax[0].plot(x,y, x,y2)
ax[1].plot(x, y**2, 'k')


# In[92]:


dir(ax)


# In[93]:


dir(plt)


# In[94]:


plt.scatter(x, y)


# In[98]:


plt.bar(y, height = 1 , width =0.5)


# In[99]:


from random import sample


# In[103]:


data = sample(range(1,1000), 10)
data


# In[105]:


plt.hist(data, rwidth = 0.5)


# In[107]:


data = [np.random.normal(1, std, 100) for std in range(1,3)]


# In[111]:


plt.boxplot(data, patch_artist=True, vert = True)
plt.show()


# In[118]:


fig, ax = plt.subplots(1,2, figsize = (10,4))
ax[0].plot(x,y,x,y2)
ax[1].plot(x, np.exp(x))
ax[1].set_yscale('log')
fig.tight_layout()


# In[122]:


fig, ax = plt.subplots(figsize = (10,5))
ax.plot(x,y,x,y2)
ax.set_xticks([1,3,5,10])
ax.set_xticklabels([r'a', r'b', r'c', r'd'], fontsize =18)


# In[128]:


fig, ax = plt.subplots(figsize = (10,5))
ax.plot(x,y2)
ax.set_xticks([1,3,5,10])
ax.set_xticklabels([r'a', r'b', r'c', r'd'], fontsize =18)

plt.show()


# In[130]:


from matplotlib import ticker


# In[137]:


fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('scintific notation')

formatter = ticker.ScalarFormatter(useMathText = True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,-1))
ax.yaxis.set_major_formatter(formatter)


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




