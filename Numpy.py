#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[4]:


list1 = [0,1,2,3,4]


# In[7]:


list1


# In[9]:


arr1d = np.array(list1)


# In[11]:


arr1d


# In[13]:


list1.append(5)


# In[15]:


list1


# In[17]:


list1 + 1


# In[19]:


arr1d + 1


# In[20]:


arr1d


# In[21]:


#creating 2D array List
list2 = [[1,1,1], [2,2,2], [3,3,3]]


# In[22]:


list2


# In[24]:


#adding list into array for making two dimensional array
arr2d = np.array(list2)


# In[26]:


arr2d


# In[28]:


#for changing data type arr2d into float
arr2d = np.array(list2, dtype = 'float')


# In[30]:


arr2d


# In[34]:


#for changing element type to integer
arr2d = arr2d.astype('int')


# In[35]:


#for changing element type to string
arr2d.astype('str')


# In[38]:


#changing array to list
np2list = arr2d.tolist()


# In[39]:


np2list


# #dtypes and shapes 

# In[40]:


list2


# In[41]:


arr2d


# In[43]:


arr2d = arr2d.astype('float')


# In[44]:


arr2d


# In[46]:


#python3 we are using pranthasis, and in pyhton2 we aren't using paranthasis
print('shape :', arr2d.shape)


# In[47]:


arr2d.size


# In[48]:


arr1d.size


# In[49]:


arr2d.ndim


# In[50]:


arr1d.ndim


# In[52]:


arr1d = arr1d*arr1d


# In[53]:


arr1d


# In[54]:


arr1d[0]


# In[55]:


arr2d


# In[58]:


arr2d[2][0] #[Row][col]


# In[59]:


boolarr = arr2d<3


# In[60]:


boolarr


# In[63]:


arr2d[boolarr]


# In[64]:


#to reverse the full array
arr2d[::-1]


# In[65]:


#to reverse the rows and columns
arr2d[::-1, ::-1]


# # np.nan and np.inf

# In[66]:


np.nan


# In[67]:


np.inf


# In[68]:


arr2d


# In[69]:


arr2d[0][0] = np.nan
arr2d[0][1] = np.inf
arr2d


# In[71]:


#to check the NAN values in array
np.isnan(arr2d)


# In[72]:


#to check the INF values in array
np.isinf(arr2d)


# In[73]:


missing_flag = np.isnan(arr2d) | np.isinf(arr2d)
missing_flag


# In[74]:


arr2d[missing_flag]


# In[75]:


#missing values are filling with Zero's
arr2d[missing_flag] = 0
arr2d


# # Statistical Operations
# -mean()
# -std()
# -var()

# In[76]:


arr2d.mean()


# In[77]:


arr2d.max()


# In[78]:


arr2d.min()


# In[79]:


arr2d.std()


# In[80]:


arr2d.var()


# In[81]:


arr2d.squeeze()


# In[83]:


arr2d.cumsum()


# In[84]:


arr = arr2d[:2, :2]


# In[85]:


arr


# In[86]:


arr2d[1:3, 1:2]


# In[87]:


arr2d.reshape(1,9)


# In[89]:


arr2d.reshape(9,1)


# In[93]:


#to know the values of array and to change multi-dimenssional array to single-dimenssional array
a= arr2d.flatten()
a


# In[94]:


#to know the values of array and to change multi-dimenssional array to single-dimenssional array
b= arr2d.ravel()
b


# In[95]:


arr2d


# In[96]:


b[0] = -1


# In[97]:


arr2d


# # sequences, repitations and random numbers

# In[98]:


#in arange method we can change data type also
np.arange(1, 5, dtype = 'int')


# In[99]:


#arange method will print start and stop range with given step
np.arange(2, 50, 2)


# In[104]:


# by using linspace we can print the numbers by given range, how many numbers we are required 
#linearSpace
np.linspace(1, 50, 50)


# In[105]:


#logspace
np.logspace(1, 50, 10)


# In[107]:


np.zeros([2,2])


# In[108]:


np.ones([2,2])


# In[109]:


a = [1,2,3]


# In[110]:


a


# In[111]:


np.tile(a,3)


# In[112]:


np.repeat(a,3)


# In[113]:


np.repeat(arr2d, 3)


# In[114]:


arr2d


# In[115]:


np.random.rand(3,3)


# In[116]:


np.random.randn(3,3)


# In[117]:


np.random.randint(1, 10, [3,3])


# In[143]:


#the seed method doesnt change the random numbers
np.random.seed(0)
np.random.randint(1, 10, [3,3])


# In[144]:


np.unique(arr2d)


# In[145]:


uniques, counts = np.unique(arr2d, return_counts = True)


# In[146]:


uniques


# In[147]:


counts


# # Section - 2

# In[149]:


arr = np.array([8,94,8,56,1,3,4,5,7])
arr


# In[151]:


#for indexing create the values like this
index_gt10 = np.where(arr>10)
index_gt10


# In[152]:


arr[index_gt10]


# In[153]:


#if we create like this, we wont get index
arr[arr>10]


# In[154]:


arr>10


# In[155]:


np.where(arr>10, 'gt10', 'lt10')


# In[156]:


arr.max()


# In[157]:


arr.argmax()


# In[158]:


arr[arr.argmax()]


# In[159]:


arr[arr.argmin()]


# In[160]:


arr.argmin()


# # read and write csv file

# In[ ]:


#np.genfromtext(), np.loadtxt()


# In[166]:


data = np.genfromtxt('https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv', delimiter = ',', skip_header =1,  filling_values = -1000, dtype = 'float')


# In[168]:


data


# In[169]:


data.shape


# In[171]:


np.set_printoptions(suppress = True)
data[:3]


# In[172]:


data2 = np.genfromtxt('https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv', delimiter = ',', skip_header =1, dtype = None)
data2[:3]


# In[173]:


#For saving file
np.savetxt('data.csv', data, delimiter = ',')


# In[175]:


#for saving in array format
np.save('data.npy', data)


# In[176]:


#for saving multiple array in single file
np.savez('data2.npz', data, data2)


# In[177]:


d = np.load('data.npy')


# In[178]:


d


# In[180]:


d2 =np.load('data2.npz')


# In[181]:


d2


# In[182]:


d2.files


# In[183]:


d2['arr_0']


# In[184]:


d2['arr_1']


# # concat with row and col wise

# In[186]:


arr1 = np.zeros([4,4])
arr2 = np.ones([4,4])


# In[187]:


arr1


# In[188]:


arr2


# In[ ]:


#np.concatente, np.vstack, np.r_


# In[193]:


#we will get similar results with these methods by "Rows"
np.concatenate([arr1, arr2], axis = 0)
np.vstack([arr1, arr2])
np.r_[arr1, arr2]


# In[198]:


#we will get similar results with these methods by "Columns"
np.concatenate([arr1, arr2], axis = 1)
np.hstack([arr1, arr2])
np.c_[arr1, arr2]


# # sort a numpy array

# In[199]:


arr = np.random.randint(1, 10, size = [10,5])


# In[200]:


arr


# In[202]:


np.sort(arr, axis=0)


# In[205]:


sorted_index = arr[:, 0].argsort()


# In[207]:


arr[sorted_index]


# In[208]:


arr


# # working with Dates

# In[209]:


d = np.datetime64('2020-05-24 23:57:00')


# In[210]:


d


# In[212]:


d+1000


# In[213]:


oneday = np.timedelta64(1, 'D')


# In[214]:


d + oneday


# In[215]:


oneminute = np.timedelta64(1, 'm')


# In[216]:


d + oneminute


# In[220]:


date = np.arange(np.datetime64('2020-05-24'), np.datetime64('2021-05-24'), 2)


# In[221]:


date


# # Numpy Advanced Functions

# In[222]:


def foo(x):
    if x%2==1:
        return x**2
    else:
        return x/2


# In[223]:


foo(10)


# In[224]:


foo(11)


# In[225]:


foo_v = np.vectorize(foo, otypes=['float'])


# In[226]:


foo_v(arr)

