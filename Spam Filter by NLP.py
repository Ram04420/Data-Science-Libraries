#!/usr/bin/env python
# coding: utf-8

# # NLP
# Natural Language Processing. it's field of Artificial Intelligence

# # Applications of NLP
# `-Text classifications
# -Spam filters
# -Voicetext messaging
# -Sentiment analysis
# -Spell or Grammer check
# -chat bot
# -Search Suggestions
# -Search Auto correct
# -Automatic Review Analysis System
# -Machine Translation
# -And many more`
# 
# `-Natural Language understanding(Text Classification)` |
# `-Natural Language Genration(Text Generation)`

# # The Process of NLP (Text Classifiaction)
# #Raw Speech Signal
#     -Speech Recognition
# #Sequence of Words spoken
#     -syntactic Analysis using knowledge of the grammar
# #Stracture of the Grammar
#     -semantic analysis using info.about meaning of words
# #Partial Representaion of meaning of sentance
#     -Pragamatic analysis using info. about context
# #Final Representation of meaning of sentance
#  

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


df = pd.read_csv('spam.tsv.txt', sep ='\t')


# In[12]:


df.head()


# In[14]:


df.isnull().sum()


# In[15]:


len(df)


# In[16]:


df['label'].value_counts()


# # Balancing the data

# In[17]:


ham = df[df['label']=='ham']
ham.head()


# In[18]:


spam = df[df['label']=='spam']
spam.head()


# In[20]:


ham.shape, spam.shape


# In[21]:


ham = ham.sample(spam.shape[0])


# In[22]:


ham.shape


# In[23]:


ham.shape, spam.shape


# In[25]:


data = ham.append(spam, ignore_index = True)


# In[27]:


data.head()


# # Exploratory Data Analysis

# In[32]:


plt.hist(data[data['label']=='ham']['length'], bins = 100, alpha =0.7)
plt.hist(data[data['label']=='spam']['length'], bins = 100, alpha =0.7)
plt.show()


# In[33]:


plt.hist(data[data['label']=='ham']['punct'], bins = 100, alpha =0.7)
plt.hist(data[data['label']=='spam']['punct'], bins = 100, alpha =0.7)
plt.show()


# # Data Preparation

# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[45]:


data.head()


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size =0.3, random_state =0, shuffle = True,stratify=data['label'] )


# In[50]:


x_train.head()


# ### Bag of word creation

# In[51]:


vectorizer = TfidfVectorizer()


# In[52]:


x_train = vectorizer.fit_transform(x_train)


# In[53]:


x_train


# ## Pipeline and RF

# In[57]:


clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', RandomForestClassifier(n_estimators =100, n_jobs = -1))])


# In[58]:


clf.fit(x_train, y_train)


# In[59]:


y_pred = clf.predict(x_test)


# In[60]:


confusion_matrix(y_test, y_pred)


# In[61]:


print(classification_report(y_test, y_pred))


# In[62]:


accuracy_score(y_test, y_pred)


# In[63]:


clf.predict(['hi, this is ram'])


# In[67]:


clf.predict(['hey ram, you won free tickets for usa summer tour'])


# ## Svm

# In[69]:


clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', SVC(C =1000, gamma = 'auto'))])


# In[70]:


clf.fit(x_train, y_train)


# In[71]:


y_pred = clf.predict(x_test)


# In[72]:


confusion_matrix(y_test, y_pred)


# In[73]:


print(classification_report(y_test, y_pred))


# In[74]:


accuracy_score(y_test, y_pred)


# In[75]:


clf.predict(['hi, this is ram'])


# In[76]:


clf.predict(['hey ram, you won free tickets for usa summer tour'])


# In[ ]:




