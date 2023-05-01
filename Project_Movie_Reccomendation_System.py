#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


dataset=pd.read_csv("movies.csv")


# In[3]:


dataset.head(5)


# In[21]:


dataset.shape()


# Feature Selection

# In[4]:


sel_features=['genres','keywords','original_title','vote_average','cast','director']


# In[5]:


for f in sel_features:
  dataset[f]=dataset[f].fillna('')


# In[6]:


dataset = dataset.astype({'vote_average':'float'})


# In[7]:


combined_features=dataset['genres']+' '+dataset['keywords']+' '+dataset['original_title']+' '+dataset['cast']+' '+dataset['director']


# In[8]:


print(combined_features)


# In[9]:


vectorizer=TfidfVectorizer()
feature_vextors=vectorizer.fit_transform(combined_features)


# In[10]:


print(feature_vextors)


# In[11]:


print(feature_vextors.shape)


# In[12]:


col=['vote_average','popularity']
for c in col:
  dataset[c]=dataset[c]  / dataset[c].abs().max()


# In[13]:


print(dataset['vote_average'])


# In[14]:


import scipy as sp


# **Computing Cosine Similarity **

# In[15]:


movie_name="iron man"
movie_names=dataset['title'].tolist()


# **Finding Closed match**

# In[16]:


closed_matches=difflib.get_close_matches(movie_name,movie_names)


# In[17]:


print(closed_matches)


# In[18]:


close_match=closed_matches[0]


# In[19]:


index= dataset[dataset.title==close_match]['index'].values[0]


# In[20]:


similarity=list(enumerate(similarity_score[index]))


# In[ ]:


sorted_similarity = sorted(similarity,key=lambda x:x[1], reverse=True)


# In[ ]:


print("Reccomended Movies \n")
for i in range(1,11):
  print(i,".",dataset.loc[sorted_similarity[i][0]].title)


# **Reccomendation based on movie name**

# In[ ]:


movie_name=input('Enter movi title : ')
movie_names=dataset['title'].tolist()

closed_matches=difflib.get_close_matches(movie_name,movie_names)
close_match=closed_matches[0]
index= dataset[dataset.title==close_match]['index'].values[0]

similarity=list(enumerate(similarity_score[index]))
sorted_similarity = sorted(similarity,key=lambda x:x[1], reverse=True)

print("Reccomended Movies \n")
for i in range(1,11):
  print(i,".",dataset.loc[sorted_similarity[i][0]].title)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




