#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('iris.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


df =df.drop(columns= ['Id'], axis= 1)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.tail()


# # pre-processing

# In[8]:


df.isnull().sum()


# # Exploratory Data Analysis

# In[9]:


df['SepalLengthCm'].hist()


# In[10]:


df['SepalWidthCm'].hist()


# In[11]:


df['PetalLengthCm'].hist()


# In[12]:


df['PetalWidthCm'].hist()


# # Correlation matrix

# In[13]:


df.corr()


# In[14]:


corr= df.corr()
fig, ax= plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True,ax =ax,cmap='coolwarm')


# # Label Encoder

# In[15]:


#from sklearn.preprocessing import LabelEncoder


# In[16]:


#le=LabelEncoder()
#df['Species']= le.fit_transform(df['Species'])


# In[17]:


df


# # Model Training

# In[18]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[19]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[20]:


model.fit(x_train, y_train)


# In[21]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


# In[23]:


model.fit(x_train, y_train)


# In[24]:


print("Accuracy: ",model.score(x_test, y_test)* 100)


# In[25]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[26]:


model.fit(x_train, y_train)


# In[27]:


print("Accuracy: ",model.score(x_test, y_test)* 100)


# # Saving the model

# In[28]:


import pickle
filename = 'savedmodel.sav'
pickle.dump(model, open(filename,'wb'))


# In[31]:


x_test.head()


# In[32]:


load_model = pickle.load(open(filename,'rb'))


# In[33]:


load_model.predict([[5.8,2.7,5.1,1.9]])


# In[ ]:




