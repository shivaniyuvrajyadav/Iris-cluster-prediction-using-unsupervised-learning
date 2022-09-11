#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df = pd.read_csv('C:/Users/shivani/Downloads/Iris.csv')
df.head()


# In[10]:


df = df.drop(columns =['Id'])
df.head()


# In[11]:


# to display stats about data 
df.describe()


# In[12]:


df.info()


# In[13]:


# to display number of samples on each species
df['Species'].value_counts()


# In[14]:


# check  for null values
df.isnull().sum()


# In[15]:


df['SepalLengthCm'].hist()


# In[16]:


df['SepalWidthCm'].hist()


# In[17]:


df['PetalLengthCm'].hist()


# In[21]:


df['PetalWidthCm'].hist()


# In[20]:


colors = ['red','orange','blue']
Species = ['Iris-setosa','Iris-versicolor','Iris-virginica ']


# In[22]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm' ],x['SepalWidthCm'],c= colors[i],label=Species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()


# In[23]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['PetalLengthCm' ],x['PetalWidthCm'],c= colors[i],label=Species[i])
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()


# In[25]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=Species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[27]:


for i in range(3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=Species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[28]:


df.corr()


# In[29]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[34]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[35]:


from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[40]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[41]:


# model training
model.fit(x_train, y_train)


# In[ ]:





# In[39]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[42]:


# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[43]:


model.fit(x_train, y_train)


# In[44]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[46]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[47]:


model.fit(x_train, y_train)


# In[48]:


# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)

