#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


data = pd.read_csv(r'C:\Users\engab\Downloads\archive/column_2C_weka.csv')


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.info()


# In[8]:


Normal = data[data["class"] == "Normal"]
Abnormal = data[data["class"] == "Abnormal"]


# In[9]:


x_data = data.drop(["class"],axis=1)
y = [1 if each == "Abnormal" else 0 for each in data["class"]]


# In[10]:


x_data_min = x_data.min(axis=0)  # Calculate minimum along columns (for each column)
x_data_max = x_data.max(axis=0)  # Calculate maximum along columns (for each column)
x = (x_data - x_data_min) / (x_data_max - x_data_min)


# In[11]:


x


# In[12]:


from sklearn.model_selection import train_test_split


x_train,x_test, y_train ,y_test = train_test_split(x,y,test_size= 0.2, random_state=42)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 1)
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)


# In[14]:


print("{} knn score : {}".format(3,knn.score(x_test,y_test)))


# In[15]:


#Find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors =each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))


# In[16]:


plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In[ ]:




