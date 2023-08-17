#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor
data.head()


# In[31]:


# %%
M = data[data.diagnosis == "B"]
B = data[data.diagnosis == "M"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()
data.head()


# In[32]:


# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[33]:


# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[34]:


#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state = 42)


# In[35]:


# %% SVM

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[36]:


# %% test
print("score: ", dt.score(x_test,y_test))


# In[37]:


y_pred = dt.predict(x_test)  
y_pred


# In[38]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




