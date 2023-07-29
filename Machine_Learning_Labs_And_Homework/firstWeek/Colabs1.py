#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)


# In[4]:


two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)


# In[11]:


sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)


# In[12]:


random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)


# In[13]:


random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1)


# In[14]:


random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)


# In[21]:


random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)


# In[22]:


feature = np.arange(6,21)
print(feature)
label = np.random.randint(low=6, high=21, size=[15])
print(label)


# In[23]:


noise = (np.random.random([15]) * 4) - 2
print(noise)
label = label + noise 
print(label)


# In[ ]:




