#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -r requirements.txt\n')


# In[ ]:


from sklearn.datasets import load_breast_cancer


# In[3]:


breast = load_breast_cancer()


# In[4]:


breast_data = breast.data


# In[5]:


breast_data.shape


# In[6]:


breast_labels = breast.target


# In[7]:


breast_labels.shape


# In[8]:


import numpy as np


# In[9]:


labels = np.reshape(breast_labels,(569,1))


# In[10]:


final_breast_data = np.concatenate([breast_data,labels],axis=1)


# In[11]:


final_breast_data.shape


# In[12]:


import pandas as pd


# In[13]:


breast_dataset = pd.DataFrame(final_breast_data)


# In[14]:


features = breast.feature_names


# In[15]:


features


# In[16]:


features_labels = np.append(features,'label')


# In[17]:


breast_dataset.columns = features_labels


# In[18]:


breast_dataset.head()


# In[19]:


breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[20]:


breast_dataset.tail()


# In[ ]:


from keras.datasets import cifar10


# In[ ]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[ ]:


(x_train, y_train), (x_test, y_test) = (x_train[:800], y_train[:800]), (x_test[:200], y_test[:200])


# In[ ]:


print('Traning data shape:', x_train.shape)
print('Testing data shape:', x_test.shape)


# In[25]:


y_train.shape,y_test.shape


# In[26]:


# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


label_dict = {
 0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck',
}


# In[29]:


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(x_train[0], (32,32,3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_train[0][0]]) + ")"))

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(x_test[0],(32,32,3))
plt.imshow(curr_img)
print(plt.title("(Label: " + str(label_dict[y_test[0][0]]) + ")"))


# In[30]:


from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features


# In[31]:


x.shape


# In[32]:


np.mean(x),np.std(x)


# In[33]:


feat_cols = ['feature' + str(i) for i in range(x.shape[1])]


# In[34]:


normalised_breast = pd.DataFrame(x,columns=feat_cols)


# In[35]:


normalised_breast.tail()


# In[36]:


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)


# In[37]:


principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])


# In[38]:


principal_breast_Df.tail()


# In[39]:


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# In[40]:


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']

for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15});


# In[41]:


np.min(x_train),np.max(x_train)


# In[42]:


x_train = x_train/255.0


# In[43]:


np.min(x_train),np.max(x_train)


# In[44]:


x_train.shape


# In[45]:


x_train_flat = x_train.reshape(-1,3072)


# In[46]:


feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])]


# In[47]:


df_cifar = pd.DataFrame(x_train_flat,columns=feat_cols)


# In[48]:


df_cifar['label'] = y_train
print('Size of the dataframe: {}'.format(df_cifar.shape))


# In[49]:


df_cifar.head()


# In[50]:


pca_cifar = PCA(n_components=2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:1000,:-1])


# In[51]:


principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar
             , columns = ['principal component 1', 'principal component 2'])
principal_cifar_Df['y'] = y_train


# In[52]:


principal_cifar_Df.head()


# 
# 

# <div>
# <style scoped>
#     .dataframe tbody tr th:only-of-type {
#         vertical-align: middle;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# 
#     .dataframe thead th {
#         text-align: right;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>principal component 1</th>
#       <th>principal component 2</th>
#       <th>y</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>-6.401018</td>
#       <td>2.729039</td>
#       <td>6</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>0.829783</td>
#       <td>-0.949943</td>
#       <td>9</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>7.730200</td>
#       <td>-11.522102</td>
#       <td>9</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>-10.347817</td>
#       <td>0.010738</td>
#       <td>4</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>-2.625651</td>
#       <td>-4.969240</td>
#       <td>1</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 

# In[53]:


print('Explained variation per principal component: {}'.format(pca_cifar.explained_variance_ratio_))


# In[54]:


import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=principal_cifar_Df,
    legend="full",
    alpha=0.3
)


# In[55]:


x_test = x_test/255.0


# In[56]:


x_test = x_test.reshape(-1,32,32,3)


# Let's ``reshape`` the test data.
# 

# In[57]:


x_test_flat = x_test.reshape(-1,3072)


# In[58]:


pca = PCA(0.9)


# In[59]:


pca.fit(x_train_flat)


# In[60]:


pca.n_components_


# In[61]:


train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)


# In[62]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop


# In[63]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[64]:


batch_size = 128
num_classes = 10
epochs = 20


# In[65]:


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(83,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[66]:


model.summary()


# In[67]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(train_img_pca, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(test_img_pca, y_test))


# In[68]:


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3072,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train_flat, y_train,batch_size=batch_size,epochs=epochs,verbose=1,
                    validation_data=(x_test_flat, y_test))

