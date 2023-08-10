#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report




# In[43]:


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv("C:\\Users\\engab\\Downloads\\archive\\diabetes.csv", header=None, names=col_names)


# In[44]:


pima.head()


# In[45]:


feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
X =X[1:]
y = pima.label # Target variable
y=y[1:]


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


# In[47]:


logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)


# In[48]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[49]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[50]:


target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[52]:


# Convert string labels to numeric (0 and 1)
y_test_numeric = y_test.apply(lambda x: 1 if x == '1' else 0)

# Calculate y_pred_proba and ROC curve
y_pred_proba = logreg.predict_proba(X_test)[:, 1]  # Probability of class 1
fpr, tpr, _ = metrics.roc_curve(y_test_numeric, y_pred_proba)
auc = metrics.roc_auc_score(y_test_numeric, y_pred_proba)

# Plot ROC curve
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




