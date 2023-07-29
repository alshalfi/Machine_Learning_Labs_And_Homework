#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Call fetch_housing_data() to download and extract the data
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()


# In[2]:


housing.info()


# In[3]:


import pandas as pd

# Assuming 'housing' DataFrame has been loaded properly

# Extract the 'ocean_proximity' column and count occurrences of each unique value
ocean_proximity_counts = housing["ocean_proximity"].value_counts()

# Display the result in a more readable format
print("Number of occurrences of each unique value in 'ocean_proximity' column:")
print(ocean_proximity_counts)


# In[4]:


housing.describe()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline # only in a Jupyter notebook')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[6]:


#matplotlib inline # only in a Jupyter notebook
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


import numpy as np
def split_train_test(data, test_ratio):
 shuffled_indices = np.random.permutation(len(data))
 test_set_size = int(len(data) * test_ratio)
 test_indices = shuffled_indices[:test_set_size]
 train_indices = shuffled_indices[test_set_size:]
 return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
16512
len(test_set)
4128


# In[9]:


from zlib import crc32
def test_set_check(identifier, test_ratio):
 return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# In[10]:


def split_train_test_by_id(data, test_ratio, id_column):
 ids = data[id_column]
 in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
 return data.loc[~in_test_set], data.loc[in_test_set]


# In[11]:


housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[12]:


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[14]:


housing["income_cat"] = pd.cut(housing["median_income"],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])


# In[15]:


housing["income_cat"].hist()


# In[19]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
 strat_train_set = housing.loc[train_index]
 strat_test_set = housing.loc[test_index]


# In[20]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[21]:


for set_ in (strat_train_set, strat_test_set):
 set_.drop("income_cat", axis=1, inplace=True)


# In[22]:


housing = strat_train_set.copy()


# In[23]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[24]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[25]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[26]:


corr_matrix = housing.corr()


# In[27]:


import pandas as pd

# Assuming 'housing' DataFrame has been loaded properly

# Calculate the correlation matrix with 'numeric_only=True' to avoid the FutureWarning
corr_matrix = housing.corr(numeric_only=True)

# Display the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

corr_matrix = housing.corr()


# In[28]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[29]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[30]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
 alpha=0.1)


# In[31]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[32]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[33]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[34]:


housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[36]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[37]:


imputer.fit(housing_num)


# In[38]:


imputer.statistics_


# In[39]:


housing_num.median().values


# In[40]:


X = imputer.transform(housing_num)


# In[41]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
 index=housing_num.index)


# In[42]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[43]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[44]:


ordinal_encoder.categories_


# In[45]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[46]:


housing_cat_1hot.toarray()


# In[47]:


cat_encoder.categories_


# In[49]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[50]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[51]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)


# In[52]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[53]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))


# In[54]:


print("Labels:", list(some_labels))


# In[55]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[56]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[57]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[58]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
 scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[67]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[70]:


from sklearn.model_selection import cross_val_score
import numpy as np

# Assuming `lin_reg` is a valid trained linear regression model
# and `housing_prepared` and `housing_labels` are appropriately prepared data and corresponding labels.

# Perform cross-validation using the negative mean squared error as the scoring metric.
# cv=10 specifies that the data is split into 10 folds for cross-validation.
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)

# Calculate the root mean squared error (RMSE) scores by taking the square root of the negative mean squared error scores.
lin_rmse_scores = np.sqrt(-lin_scores)

# Function to display the scores, mean, and standard deviation of the RMSE values obtained during cross-validation.
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Call the display_scores function to print the results of the cross-validation.
display_scores(lin_rmse_scores)


# In[73]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming `housing_prepared` and `housing_labels` are appropriately prepared data and corresponding labels.

# Create a RandomForestRegressor model
forest_reg = RandomForestRegressor()

# Train (fit) the model with the training data
forest_reg.fit(housing_prepared, housing_labels)

# Use the model to make predictions on the training data
housing_predictions = forest_reg.predict(housing_prepared)

# Calculate the mean squared error (MSE) between the predictions and the actual labels
forest_mse = mean_squared_error(housing_labels, housing_predictions)

# Calculate the root mean squared error (RMSE)
forest_rmse = np.sqrt(forest_mse)

# Now you can print or use the `forest_rmse` variable
print("Random Forest RMSE:", forest_rmse)


# In[74]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Assuming `housing_prepared` and `housing_labels` are appropriately prepared data and corresponding labels.

# Create a RandomForestRegressor model
forest_reg = RandomForestRegressor()

# Perform cross-validation using the negative mean squared error as the scoring metric.
# cv=10 specifies that the data is split into 10 folds for cross-validation.
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)

# Calculate the root mean squared error (RMSE) scores by taking the square root of the negative mean squared error scores.
forest_rmse_scores = np.sqrt(-forest_scores)

# Function to display the scores, mean, and standard deviation of the RMSE values obtained during cross-validation.
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Call the display_scores function to print the results of the cross-validation.
display_scores(forest_rmse_scores)


# In[75]:


import joblib
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
 return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(np.sqrt(-mean_score), params)


# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[ ]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[76]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Assuming you have defined 'full_pipeline', 'housing_prepared', 'housing_labels', and performed any necessary preprocessing.

# Define the parameter grid for the Random Forest model
param_grid = {
    'n_estimators': [50, 100, 150],  # example values for the number of trees
    'max_depth': [None, 10, 20],     # example values for the maximum depth of trees
    # add more hyperparameters as needed
}

# Create a RandomForestRegressor model
forest_reg = RandomForestRegressor()

# Create the GridSearchCV object with the RandomForestRegressor and the parameter grid
grid_search = GridSearchCV(estimator=forest_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the grid search object to the training data to find the best hyperparameters
grid_search.fit(housing_prepared, housing_labels)

# Access the best model from the grid search results
final_model = grid_search.best_estimator_


# In[ ]:


from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
         loc=squared_errors.mean(),
         scale=stats.sem(squared_errors)))


# In[ ]:




