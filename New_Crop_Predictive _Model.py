#!/usr/bin/env python
# coding: utf-8

# In[296]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# In[297]:


# Load the crop_data table into a pandas DataFrame
crop_data = pd.read_csv('new_cropdata.csv')


# In[298]:


print(crop_data)


# In[299]:


# One-hot encode the categorical variable
crop_data = pd.get_dummies(crop_data, columns=['Varieties of Crops grown'])


# In[300]:


# Split the data into features and labels
X = crop_data.drop(['Varieties of Crops grown_Maize', 'Varieties of Crops grown_Rice'], axis=1)
y = crop_data[['Varieties of Crops grown_Maize', 'Varieties of Crops grown_Rice']]


# In[301]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[302]:


# Train a multi-output regression model on the training set
reg = MultiOutputRegressor(LinearRegression())
reg.fit(X_train, y_train)


# In[303]:


# Make predictions on the testing set
y_pred = reg.predict(X_test)


# In[304]:




# Evaluate the model on the testing set
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MAE:', mae)
print('MSE:', mse)
print('R2 Score:', r2)


# In[307]:


# Save the model to a file
filename = 'New_crop_prediction_model.sav'
joblib.dump(reg, filename)


# In[ ]:




