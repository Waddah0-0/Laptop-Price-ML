import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

data= pd.read_csv("E:\Waddah\Alex\data_sets\laptop_pricing_dataset.csv")
data.info()
#print(data.isnull().sum())
y= data.Price
x= data.drop(columns=["Price"])
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)
low_cardinality_col= [col for col in x.columns if x[col].nunique()<10 and x_train[col].dtype == "object"]
num_col = [col for col in x.columns if x[col].dtype in ["int64","float64"] and col not in ["Price"]]
columns= low_cardinality_col+num_col
x_train= x_train[columns]
x_test= x_test[columns]
print(x_train.isnull().sum())
#__________________________________________________________________________________________________________________
#adding function to calculate mean absolute error for our models
def mae(x_train,x_test,y_train,y_test):
    """
    we gonna use random forest regressor to calculate the mean absolute error
    cause it's a good model for this kind of data
    and it's not overfitting/underfitting
    Usage Example:
    >>> mae(x_train,x_test,y_train,y_test)
    >>> 244.7
    """
    model= RandomForestRegressor(n_estimators=100,random_state=0)
    model.fit(x_train, y_train)
    pred= model.predict(x_test)
    return mean_absolute_error(y_test,pred)

#__________________________________________________________________________________________________________________

#so as we can see that there are some missing values in the our data
#there is three ways dealing with it
#fisrt one is to drop the rows with missing values
#second one is to fill it with mean,median,mode, frequent value/constant
#and third one is to fill it with one of above methods and mark the filling gap by true( for fill_value) and false( for the original value)
# ...and thats done via adding a new column to the data frame holding the true/false values
#__________________________________________________________________________________________________________________
#droping the rows with missing values
#x_train.dropna(inplace=True) 
#x_test.dropna(inplace=True)
#print(x_train.isnull().sum())
#print(x_test.isnull().sum())
#filling gaps with mean and that what we gonna do here 
imputer= SimpleImputer(strategy="mean")
x_train[num_col]= imputer.fit_transform(x_train[num_col])
x_test[num_col]= imputer.transform(x_test[num_col])
print(x_train.isnull().sum())
print(x_test.isnull().sum())
#__________________________________________________________________________________________________________________
# transforming the object columns to numerical columns is done by several methods
# drop the object columns or label encoding (ordinal encoding/ one hot encoding)
#droping columns
#reduced_x_train= x_train.select_dtypes(exclude=["object"])
#reduced_x_test= x_test.select_dtypes(exclude=["object"])
#print(reduced_x_train.head())
#print(reduced_x_test.head())
#print('mean absolute error: ',mae(reduced_x_train,reduced_x_test,y_train,y_test))
#from the line above the mean absolute error is 247.5
# we're not doing these cause it leads to data loss and we need to keep all the data
#however heres a way of doing it without data loss
encoder= OneHotEncoder(handle_unknown='ignore',sparse_output=False)
encoded_x_train_cols= pd.DataFrame(encoder.fit_transform(x_train[low_cardinality_col]))
encoded_x_test_cols= pd.DataFrame(encoder.transform(x_test[low_cardinality_col]))

encoded_x_train_cols.index= x_train.index
encoded_x_test_cols.index= x_test.index

num_train_col= x_train.drop(columns=low_cardinality_col)
num_test_col= x_test.drop(columns=low_cardinality_col)

encoded_x_train= pd.concat([num_train_col, encoded_x_train_cols], axis=1)
encoded_x_test= pd.concat([num_test_col, encoded_x_test_cols], axis=1)

encoded_x_train.columns= encoded_x_train.columns.astype(str)
encoded_x_test.columns= encoded_x_test.columns.astype(str)
print('mean absolute error: ',mae(encoded_x_train,encoded_x_test,y_train,y_test))
#from the line above the mean absolute error is 244.7 which is better than the previous one
#__________________________________________________________________________________________________________________
#creating the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(encoded_x_train, y_train)
predictions= model.predict(encoded_x_test)
score= mean_absolute_error(y_test, predictions)
#print('xgboost', score)
#print('mean absolute error: ',mae(encoded_x_train,encoded_x_test,y_train,y_test))
#__________________________________________________________________________________________________________________
#plotting our model
plt.figure(figsize=(10, 8))
plt.scatter(y_test, predictions, alpha=0.5, color='blue', label='Predictions')

min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

plt.xlabel('Actual Prices', fontsize=12)
plt.ylabel('Predicted Prices', fontsize=12)
plt.title('Actual vs Predicted Prices', fontsize=14, pad=15)

plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(fontsize=10)

correlation = np.corrcoef(y_test, predictions)[0,1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# for error distribution
plt.figure(figsize=(10, 6))
errors = predictions - y_test
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#__________________________________________________________________________________________________________________
# for model saving and loading
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Save the model
joblib.dump(model, 'saved_models/laptop_price_model.joblib')

# Save preprocessing objects
joblib.dump(imputer, 'saved_models/imputer.joblib')
joblib.dump(encoder, 'saved_models/encoder.joblib')

print("\nModel and preprocessing objects have been saved in the 'saved_models' directory")

def load_and_predict(new_data):
    """
    Load the saved model and preprocessing objects and make predictions on new data
    
    Parameters:
    new_data (DataFrame): New data to make predictions on
    
    Returns:
    array: Predicted prices

    Usage Example:
    >>> new_data = pd.read_csv('path_to_new_data.csv')
    >>> predictions = load_and_predict(new_data)
    >>> print('Predicted prices:', predictions)
    """
    # Load the saved objects
    loaded_model = joblib.load('saved_models/laptop_price_model.joblib')
    loaded_imputer = joblib.load('saved_models/imputer.joblib')
    loaded_encoder = joblib.load('saved_models/encoder.joblib')
    
    
    new_data_imputed = loaded_imputer.transform(new_data[num_col])
    new_data[num_col] = new_data_imputed
    
    encoded_new_cols = pd.DataFrame(loaded_encoder.transform(new_data[low_cardinality_col]))
    encoded_new_cols.index = new_data.index
    
    num_new_col = new_data.drop(columns=low_cardinality_col)
    encoded_new_data = pd.concat([num_new_col, encoded_new_cols], axis=1)
    encoded_new_data.columns = encoded_new_data.columns.astype(str)
    
    predictions = loaded_model.predict(encoded_new_data)
    
    return predictions







