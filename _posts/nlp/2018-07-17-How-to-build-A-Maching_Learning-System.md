---
layout: post
comments: true
categories: nlp
---

# How to build a Maching Learing system

* The goal of our experiment is to build a maching learning system to make the House-price prediction based on the given dataset
* We use imputer-method to imputer the missing value
* We transform categorical variable to binary value via one-hot encoding method
* We compare the model prediction performance of RandomForestRegressor algorithm and XGBoost algorithm
* At last,we calculate the prediction error based on data without processing so as to show the model improvment afte processing data.



###  1)Import the dataset and split it into train data and test data


```python
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/house-prices/train.csv')
# Drop houses where the target is missing
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
X_data = data.drop(['SalePrice'],axis=1)
y_data = data.SalePrice
X_train,X_test,y_train,y_test = train_test_split(X_data,
                                                y_data,
                                                train_size=0.7,
                                                test_size=0.3,
                                                random_state=0)
```



### 2)Transform categorical variable to binary value via one-hot encoding


```python
# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in X_train.columns if 
                                X_train[cname].nunique() < 10 and
                                X_train[cname].dtype == "object"]
numeric_cols = [cname for cname in X_train.columns if 
                                X_train[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
X_train_predictors = X_train[my_cols]
X_test_predictors = X_test[my_cols]

#one-hot encoding
X_train_predictors['tmp'] = 'train'
X_test_predictors['tmp'] = 'test'
concat_data = pd.concat([X_train_predictors , X_test_predictors])
features_data = pd.get_dummies(concat_data, columns=low_cardinality_cols, dummy_na=True)
# Split your data
X_train_encoded = features_data[features_data['tmp'] == 'train']
X_test_encoded = features_data[features_data['tmp'] == 'test']

# Drop your labels
X_train_encoded_predictors = X_train_encoded.drop('tmp', axis=1)
X_test_encoded_predictors = X_test_encoded.drop('tmp', axis=1)
```



### 3)Use imputer method to imputer the missing values


```python
#Use the Imputer class so you can impute missing values
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train_encoded_predictors)
imputed_X_test = my_imputer.fit_transform(X_test_encoded_predictors)
```



### 4)Define function to calculate the mean-absolute-error


```python
def cal_error(my_model,X_train,y_train,X_test,y_test):
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    model = my_model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test,predictions)
    return mae
```



### 5)In order to compare the performance between different model algorithm,we select RandomForestRegressor algorithm and XGBoost algorithm


```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
model_1 = RandomForestRegressor()
model_2 = XGBRegressor()
model_3 = DecisionTreeRegressor()
```



### 6)The final prediction mean-absolute-error


```python
mae_1 = cal_error(model_1,imputed_X_train,y_train,imputed_X_test,y_test)
mae_2 = cal_error(model_2,imputed_X_train,y_train,imputed_X_test,y_test)
print("The Mean absolute error of RandomForestRegressor algorithm is %2f" %(mae_1))
print("The Mean absolute error of XGBoost algorithm is %2f" %(mae_2))
```

    The Mean absolute error of RandomForestRegressor algorithm is 18703.810046
    The Mean absolute error of XGBoost algorithm is 16662.033319

### 7)Compare to model which only use columns with numerical and non-null value


```python
used_X_train = X_train[numeric_cols]
used_X_test = X_test[numeric_cols]
cols_with_missing = [col for col in used_X_train
                                    if used_X_train[col].isnull().any()]
X = used_X_train.drop(cols_with_missing,axis=1)
y = used_X_test.drop(cols_with_missing,axis=1)
mae_3 = cal_error(model_1,X,y_train,y,y_test)
mae_4 = cal_error(model_2,X,y_train,y,y_test) 
print("The Mean absolute error of RandomForestRegressor algorithm with numerical and non-null value is %2f" %(mae_3))
print("The Mean absolute error of XGBoost algorithm with numerical and non-null value is %2f" %(mae_4))
```

    The Mean absolute error of RandomForestRegressor algorithm with numerical and non-null value is 19547.234247
    The Mean absolute error of XGBoost algorithm with numerical and non-null value is 17180.221604


* The Mean-Absolute-Error of XGBoost algorithm is much smaller than that of RandomForestRegressor
* We use impter-method to imputer the missing values and apply one-hot encoding to transform categorical variable to binary value,in doing so,we can reserve features as much as possible.It turns out that we can recieve much better error performance based on these data than that remove the columns with non-numerical and non-null value
