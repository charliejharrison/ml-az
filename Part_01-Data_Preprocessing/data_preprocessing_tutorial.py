# -*- coding: utf-8 -*-
"""
Cmd + i to inspect
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Dealing with missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', 
                  strategy='mean',
                  axis=0)
imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

## Encoding categorical data
# LabelEncoder encodes unique values as contiguous integers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:, 0] = labelencoder_country.fit_transform(X[:, 0])

# OneHotEncoder converts categorical variables to one-hot using dummy variables
onehotencoder = OneHotEncoder(categorical_features=[0], sparse=False)
X = onehotencoder.fit_transform(X) #.toarray() - needed is sparse is not set to False

# This is equivalent to binariser here
labelencoder_purchased = LabelEncoder()
y = labelencoder_purchased.fit_transform(y)

## Splitting dataset into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=0)

## Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# We don't need rescale the dummy variables
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc_X.transform(X_test[:, 3:])

