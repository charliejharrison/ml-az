# -*- coding: utf-8 -*-
"""
Section 2 - Data preprocessing

Cmd + i to inspect
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Splitting dataset into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=0)
"""
## Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
"""
