"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import *
from metrics import *

np.random.seed(42)
# Test case 1
# Real Input and Real Output
print('########### Real input & Real Output ###############')
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P),columns=[f"X{i+1}" for i in range(P)])
y = pd.Series(np.random.randn(N))

tree = DecisionTree(criterion=None,max_depth=5)  # Split based on MSE reduction
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
#print("Criteria :", 'MSE')
print("RMSE: ", round(rmse(y_hat, y),4))
print(f"MAE: ", round(mae(y_hat, y),4))
print('###################################################### \n')

# Test case 2
# Real Input and Discrete Output
print('########### Real input & Discrete Output ###############')
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P),columns=[f"X{i+1}" for i in range(P)])
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print('\n')
    tree = DecisionTree(criterion=criteria,max_depth=5)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Class:{cls}:")
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))
        print('\n')

print('###################################################### \n')

# Test case 3
# Discrete Input and Discrete Output
print('########### Discrete input & Discrete Output ###############')
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in [f"X{i+1}" for i in range(5)]})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    print('\n')
    tree = DecisionTree(criterion=criteria,max_depth=5)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print(f"Class:{cls}:")
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))
        print('\n')

print('###################################################### \n')

# Test case 4
# Discrete Input and Real Output
print('########### Discrete input & Real Output ###############')
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in [f"X{i+1}" for i in range(5)]})
y = pd.Series(np.random.randn(N))

tree = DecisionTree(criterion=None,max_depth=5)  # Split based on MSE reduction
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
#print("Criteria :", 'MSE')
print("RMSE: ", round(rmse(y_hat, y),4))
print(f"MAE: ", round(mae(y_hat, y),4))
