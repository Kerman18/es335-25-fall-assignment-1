import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from classification_exp import kfold_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)

########### Reading the data ###################

#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
#data = pd.read_csv(url, delim_whitespace=True, header=None,
#                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
#                        "acceleration", "model year", "origin", "car name"])

cols="displacement,cylinders,mpg,horsepower,weight,accelaration,model_year,origin,car_name".split(',')
df = pd.read_csv("auto-mpg.data", sep='\\s+', names=cols,na_values='?')
#display(df)

###### Cleaning the data ########################
#print(df.isna().sum()) #check how many NaN values are there
# Filing the NaN values with mean so it wont't effect the data, mean because horsepower is continuous feature not classification
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())
#print(df.isna().sum()) # checking if that worked, should see 0 NaN

X_auto = df.drop(columns=["mpg","car_name"]) #dropping car names because it's not that useful
y_auto = df["mpg"]

# Compare the performance of your model with the decision tree module from scikit learn

#################################################################################
###################### Decision tree with self-made algorithm ###################
#################################################################################

folds = kfold_split(X_auto, y_auto, 5, seed=42) # getting the folds
all_idx = np.arange(len(X_auto)) # array of all indices

rmse_self = [] # list to store RMSE for each fold
mae_self=[] # lsit to store MAE for each fold

for _,test_idx in enumerate(folds,start=1): # iterating over each fold
    train_idx = np.setdiff1d(all_idx,test_idx) # getting train indices by removing test indices from all indices
    
    #Train-test split
    X_auto_train, y_auto_train = X_auto.iloc[train_idx], y_auto.iloc[train_idx]
    X_auto_test, y_auto_test = X_auto.iloc[test_idx], y_auto.iloc[test_idx]
    
    # Make the decision tree with self made algorithm
    tree_self = DecisionTree(criterion=None, max_depth=5) 
    tree_self.fit(X_auto_train, y_auto_train)
    y_pred_self = tree_self.predict(X_auto_test)

    rmse_self.append(rmse(y_auto_test,y_pred_self))
    mae_self.append(mae(y_auto_test,y_pred_self))

print('---Result for self-made algorithm --- ')
print(f"At depth 5, Mean RMSE: {np.mean(rmse_self):.4f}|Mean MAE:{np.mean(mae_self):.4f} ")

########################################################################
################### Decision tree using sklearn library ################
########################################################################

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_sk, rmse_sk = [], []

for train_idx, test_idx in kf.split(X_auto):
    Xauto_train, Xauto_test = X_auto.iloc[train_idx], X_auto.iloc[test_idx]
    yauto_train, yauto_test = y_auto.iloc[train_idx], y_auto.iloc[test_idx]

    # Make decision tree using sklearn library
    tree_sk = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree_sk.fit(Xauto_train, yauto_train)
    y_pred_sk = tree_sk.predict(Xauto_test)

    mae_sk.append(mean_absolute_error(yauto_test, y_pred_sk))
    rmse_sk.append(np.sqrt(mean_squared_error(yauto_test, y_pred_sk)))

print('---Result for sklearn algorithm---')
print(f"At depth 5, Mean RMSE: {np.mean(rmse_sk):.4f}|Mean MAE:{np.mean(mae_sk):.4f} ")
