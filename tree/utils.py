"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np
from typing import Union

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # Check if it's numeric
    if not pd.api.types.is_numeric_dtype(y):
        return False
    
    # Check if it's boolean or categorical, it's classification
    if pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return False
    
    # Get unique values
    unique_vals = y.unique()
    n_unique = len(unique_vals) #coounting unique values
    
    # Check for binary classification (0 or 1 type data)
    if n_unique <= 2:
        return False
    
    # If there are very few unique values relative to the total number of samples then it's likely classification case
    unique_ratio = n_unique / len(y) # checking the ratio of unique values
    if unique_ratio < 0.25:  # Less than 25% unique values suggests classification, valaue is choosen arbitarily
        return False
    
    return True  # If none of the above conditions are met, assume regression
    


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    p=Y.value_counts(normalize=True)
    p = p[p > 0] # Removing zero probabilities to avoid log2(0)
    return -(p*np.log2(p)).sum() #Entropy formula


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    p=Y.value_counts(normalize=True)
    return 1-(p**2).sum() #Gini index formula


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """
    return ((Y - Y.mean()) ** 2).mean()  # MSE formula


def information_gain(Y: pd.Series, attr: pd.Series, criterion: Union[str, None]) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    Here, calculating information gain when attribute is discreate
    For real-valued attribute, calculation is done in the function opt_split_attribute 
    """
    assert criterion in ['information_gain', 'gini_index', None],f"Expecting 'information_gain' or 'gini_index' for classification and {None} for regeresion, got {criterion} instead"
    assert Y.size == attr.size, 'Size of Y and attribute is not same' # makes sure that Y and attr are of same size

    Y_len = len(Y)
    weighted_entropy = 0
    weighted_gini=0
    weighted_mse = 0

    if check_ifreal(Y): # regression
        Y_mse = mse(Y)
        for v in attr.unique():
            Y_sub = Y[attr == v]
            weighted_mse += (len(Y_sub) / Y_len) * mse(Y_sub)
        return Y_mse - weighted_mse #MSE reduction formula
    else: 
        if criterion=='information gain': # classification with information gain
            for v in attr.unique():
                Y_sub = Y[attr == v]
                weighted_entropy += (len(Y_sub) / Y_len) * entropy(Y_sub)
            return entropy(Y) - weighted_entropy #Information gain formula
        else: #classification with gini index
            for v in attr.unique():
                Y_sub = Y[attr == v]
                weighted_gini += (len(Y_sub) / Y_len) * gini_index(Y_sub)
            return  gini_index(Y) - weighted_gini

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion:Union[str,None], features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon and the value of that attribute to split upon
    """
    
    assert criterion in ['information_gain', 'gini_index', None],f"Expecting 'information_gain' or 'gini_index':for classification and {None} for regeresion, got {criterion} instead"

    best_attr = None
    best_score = -float('inf') # starting with the lowest possible value because if all gains are negative still want to return the best one
    best_val = None

    for attr in features:
        col = X[attr]

        if check_ifreal(col): #real-valued feature
            unique_vals = np.sort(col.unique())
            if len(unique_vals) <= 1: #skiping features with only one unique value
                continue
            
            # midpoints between consecutive values as potential splits
            potential_splits = [(unique_vals[i] + unique_vals[i+1]) / 2 for i in range(len(unique_vals) - 1)]

            for t in potential_splits: #calculating score for each potential split and choosing the best one
                left= col <= t
                right= col > t
                left_y, right_y = y[left], y[right]

                if len(left_y) == 0 or len(right_y) == 0: # skiping splits that result in empty nodes
                    continue

                if check_ifreal(y):  # regression problem, so use MSE reduction
                    score = mse(y) - (len(left_y)/len(y))*mse(left_y) - (len(right_y)/len(y))*mse(right_y)
                else:  # classification, so use information gain or gini gain
                    if criterion == "information_gain":
                        score = entropy(y) - (len(left_y)/len(y))*entropy(left_y) - (len(right_y)/len(y))*entropy(right_y)
                    else:
                        score = gini_index(y) - (len(left_y)/len(y))*gini_index(left_y) - (len(right_y)/len(y))*gini_index(right_y)

                if score > best_score:
                    best_score, best_attr, best_val = score, attr, t
        else: # discreate features
            score = information_gain(y, col, criterion)
            if score > best_score:
                best_score, best_attr, best_val = score, attr, None

    return best_attr, best_val


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    col = X[attribute]
    if check_ifreal(col) and value is not None: #split in left and right for real-valued feature
        left = col <= value
        right = col > value
        return (X[left], y[left]), (X[right], y[right])
    else:  #split in multiple vaues for discrete feature 
        splits = {}
        for v in col.unique():
            mask = col == v
            splits[v] = (X[mask], y[mask])
        return splits
