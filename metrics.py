from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    
    return (y_hat == y).sum() / y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size

    TP= ((y==cls) & (y_hat==cls)).sum() # Counting True Positive cases
    FP= ((y!=cls) & (y_hat==cls)).sum() # Counting False Positive cases
    if TP+FP==0:
        return f'N/A, TP+TF=0'
    else:
        return TP/(TP+FP)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    
    TP= ((y==cls) & (y_hat==cls)).sum() # Counting True Positive cases
    FN= ((y==cls) & (y_hat!=cls)).sum() # Counting False Negative cases
    if TP+FN==0:
        return 'N/A, TP+FN=0'
    else:
        return TP/(TP+FN)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size

    return np.sqrt(((y - y_hat) ** 2).mean()) #RMSE formula


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size

    return (y - y_hat).abs().mean() #MAE formula
