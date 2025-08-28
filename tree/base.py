"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


class Node:
    def __init__(self, is_leaf=False, prediction=None,
                 feature=None, threshold=None, children=None):
        self.is_leaf = is_leaf # check if the node is a leaf node
        self.prediction = prediction # prediction value for leaf nodes
        self.feature = feature # feature to split upon
        self.threshold = threshold # threshold for real-valued feature
        self.children = children or {}  # dict for categorical or {left,right} for real


@dataclass
class DecisionTree:
    criterion: Literal["entropy", "gini_index",None]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth):
        self.criterion = criterion 
        self.max_depth = max_depth 
        self.root = None # root node of the tree
        self.is_regression = None #check if target is regression or classification

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
         Function to train and construct the decision tree
        """
        self.is_regression = check_ifreal(y) # checking if target is regression or classification
        features = X.columns #list of all features
        self.root = self._build(X, y, features, depth=0) #building the tree

    def _build(self, X, y, features, depth):
        if len(y) == 0: # no samples
            return None

        # stop when max depth reached or pure leaf
        if depth >= self.max_depth or len(y.unique()) == 1:
            prediction = round(y.mean(),4) if self.is_regression else y.mode()[0] # mean for regression, mode for classification
            return Node(is_leaf=True, prediction=prediction)

        # choose best attribute
        attr, val = opt_split_attribute(X, y, self.criterion, features)
        if attr is None: # no valid split found
            prediction = round(y.mean(),4) if self.is_regression else y.mode()[0]
            return Node(is_leaf=True, prediction=prediction)

        # numeric feature gives binary split (left & right)
        if val is not None: 
            (X_left, y_left), (X_right, y_right) = split_data(X, y, attr, val)
            left_child = self._build(X_left, y_left, features, depth + 1) # recurse for left child
            right_child = self._build(X_right, y_right, features, depth + 1) # recurse for right child
            return Node(feature=attr, threshold=val, children={"left": left_child, "right": right_child})

        # categorical feature give multiple split depending on number of unique values
        else:
            splits = split_data(X, y, attr, None)
            children = {}
            for v, (X_sub, y_sub) in splits.items(): # recurse for each split
                children[v] = self._build(X_sub, y_sub, features, depth + 1)
            return Node(feature=attr, children=children)

    def predict(self, X: pd.DataFrame) -> pd.Series: # predicting for multiple rows
        preds = []
        for _, row in X.iterrows(): # iterating over each row
            preds.append(self._predict_row(row, self.root)) # predicting for each row
        return pd.Series(preds, index=X.index)

    def _predict_row(self, row, node: Node):
        if node.is_leaf:
            return node.prediction # return prediction at leaf

        if node.threshold is not None:  # numeric split
            if row[node.feature] <= node.threshold: # go left
                return self._predict_row(row, node.children["left"])
            else: # go right
                return self._predict_row(row, node.children["right"])
        else:  # categorical split
            val = row[node.feature] # get feature value
            if val in node.children: # if value seen during training
                return self._predict_row(row, node.children[val]) 
            else:
                # unseen category then fallback to majority prediction
                return node.prediction

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self._print_tree(self.root)

    def _print_tree(self, node: Node, depth=0):
        indent = "    " * depth
        if node is None: # empty node
            print(f"{indent}[Empty Node]")
            return

        if node.is_leaf: # leaf node
            if isinstance(node.prediction, float): # for regression
                pred_str = f"{node.prediction:.4f}"
            else: # for classification
                pred_str = str(node.prediction)
            print(f"{indent}Predict -> {pred_str}")

        elif node.threshold is not None:  # numeric feature
            print(f"{indent}?({node.feature} <= {node.threshold:.4f})")
            print(f"{indent}Y:")
            self._print_tree(node.children["left"], depth + 1)
            print(f"{indent}N:")
            self._print_tree(node.children["right"], depth + 1)

        else:  # categorical feature
            print(f"{indent}?({node.feature})")
            for v, child in node.children.items():
                print(f"{indent}{v}:")
                self._print_tree(child, depth + 1)
