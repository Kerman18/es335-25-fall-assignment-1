import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
#plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X, columns=["X1", "X2"])
y = pd.Series(y)

################################################################
############# Testing self made algorithm ######################
################################################################

print('############ (a)Testing the self made algorithm ################ \n')

def train_test_split(X:pd.DataFrame, y:pd.Series, test_size:float, seed:int):
    """
    Function to split the data into train and test sets
    """
    assert 0 < test_size < 1, "test_size should be between 0 and 1"
    np.random.seed(seed)

    idx = np.arange(len(X)) # array of indices
    np.random.shuffle(idx) # shuffling the indices
    split = int( len(X)* (1 - test_size)) # calculating the split index
    train_idx, test_idx = idx[:split], idx[split:] # splitting the indices
    return X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]

### testing the algorithm

## train-test split
X_train,Y_train,X_test,y_test=train_test_split(X,y,test_size=0.3,seed=42)

## Training the decision tree
tree = DecisionTree(criterion="information_gain", max_depth=5)
tree.fit(X_train, Y_train)

y_pred=tree.predict(X_test) # Predicting on the test set


## Evaluating the model
print('---Results for train-test split---')
print("Train Accuracy: ", round(accuracy(tree.predict(X_train), Y_train),4))
print("Test Accuracy: ", round(accuracy(y_pred, y_test),4))
for cls in y.unique():
    print(f"Class:{cls}:")
    print("Precision: ", round(precision(y_pred, y_test, cls),4))
    print("Recall: ", round(recall(y_pred, y_test, cls),4))

print('\n')

######################################################
############## Cross-Validation ######################
######################################################

print('############## (b) Cross- Validation ##############')

############# 5 fold cross-validation #################

def kfold_split(X:pd.DataFrame, y:pd.Series, k:int, seed:int):
    """
    Function to get indices for k-fold
    """
    assert k > 1, "k should be greater than 1"
    #assert len(y)%k==0, "Number of samples should be divisible by k"

    np.random.seed(seed)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    fold_size = len(X) // k
    folds = [idx[i*fold_size:(i+1)*fold_size] for i in range(k)]
    return folds

print('###### 5 fold CV #######')

folds = kfold_split(X, y, 5, seed=42) # getting the folds
all_idx = np.arange(len(X)) # array of all indices

acc_scores = [] # list to store accuracy for each fold

for _,test_idx in enumerate(folds,start=1): # iterating over each fold
    train_idx = np.setdiff1d(all_idx,test_idx) # getting train indices by removing test indices from all indices
    
    #Train-test split
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Training the decision tree
    tree = DecisionTree(criterion='information_gain', max_depth=5) 
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    acc_scores.append(accuracy(y_test,y_pred))

print('--- Result for 5 fold CV---')
print(f"Mean Accuracy: {np.mean(acc_scores):.4f} at depth 5")
print('\n')

################# Nested cross-validation #####################

print('### Nested CV ###')

outer_k=5 #number outer folds
inner_k=5 #number inner folds
candidate_depths=np.array([1,2,3,4,5,6])
seed=42

outer_fold=kfold_split(X, y, k=outer_k, seed=seed) #outer fold
outer_scores,chosen_depths = [],[]
all_idx = np.arange(len(X)) #all indices of X

for _,test_idx in enumerate(outer_fold,start=1):
    tem_idx = np.setdiff1d(all_idx,test_idx) 
    #Test-temperory split
    X_tem,y_tem = X.iloc[tem_idx],y.iloc[tem_idx]
    X_test,y_test = X.iloc[test_idx],y.iloc[test_idx]

    best_depth,best_cv = None,-1

    # Inner CV to pick best depth
    for d in candidate_depths:
        inner_fold=kfold_split(X_tem,y_tem,inner_k,seed=42) #inner fold
        temp_ind=np.arange(len(X_tem)) # all indices of X_temp
        inner_score=[]

        for _,val_idx in enumerate(inner_fold,start=1): 
            train_idx = np.setdiff1d(tem_idx,val_idx) #validation incies
            
            #Train-validation split
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx] 
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            #Tarining the tree
            tree = DecisionTree(criterion='information_gain', max_depth=d) 
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_val)

            inner_score.append(accuracy(y_val,y_pred))
        if np.mean(inner_score) > best_cv:
            best_cv,best_depth = np.mean(inner_score),d

    chosen_depths.append(best_depth)
    tree=DecisionTree(criterion="information_gain", max_depth=best_depth)
    tree.fit(X_tem,y_tem) #train on X_train+X_val set
    y_final_pred = tree.predict(X_test) #prediction of X_test set
    
    outer_scores.append(accuracy(y_final_pred, y_test))

print('---Result for Nested CV---')
print(f"Mean accuracy:{np.mean(outer_scores):.4f}|Depths choosen at each outer fold:{np.array(chosen_depths).astype(int)}")
