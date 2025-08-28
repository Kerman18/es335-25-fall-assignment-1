import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import time
import scienceplots
plt.style.use(['science','notebook','grid']) # To make plot looks better

np.random.seed(42)
#num_average_time = 100  # Number of times to run each experiment to calculate the average values

# Function to create fake data (take inspiration from usage.py)
def generate_data(N, M, target="classification", feature_is_real=True):
    """
    N = number of samples
    M = number of features
    target = "classification" or "regression"
    feature_is_real = if True, generate continuous features; else discreate features
    """
    if feature_is_real:
        X = pd.DataFrame(np.random.randn(N, M),columns=[f"X{i+1}" for i in range(M)])
    else:
        X=pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in [f"X{i+1}" for i in range(M)]})

    if target == "classification":
        y = pd.Series(np.random.randint(2, size=N), dtype="category")
    else:
        y = pd.Series(np.random.randn(N))

    return X, y


# Function to calculate average time (and std) taken by fit() and predict() for different N and M for 4 different cases of DTs
def benchmark_tree(N_list, M_list, target='classification', feature_is_real=True,num_average_time=1):
    fit_times_avg = pd.DataFrame(index=N_list, columns=M_list, dtype=float)
    pred_times_avg = pd.DataFrame(index=N_list, columns=M_list, dtype=float)
    
    fit_times_std = pd.DataFrame(index=N_list, columns=M_list, dtype=float)
    pred_times_std = pd.DataFrame(index=N_list, columns=M_list, dtype=float)

    for M in M_list:
        for N in N_list:
            X, y = generate_data(N, M, target=target, feature_is_real=feature_is_real) #generating data
            if target=='classification':
                tree = DecisionTree(max_depth=5, criterion='information_gain')
            else:
                tree=DecisionTree(max_depth=5,criterion=None)

            # Measure fit timing
            fit_elapsed = []
            for _ in range(num_average_time):
                start = time.perf_counter()*1000 # to get time in miliseconds
                tree.fit(X, y)
                end = time.perf_counter()*1000
                fit_elapsed.append(end - start)
            fit_times_avg.loc[N,M] = np.mean(fit_elapsed)
            fit_times_std.loc[N,M]=np.std(fit_elapsed)

            # Measure predict time
            pred_elapsed = []
            for _ in range(num_average_time):
                start = time.perf_counter() * 1000
                tree.predict(X)
                end = time.perf_counter()*1000
                pred_elapsed.append(end - start)
            pred_times_avg.loc[N,M] = np.mean(pred_elapsed)
            pred_times_std.loc[N,M]=np.std(pred_elapsed)

    return fit_times_avg,fit_times_std,pred_times_avg, pred_times_std


# Function to plot the results
import matplotlib.pyplot as plt

def plot_results(N_list, M_list, fit_time_avg, fit_time_std, pred_time_avg, pred_time_std,title=''):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1st subplot: Fit avg
    for M in M_list:
        axs[0, 0].plot(N_list, fit_time_avg[M], label=f"M={M}")
    axs[0, 0].set_title("Fit Time (Avg)")
    axs[0, 0].set_xlabel("N (samples)")
    axs[0, 0].set_ylabel("Time (ms)")
    axs[0, 0].legend(loc='upper right')

    # 2nd subplot: Fit std
    for M in M_list:
        axs[0, 1].plot(N_list, fit_time_std[M], label=f"M={M}")
    axs[0, 1].set_title("Fit Time (Std)")
    axs[0, 1].set_xlabel("N (samples)")
    axs[0, 1].set_ylabel("Time (ms)")
    axs[0, 1].legend()

    # 3rd subplot: Predict avg
    for M in M_list:
        axs[1, 0].plot(N_list, pred_time_avg[M], label=f"M={M}")
    axs[1, 0].set_title("Prediction Time (Avg)")
    axs[1, 0].set_xlabel("N (samples)")
    axs[1, 0].set_ylabel("Time (ms)")
    axs[1, 0].legend()

    # 4th subplot: Predict std
    for M in M_list:
        axs[1, 1].plot(N_list, pred_time_std[M], label=f"M={M}")
    axs[1, 1].set_title("Prediction Time (Std)")
    axs[1, 1].set_xlabel("N (samples)")
    axs[1, 1].set_ylabel("Time (ms)")
    axs[1, 1].legend()

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1)

    plt.tight_layout()
    plt.show()

#########################################
############## Plots ####################
#########################################

N_list=np.arange(50,500,50)
M_list = np.arange(5,25,5)

### Discrete input discrete output
DIDO_fit_time_avg,DIDO_fit_time_std,DIDO_pred_time_avg, DIDO_pred_time_std = benchmark_tree(N_list, M_list, target='classfication', feature_is_real=False,num_average_time=25)

plot_results(N_list, M_list, DIDO_fit_time_avg, DIDO_fit_time_std, DIDO_pred_time_avg, DIDO_pred_time_std,title='DIDO')

### Discrete input real output
DIRO_fit_time_avg,DIRO_fit_time_std,DIRO_pred_time_avg, DIRO_pred_time_std = benchmark_tree(N_list, M_list, target='regression', feature_is_real=False,num_average_time=25)

plot_results(N_list, M_list, DIRO_fit_time_avg, DIRO_fit_time_std, DIRO_pred_time_avg, DIRO_pred_time_std,title='DIRO')

### Changing N & M because taking too much time to give results
N_list_diff=np.arange(50,201,50)
M_list_diff = np.arange(2,11,2)

### Real input discrete output
RIDO_fit_time_avg,RIDO_fit_time_std,RIDO_pred_time_avg, RIDO_pred_time_std = benchmark_tree(N_list, M_list, target='classfication', feature_is_real=True,num_average_time=10)

plot_results(N_list_diff, M_list_diff, RIDO_fit_time_avg, RIDO_fit_time_std, RIDO_pred_time_avg, RIDO_pred_time_std,title='RIDO')

### real input real output
RIRO_fit_time_avg,RIRO_fit_time_std,RIRO_pred_time_avg, RIRO_pred_time_std = benchmark_tree(N_list, M_list, target='regression', feature_is_real=True,num_average_time=10)

plot_results(N_list_diff, M_list_diff, RIRO_fit_time_avg, RIRO_fit_time_std, RIRO_pred_time_avg, RIRO_pred_time_std,title='RIRO')
