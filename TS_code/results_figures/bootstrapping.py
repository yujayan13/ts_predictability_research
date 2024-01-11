import pandas as pd
import numpy as np
from bstrap import bootstrap, boostrapping_CI
import glob

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
import scipy.stats as measures

#bootstrap function from: https://github.com/fpgdubost/bstrap
def boostrapping_CI(metric, data,observes,nbr_runs=1000, verbose=False):

    if verbose:
        print("Computing bootstrap confidence intervals...")

    nbr_scans = len(data.index)
    list_results = []
    # compute metric for each bootstrapped subset
    for r in range(nbr_runs):
        # sample random indexes
        ind = np.random.randint(nbr_scans ,size=nbr_scans)

        # select random subset
        data_bootstrapped = data.iloc[ind]
        observed_bootstrapped = observes.iloc[ind]

        # compute metrics
        result = metric(data_bootstrapped,observed_bootstrapped)
        list_results.append(result)

    # store variable in dictionary
    metric_stats = dict()
    metric_stats['avg_metric'] = np.average(list_results)
    metric_stats['metric_ci_lb'] = np.percentile(list_results, 5)
    metric_stats['metric_ci_ub'] = np.percentile(list_results, 95)
    
    if verbose:
        print("Bootstrap confidence intervals computed.")

    return metric_stats


def bootstrap(metric, data_method1, data_method2, observe1, observe2, nbr_runs=100000, compute_bounds=True, verbose=False):

    # reset index
    data_method1_reindexed = data_method1.reset_index(drop=True)
    data_method2_reindexed = data_method2.reset_index(drop=True)
    observe1_reindexed = observe1.reset_index(drop=True)
    observe2_reindexed = observe2.reset_index(drop=True)


    # get length of each data
    n = len(data_method1_reindexed.index)
    m = len(data_method2_reindexed.index)
    total = n + m

    # compute the metric for both methods
    result_method1 = metric(data_method1_reindexed,observe1)
    result_method2 = metric(data_method2_reindexed,observe2)

    # compute statistic t
    t = abs(result_method1 - result_method2)    
    
    
    # merge data from both methods
    data = pd.concat([data_method1_reindexed, data_method2_reindexed])
    data_observe = pd.concat([observe1_reindexed,observe2_reindexed])
    

    # compute bootstrap statistic
    if verbose:
        print("Computing bootstrap test...")
    nbr_cases_higher = 0
    for r in range(nbr_runs):
        # sample random indexes with replacement
        ind = np.random.randint(total, size=total)

        # select random subset with replacement
        data_bootstrapped = data.iloc[ind]
        data_observed = data_observe.iloc[ind]
        

        # split into two groups
        data_bootstrapped_x = data_bootstrapped[:n]
        data_bootstrapped_y = data_bootstrapped[n:]
        data_observe_x = data_observed[:n]
        data_observe_y = data_observed[n:]

        # compute metric for both groups
        result_x = metric(data_bootstrapped_x,data_observe_x)
        result_y = metric(data_bootstrapped_y,data_observe_y)

        # compute bootstrap statistic
        t_boot = abs(result_x - result_y)


        # compare statistics
        if t_boot > t:
            nbr_cases_higher += 1

    p_value = nbr_cases_higher * 1. / nbr_runs

    if verbose:
        print("Bootstrap test computed.")

    if not compute_bounds:
        return p_value
        
    else:
        # compute CI and means
        stats_method1 = boostrapping_CI(metric, data_method1,observe1, nbr_runs, verbose)
        stats_method2 = boostrapping_CI(metric, data_method2,observe2, nbr_runs, verbose)

        return stats_method1, stats_method2, p_value
 


months = ["march","april","may","june","july","august"]



res_m = {}
res_l = {}

for month in months:
    res_m[month] = np.load(fr"C:\ts_research\results_{month}_ts.npz")
    res_l[month] = np.load(fr"C:\ts_research\results_lin_{month}_ts.npz")

cnn_dict = {}
lin_dict = {}
pval_dict = {}


def MSSS(method,observations):
    MSE_point = metrics.mean_squared_error(method,observations)
    total = 0
    divider = 0
    o_mean = np.mean(observations)
    for observe in observations:
        total+=np.square((observe-o_mean))
        divider+=1
    MSE_O_point = total/divider
    msss_point = 1-(MSE_point/MSE_O_point)
    return(msss_point)

#run bootstrapping for MSSS for each month and point in US
for month in months:
    predictions_cnn = res_m[month]["pred"].reshape(691,960)
    correct_cnn = res_m[month]["y_test"]
    predictions_linear = res_l[month]["pred"]
    correct_linear = res_l[month]["y"]
    
    temp_cnn = []
    temp_lin = []
    temp_pval = []
    print(month)

    for point in range(691): #755
        data_dict = {"cnn_pred":predictions_cnn[point],"true_cnn":correct_cnn[point],"lin_pred":predictions_linear[point],"true_lin":correct_linear[point]}
        df = pd.DataFrame(data_dict)
        
        stats_method1, stats_method2, p_value = bootstrap(MSSS,df["cnn_pred"],df["lin_pred"],df["true_cnn"],df["true_lin"], nbr_runs=1000)
        
        temp_cnn.append(stats_method1)
        temp_lin.append(stats_method2)
        temp_pval.append(p_value)
        print(point)
    
    pval_dict[month] = temp_pval
    cnn_dict[month] = temp_cnn
    lin_dict[month] = temp_lin








np.savez("bootstrap_march_ts",pval = np.array(pval_dict["march"]),cnn_mean = np.array([a["avg_metric"] for a in cnn_dict["march"]]),lin_mean = np.array([a["avg_metric"] for a in lin_dict["march"]]), cnn_lbs = np.array([a["metric_ci_lb"] for a in cnn_dict["march"]]),lin_lbs = np.array([a["metric_ci_lb"] for a in lin_dict["march"]]),cnn_ubs = np.array([a["metric_ci_ub"] for a in cnn_dict["march"]]),lin_ubs = np.array([a["metric_ci_ub"] for a in lin_dict["march"]]))
np.savez("bootstrap_april_ts",pval = np.array(pval_dict["april"]),cnn_mean = np.array([a["avg_metric"] for a in cnn_dict["april"]]),lin_mean = np.array([a["avg_metric"] for a in lin_dict["april"]]), cnn_lbs = np.array([a["metric_ci_lb"] for a in cnn_dict["april"]]),lin_lbs = np.array([a["metric_ci_lb"] for a in lin_dict["april"]]),cnn_ubs = np.array([a["metric_ci_ub"] for a in cnn_dict["april"]]),lin_ubs = np.array([a["metric_ci_ub"] for a in lin_dict["april"]]))
np.savez("bootstrap_may_ts",pval = np.array(pval_dict["may"]),cnn_mean = np.array([a["avg_metric"] for a in cnn_dict["may"]]),lin_mean = np.array([a["avg_metric"] for a in lin_dict["may"]]), cnn_lbs = np.array([a["metric_ci_lb"] for a in cnn_dict["may"]]),lin_lbs = np.array([a["metric_ci_lb"] for a in lin_dict["may"]]),cnn_ubs = np.array([a["metric_ci_ub"] for a in cnn_dict["may"]]),lin_ubs = np.array([a["metric_ci_ub"] for a in lin_dict["may"]]))
np.savez("bootstrap_june_ts",pval = np.array(pval_dict["june"]),cnn_mean = np.array([a["avg_metric"] for a in cnn_dict["june"]]),lin_mean = np.array([a["avg_metric"] for a in lin_dict["june"]]), cnn_lbs = np.array([a["metric_ci_lb"] for a in cnn_dict["june"]]),lin_lbs = np.array([a["metric_ci_lb"] for a in lin_dict["june"]]),cnn_ubs = np.array([a["metric_ci_ub"] for a in cnn_dict["june"]]),lin_ubs = np.array([a["metric_ci_ub"] for a in lin_dict["june"]]))
np.savez("bootstrap_july_ts",pval = np.array(pval_dict["july"]),cnn_mean = np.array([a["avg_metric"] for a in cnn_dict["july"]]),lin_mean = np.array([a["avg_metric"] for a in lin_dict["july"]]), cnn_lbs = np.array([a["metric_ci_lb"] for a in cnn_dict["july"]]),lin_lbs = np.array([a["metric_ci_lb"] for a in lin_dict["july"]]),cnn_ubs = np.array([a["metric_ci_ub"] for a in cnn_dict["july"]]),lin_ubs = np.array([a["metric_ci_ub"] for a in lin_dict["july"]]))
np.savez("bootstrap_august_ts-",pval = np.array(pval_dict["august"]),cnn_mean = np.array([a["avg_metric"] for a in cnn_dict["august"]]),lin_mean = np.array([a["avg_metric"] for a in lin_dict["august"]]), cnn_lbs = np.array([a["metric_ci_lb"] for a in cnn_dict["august"]]),lin_lbs = np.array([a["metric_ci_lb"] for a in lin_dict["august"]]),cnn_ubs = np.array([a["metric_ci_ub"] for a in cnn_dict["august"]]),lin_ubs = np.array([a["metric_ci_ub"] for a in lin_dict["august"]]))








