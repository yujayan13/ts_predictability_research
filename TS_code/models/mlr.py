import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn import metrics
import scipy.stats as measures

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

intercepts = []
coeffs = []
rsquared = []
MSE = []
RMSE = []
averages = []
predictions = []
x_tests = []
y_tests = []

months = ["march","april","may","june","july","august"]

data_dict = {}
for month in months:
    
    month_data = np.load(fr"C:\ts_research\training_{month}_ts.npz")
    data_dict[month] = month_data 

#same seed as cnn, so that shuffled data will be same
np.random.seed(28)

#mlr models for each month
for month in months:
    x = data_dict[month]["enso"]
    x = x.mean(axis = 2)
    x = x.mean(axis = 2)
    
    Y = data_dict[month]["ts"]

    
    month_intercepts = []
    month_coeffs = []
    month_rsquared = []
    month_MSE = []
    month_RMSE = []
    month_averages = []
    month_predictions = []
    month_x_tests = []
    month_y_tests = []
    
    
    x,Y = unison_shuffled_copies(x, Y)
    #fit mlr model for each gridpoint in us
    for point in range(691):
        
        #selecting TS output for gridpoint
        y = [j[point] for j in Y]
        averages.append(np.mean(y))
        
        #split training and testing data, discarding 15% for consistency as MLR models do not have validation data
        DATASET_SIZE = len(x)
        TRAIN_SIZE=int(7*DATASET_SIZE/10)
        TEST_SIZE = int(15*DATASET_SIZE/100)
        
        x_train = x[:TRAIN_SIZE]
        y_train = y[:TRAIN_SIZE]
        x_test = x[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
        y_test = y[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
        
        #create mlr model
        mlr = LinearRegression()  
        mlr.fit(x_train, y_train)
        
        y_pred_mlr= mlr.predict(x_test)
        month_intercepts.append(mlr.intercept_)
        
        #skill scores
        per_coef = measures.pearsonr(y_pred_mlr, y_test)[0]
        month_coeffs.append(per_coef)
        meanAbErr = metrics.mean_absolute_error(y_pred_mlr, y_test)
        meanSqErr = metrics.mean_squared_error(y_pred_mlr, y_test)
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_pred_mlr, y_test))
        
        month_rsquared.append(mlr.score(x,y))
        month_MSE.append(meanSqErr)
        month_RMSE.append(rootMeanSqErr)
        month_predictions.append(y_pred_mlr)
        month_y_tests.append(y_test)
        month_x_tests.append(x_test)
        print(point)

    MSE.append(month_MSE)
    RMSE.append(month_RMSE)
    x_tests.append(month_x_tests)
    predictions.append(month_predictions)
    coeffs.append(month_coeffs)
    intercepts.append(month_intercepts)
    rsquared.append(month_rsquared)
    y_tests.append(month_y_tests)

np.savez("results_lin_march_ts.npz", intercepts = np.array(intercepts[0]), coeffs = np.array(coeffs[0]), rsquared = np.array(rsquared[0]),MSE = np.array(MSE[0]),RMSE = np.array(RMSE[0]),averages = np.array(averages[0]), pred = np.array(predictions[0]), x = np.array(x_tests[0]), y = np.array(y_tests[0]))
np.savez("results_lin_april_ts.npz", intercepts = np.array(intercepts[1]), coeffs = np.array(coeffs[1]), rsquared = np.array(rsquared[1]),MSE = np.array(MSE[1]),RMSE = np.array(RMSE[1]),averages = np.array(averages[1]), pred = np.array(predictions[1]), x = np.array(x_tests[1]),y = np.array(y_tests[1]))
np.savez("results_lin_may_ts.npz", intercepts = np.array(intercepts[2]), coeffs = np.array(coeffs[2]), rsquared = np.array(rsquared[2]),MSE = np.array(MSE[2]),RMSE = np.array(RMSE[2]),averages = np.array(averages[2]), pred = np.array(predictions[2]), x = np.array(x_tests[2]),y = np.array(y_tests[2]))
np.savez("results_lin_june_ts.npz", intercepts = np.array(intercepts[3]), coeffs = np.array(coeffs[3]), rsquared = np.array(rsquared[3]),MSE = np.array(MSE[3]),RMSE = np.array(RMSE[3]),averages = np.array(averages[3]), pred = np.array(predictions[3]), x = np.array(x_tests[3]),y = np.array(y_tests[3]))
np.savez("results_lin_july_ts.npz", intercepts = np.array(intercepts[4]), coeffs = np.array(coeffs[4]), rsquared = np.array(rsquared[4]),MSE = np.array(MSE[4]),RMSE = np.array(RMSE[4]),averages = np.array(averages[4]), pred = np.array(predictions[4]), x = np.array(x_tests[4]),y = np.array(y_tests[4]))
np.savez("results_lin_august_ts.npz", intercepts = np.array(intercepts[5]), coeffs = np.array(coeffs[5]), rsquared = np.array(rsquared[5]),MSE = np.array(MSE[5]),RMSE = np.array(RMSE[5]),averages = np.array(averages[5]), pred = np.array(predictions[5]), x = np.array(x_tests[5]),y = np.array(y_tests[5]))
