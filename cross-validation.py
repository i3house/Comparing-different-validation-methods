#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 

import itertools
import time
import warnings
warnings.filterwarnings('ignore')
 
data = pd.read_csv('community.csv') #Dataset has 2118 observations and 102 columns

predictors = data.copy()
predictors.drop('nonViolPerPop', axis = 1, inplace = True) # There are 101 predictors
response = data[['nonViolPerPop']] # And 1 response variable

#%%
# 1 - LASSO Regression WITHOUT K-Fold Cross-Validation

# Split the data into (Training + Validation) set and Testing set (or the HOLDOUT set)
X_train_valid, X_test, y_train_valid, y_test = train_test_split(predictors, response, test_size = 0.3, random_state = 8440)

# Split the (Training + Validation) set into Training set and Validation set. Allocate 5/14 to validation and 9/14 to training
# such that the overall ratio of train, validate, test is 45%, 25%, and 30% respectively. 
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = 5/14, random_state = 8440)

# Defining the set of candidate hyperparameters
alphas = np.logspace(-5, 5, 11) # Lambda values on powers of 10 scale
max_iters = np.array([50, 100, 250, 500, 750, 1000, 2000, 5000, 7500, 10000]) # Max and min number of iterations
tols = np.array([0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

# Forming all unique combination triplets of Lambdas, Max iterations, and tolerances. The total number of triplets in our case are 1100 (11x10x10)
hyperparameter_triplets = list(itertools.product(alphas, max_iters, tols))
print('Number of hyperparameter triplets:', len(hyperparameter_triplets))

# Pre-processing the data
scaler = StandardScaler() # Instantiate
scaler.fit(X_train) # Fit the data (learn mean and sd)
X_train = pd.DataFrame(scaler.transform(X_train)) # Transform training data
X_valid = pd.DataFrame(scaler.transform(X_valid)) # Transform validation data
X_train.columns = predictors.columns.values
X_valid.columns = predictors.columns.values

# Perform Lasso Regression on all hyperparameter triplets. Fitting the training data and finding the MSE on validation data. Also tracking the time to find the best triplet.

validation_MSE_scores = []

start_time = time.perf_counter()
for triplet in hyperparameter_triplets:
    lm_lasso = linear_model.Lasso(alpha = triplet[0], max_iter = triplet[1], tol = triplet[2])
    lm_lasso.fit(X_train, y_train) # Fit model on training data
    validation_MSE_scores.append(metrics.mean_squared_error(lm_lasso.predict(X_valid), y_valid)) # Evaluate model on validation set
end_time = time.perf_counter()

# Print time taken
print('Time taken (seconds) to tune the 3 parameters (1100 triplets) for Lasso Regression w/o K-Fold CV:', end_time - start_time)

# Print minimum validation MSE and the best parameter triplet (Lambda, Max Iteration, Tolerance) which gives that minimum MSE
print('Lasso Regression w/o K-Fold CV: Minimum MSE of validation set =', min(validation_MSE_scores))
best_triplet = hyperparameter_triplets[np.argmin(validation_MSE_scores)]
print('Lasso Regression w/o K-Fold CV: Parameter Triplet (Lambda, Max Iteration, Tolerance) for minimum MSE of validation set =', best_triplet)

# Scaling data again using the (Training + Validation) set to fit, then transform both: The (Training + Validation set) and the Testing set
scaler = StandardScaler()
scaler.fit(X_train_valid)
X_train_valid = pd.DataFrame(scaler.transform(X_train_valid)) # Transform the (Training + Validation) set
X_test = pd.DataFrame(scaler.transform(X_test)) # Transform the Testing set
X_train_valid.columns = predictors.columns.values
X_test.columns = predictors.columns.values

# Refit model with (Training + Validation) set and the best parameters triplet
lm_lasso = linear_model.Lasso(alpha = best_triplet[0], max_iter = best_triplet[1], tol = best_triplet[2])
lm_lasso.fit(X_train_valid, y_train_valid)

# MSE ON TEST SET
print("Lasso Regression w/o K-Fold CV: MSE on the test set =", metrics.mean_squared_error(lm_lasso.predict(X_test), y_test))

# Print coefficients of Lasso.
print('Lasso Regression w/o K-Fold CV: Coefficients ')
method_1_lasso_coefficients = pd.DataFrame(zip(lm_lasso.coef_,X_train_valid.columns))
method_1_lasso_coefficients.columns = np.array(['Coefficient', 'Predictor'])
print(method_1_lasso_coefficients)

# Print number of coefficients shrunk to Zero
print('Lasso Regression w/o K-Fold CV: Number of Coefficients that are shrunk to Zero:', (len(lm_lasso.coef_) - np.count_nonzero(lm_lasso.coef_)))


#%%

#2 and #3 - Defined a custom function perform_kfold_CV that performs LASSO Regression WITH K-Fold Cross-Validation.
# K is the input parameter of this function. Its called two times later in the code with values 5 and 10
# (Tuning Parameters already defined in the code prior to this and hence, they can act kind of like global variables)

def perform_kfold_CV (number_of_folds):
    # Split the data into (Training + Validation) set and Testing set (or the HOLDOUT set)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(predictors, response, test_size = 0.3, random_state = 8440)
    
    validation_MSE_scores = []
    
    start_time = time.perf_counter()
    for ind, triplet in enumerate(hyperparameter_triplets): # Loop over each candidate hyperparameter triplet
        kfold = KFold(n_splits = number_of_folds, shuffle = True, random_state = 8440) # Instantiating the fold
        tmp_mse = [] # To store the current hyperparameter triplet's validation MSE for the current fold's data
        for train_index, valid_index in kfold.split(X_train_valid): # Splitting into Training and Validation set
            X_train, y_train = X_train_valid.iloc[train_index], y_train_valid.iloc[train_index] # Training set
            X_valid, y_valid = X_train_valid.iloc[valid_index], y_train_valid.iloc[valid_index] # Validation set
            
            #Scale the data
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train))
            X_valid = pd.DataFrame(scaler.transform(X_valid))
            X_train.columns = predictors.columns.values
            X_valid.columns = predictors.columns.values
                
            #Perform Lasso fitting
            lm_lasso = linear_model.Lasso(alpha = triplet[0], max_iter = triplet[1], tol = triplet[2])
            lm_lasso.fit(X_train, y_train)
            tmp_mse.append(metrics.mean_squared_error(lm_lasso.predict(X_valid), y_valid))
    
        validation_MSE_scores.append(np.mean(tmp_mse)) # Calculate the average MSE across the K folds
    end_time = time.perf_counter()
    
    # Time taken
    print('Time taken (seconds) to tune the 3 parameters (1100 triplets) for Lasso Regression with {}-Fold CV: {}'.format(number_of_folds, end_time - start_time))
    
    # Print minimum validation MSE and the best parameter triplet (Lambda, Max Iteration, Tolerance) which gives that minimum MSE
    print('Lasso Regression with {}-Fold CV: Minimum MSE of validation set = {}'.format(number_of_folds, min(validation_MSE_scores)))
    best_triplet = hyperparameter_triplets[np.argmin(validation_MSE_scores)]
    print('Lasso Regression with {}-Fold CV: Parameter Triplet (Lambda, Max Iteration, Tolerance) for minimum MSE of validation set = {}'.format(number_of_folds, best_triplet))
    
    # Refit the model on the whole training set, which the selected lambda
    scaler = StandardScaler()
    scaler.fit(X_train_valid)
    X_train_valid = pd.DataFrame(scaler.transform(X_train_valid))
    X_test = pd.DataFrame(scaler.transform(X_test))
    X_train_valid.columns = predictors.columns.values
    X_test.columns = predictors.columns.values
    
    # Refit model with (Training + Validation) set and the best parameters triplet
    lm_lasso = linear_model.Lasso(alpha = best_triplet[0], max_iter = best_triplet[1], tol = best_triplet[2])
    lm_lasso.fit(X_train_valid, y_train_valid)
    
    # MSE ON TEST SET
    print('Lasso Regression with {}-Fold CV: MSE on the test set = {}'.format(number_of_folds, metrics.mean_squared_error(lm_lasso.predict(X_test), y_test)))
    
    # Print coefficients of Lasso.
    print('Lasso Regression with {}-Fold CV: Coefficients'.format(number_of_folds))
    if (number_of_folds == 5):
        global method_2_lasso_coefficients # We need this variable outside the function, hence creating a global variable
        method_2_lasso_coefficients = pd.DataFrame(zip(lm_lasso.coef_,X_train_valid.columns))
        method_2_lasso_coefficients.columns = np.array(['Coefficient', 'Predictor'])
        print(method_2_lasso_coefficients)
   
    elif (number_of_folds == 10):
        global method_3_lasso_coefficients # We need this variable outside the function, hence creating a global variable
        method_3_lasso_coefficients = pd.DataFrame(zip(lm_lasso.coef_,X_train_valid.columns))
        method_3_lasso_coefficients.columns = np.array(['Coefficient', 'Predictor'])
        print(method_3_lasso_coefficients)

    # Print number of coefficients shrunk to 0
    print('Lasso Regression with {}-Fold CV: Number of Coefficients that are shrunk to Zero: {}'.format(number_of_folds, (len(lm_lasso.coef_) - np.count_nonzero(lm_lasso.coef_))))

perform_kfold_CV (5) # Performing Lasso with 5-Fold Cross Validation 
perform_kfold_CV (10) # Performing Lasso with 10-Fold Cross Validation

# Code to find Predictors with a value of 0 in all the 3 Lasso methods
result = method_1_lasso_coefficients[(method_1_lasso_coefficients['Coefficient'] == 0) & (method_2_lasso_coefficients['Coefficient'] == 0) & (method_3_lasso_coefficients['Coefficient'] == 0)]['Predictor']
print('Number of predictors with a value of zero in all 3 Lasso Methods:', result.size)
print('Predictors with a value of zero in all 3 Lasso Methods:', list(result))

# %%
#
# COMPARISON OF THE 3 METHODS
# Note: The results below are only for the random state of 8440
#
#                                                    Train-Valid-Split      5-Fold CV      10-Fold CV
#                           Train-Valid-Split Ratio      45-25-30           56-14-30         63-7-30
#-----------------------------------------------------------------------------------------------------
# Time taken for hyperparameter selection (seconds)        114.8              779.4         1843.6                      
# Time taken for hyperparameter selection (minutes)         1.9               12.99          30.7
#                                Validation set MSE      4486127.8          3538782.4      3528864.6
#                                       Best Lambda          10                10             10
#                           Best Maximum Iterations         250                50             50
#                                    Best Tolerance        0.005              0.1            0.1
#                                      Test set MSE       3498974           3502874.9      3502874.9
#    Number of coefficients shrunk to Zero by Lasso          31                34             34
#
# OBSERVATIONS / SUMMARY OF RESULTS
# 1) Time: As expected, the first method takes the least amount of time since there was only 1 validation set it it. 10-Fold CV takes the most amount of time.
#          This makes sense as the more number of folds, the more time taken by the model.
# 2) Test Set MSE: 5-Fold CV and 10-Fold CV MSE is equal (because their optimal hyperparameters are equal as well).
#                  This is interesting because 10-Fold CV took almost 2.5x time compared to 5-Fold CV and was not more accurate
#                  Their test set MSE is also slightly higher than that of the first method.
# 3) Coefficient Shrinkage to Zero by Lasso: Method 1 shrinks 31 coefficients to 0, while 5-Fold CV and 10-Fold CV shrink 34.
#                                            Hence the latter 2 methods would give the least complicated Regression Model.
# 4) The following 26 predictor variables have a value of 0 in all the 3 Lasso methods: 'population', 'racePctWhite', 'numbUrban', 'medIncome', 'pctWWage', 
#    These are pretty insignificant                                                     'pctWFarmSelf', 'perCapInc', 'PctOccupManu', 'TotalPctDiv', 'PersPerFam', 
#                                                                                       'PctYoungKids2Par', 'PctWorkMom', 'PctKidsBornNeverMar', 'PctImmigRec8',
#                                                                                       'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctNotSpeakEnglWell',
#                                                                                       'PctPersDenseHous', 'OwnOccHiQuart', 'OwnOccQrange', 'MedRent',
#                                                                                       'MedOwnCostPctIncNoMtg', 'NumInShelters', 'PctSameHouse85', 'PctSameCity85'
#
# MY CHOICE: Amongst the 3 methods it probably makes sense to choose method 1, considering it takes the least amount of time, gives better MSE on the test set, and shrinks 31 coefficients to zero.
#
# PROS AND CONS OF EACH METHOD
#
#                                PROS                                                CONS
# ------------------------------------------------------------------------------------------------------------------------
# Train-Valid-Split             Simple                               May overfit if dataset is very small
#                     Fastest (if time is major factor)              May not provide reliable estimate of model performance
#
# 5-Fold CV           Suitable for most datasets (small to medium)   Requires more time/computation
#                     More reliable estimate of model performance
#
# 10-Fold CV          Suitable for large datasets                    Requires even more time/computation than 5-Fold.
#%%