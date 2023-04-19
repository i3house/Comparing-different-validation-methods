# Comparing-different-validation-methods
Create Lasso Regression Model and compare 3 cross-validation methods

The objective of this project is to implement and compare different validation methods, and compare their performances.

THe dataset we will be using is a cleaned up version of [Communities and Crime](http://archive.ics.uci.edu/ml/datasets/communities+and+crime+unnormalized) dataset from the UCI repository. Attached the cleaned version (removed missing values, removed unnecessary other response variables) under files. 

It is a crime rate dataset from different communities. There are 101 predictor variables such as household size, number of police officers, population, etc. The response variable is the total number of non-violent crimes per 100k population.

The cleaned dataset consists of 2118 observations, 1 response variable, and 101 predictors (medium-sized dataset)

Since there are a high number of predictors, I use Lasso regression model to predict the response variable. I use 3 types of validation methods:-

1) A simple train-validate-test (can be thought of as '1-fold cross validation')
2) 5-fold Cross-validation
3) 10-fold Cross-validation

to find the best hyper-parameters (Lambda, Max Iteration, and Tolerance) and compare their performances. I implement 5-Fold and 10-Fold using code, rather than GridSearchCV.

# Observations at the end
1) Time: As expected, Method 1 takes the least amount of time since there was only 1 validation set it it. Method 3 takes the most amount of time. This makes sense as the more number of folds, the more time taken by the model.
2) Test Set MSE: Method 2 and Method 3 MSE is equal (because their optimal hyperparameters are equal as well). This is interesting because Method 3 took almost 2.5x time compared to Method 2 and was not more accurate. Their test set MSE is also slightly higher than that of the first method.
3) Coefficient Shrinkage to Zero by Lasso: Method 1 shrinks 31 coefficients to 0, while Method 2 and Method 3 shrunk 34. The latter 2 methods thus give the least complicated Regression Model.
4) There were 26 predictors that shrunk to 0 in all the 3 Lasso methods: These are pretty insignificant.

Amongst the 3 methods it probably makes sense to choose Method 1, considering it takes the least amount of time, gives better MSE on the test set, and shrinks 31 coefficients to zero.

# Pros and Cons of each

                                PROS                                                CONS
------------------------------------------------------------------------------------------------------------------------
Train-Valid-Split             Simple                               May overfit if dataset is very small
                     Fastest (if time is major factor)             May not provide reliable estimate of model performance

5-Fold CV           Suitable for most datasets (small to medium)   Requires more time/computation
                    More reliable estimate of model performance

10-Fold CV          Suitable for large datasets                    Requires even more time/computation than 5-Fold.
