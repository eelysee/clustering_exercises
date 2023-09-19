import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, explained_variance_score
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns


# creates a residual plot
# plot_residuals(y, yhat): 

# returns the following values
def regression_errors(df, y_train): 
    sns.scatterplot(data=df, x='y_train', y='yhat')
    plt.axhline(0, color='firebrick')
    plt.title("Model Yhat Plot")
    plt.xlabel("Actual")
    plt.ylabel("Yhat")
    
    sns.scatterplot(data=df, x='y_train', y='residual')
    plt.axhline(0, color='firebrick')
    plt.title("Model Residual Plot")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.show()

    
    
def f_y_mean(y_train):
    '''
    calculates the mean of y_train
    TAKES 1 array
    RETURNS 1 value
    '''
    y_mean = y_train.mean()
    return y_mean

#explained sum of squares (ESS)
def f_ess(yhat,y_mean):
    '''
    calculates the exlpained sum of squares
    TAKES in 2 arrays
    '''
    ess = sum((yhat-y_mean)**2)
    return ess 


def f_r2(y_train, yhat):
    '''
    calculates the explained variance score
    TAKES 2 arrays
    RETURNS one value
    '''
    r2 = explained_variance_score(y_train, yhat)
    return r2
    
    
def f_tss():
    '''
    calculates total sum of squares 
    RETURNS one value
    REQUIRES 
            FUNCTION f_ess(), f_sse()
    '''
    tss = f_ess() + f_sse()
    
    return tss


def f_mse(y_train, yhat):
    '''
    calculates mean squared error
    TAKES 2 DataFrames
    RETURNS one value
    REQUIRES from sklearn.metrics import mean_squared_error,
             import pandas as pd
    '''
    mse = mean_squared_error(y_train,yhat)
    
    return mse


def f_sse(y_train, yhat):
    '''
    calculates the sum of squared errors
    RETURNS single value
    '''
    mse = mean_squared_error(y_train, yhat)
    sse = mse * len(y_train)
    return sse


   
#root mean squared error (RMSE)
def f_rmse(y_train, yhat):
    '''
    calculates root of the mean squared error
    TAKES 2 DataFrames 
    RETURNS one number
    REQUIRES import math,
            from sklearn.metrics import mean_squared_error,
            import pandas as pd,
            FUNCTION: f_mse()
    
    '''
    rmse = sqrt(f_mse(y_train,yhat))
    
    return rmse


def mean_or_median_baseline(y_train, y_val):
    '''
    used for determining best baseline for regression test
    TAKES argument y_train(target variable), y_val (validation target variable). 
    - Two arrays or dataframes with one column each.
    determines mean and median, whichever has a lower RSME score on training data
    RETURNS two series: baseline predictions on training and validation data respectively.
    REQUIRES from math import sqrt, 
            import pandas as pd, 
            from sklearn.metrics import mean_squared_error
    '''
    mean = y_train.mean()
    median = y_train.median()
    
    rmse_mean = sqrt(mean_squared_error(y_train, [mean]*len(y_train)))
    rmse_median = sqrt(mean_squared_error(y_train, [median]*len(y_train)))
    
    if rmse_mean < rmse_median:
        print('mean')
        baseline_train = [mean] * len(y_train)
        baseline_val = [mean] * len(y_val)
    else:
        print('median')
        baseline_train = [median] * len(y_train)
        baseline_val = [median] * len(y_val)
        
    return pd.Series(baseline_train, index=y_train.index), pd.Series(baseline_val, index=y_val.index)




def calculate_base_metrics(y_true, yhat):
    y_true = pd.Series(y_true).astype(float)
    yhat = pd.Series(yhat).astype(float) 
    y_mean = y_true.mean()
    ess = sum((yhat - y_mean)**2)
    mse = mean_squared_error(y_true, yhat)
    sse = mse * len(y_true)
    rmse = sqrt(mse)
    r2 = explained_variance_score(y_true, yhat)
    
    results = pd.DataFrame({
        'sse': [sse], 
        'ess': [ess],
        'tss': [sse + ess],
        'mse': [mse], 
        'rmse': [rmse], 
        'r2': [r2]
    })
    
    return results


def calculate_metrics(y_true, yhat):
    y_mean = y_true.mean()
    ess = sum((yhat - y_mean)**2)
    mse = mean_squared_error(y_true, yhat)
    sse = mse * len(y_true)
    rmse = sqrt(mse)
    r2 = explained_variance_score(y_true, yhat)
    
    results = pd.DataFrame({
        'sse': [sse], 
        'ess': [ess],
        'tss': [sse + ess],
        'mse': [mse], 
        'rmse': [rmse], 
        'r2': [r2]
    })
    
    return results


def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    results_df = pd.concat([
        linear_regression_model(X_train, y_train, X_val, y_val),
        lasso_lars_model(X_train, y_train, X_val, y_val),
        polynomial_regression_model(X_train, y_train, X_val, y_val),
        tweedie_regressor_model(X_train, y_train, X_val, y_val),
        random_forest_regressor_model(X_train, y_train, X_val, y_val),
        xgboost_regressor_model(X_train, y_train, X_val, y_val)
    ], axis=0)
    
    return results_df


# Model section moved to evalute to use evalute in funtions

def linear_regression_model(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    
    train_evals = calculate_metrics(y_train, yhat_train)
    val_evals = calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return pd.concat([train_evals, val_evals], axis=0)



def lasso_lars_model(X_train, y_train, X_val, y_val):
    model = LassoLars(alpha=1.0)
    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    
    train_evals = calculate_metrics(y_train, yhat_train)
    val_evals = calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return pd.concat([train_evals, val_evals], axis=0)


def polynomial_regression_model(X_train, y_train, X_val, y_val):
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    yhat_train = model.predict(X_train_poly)
    yhat_val = model.predict(X_val_poly)
    
    train_evals = calculate_metrics(y_train, yhat_train)
    val_evals = calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return pd.concat([train_evals, val_evals], axis=0)


def tweedie_regressor_model(X_train, y_train, X_val, y_val):
    model = TweedieRegressor(power=1, alpha=0.5, link='log')
    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    
    train_evals = calculate_metrics(y_train, yhat_train)
    val_evals = calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return pd.concat([train_evals, val_evals], axis=0)


def random_forest_regressor_model(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    
    train_evals = calculate_metrics(y_train, yhat_train)
    val_evals = calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return pd.concat([train_evals, val_evals], axis=0)


def xgboost_regressor_model(X_train, y_train, X_val, y_val):
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    
    train_evals = calculate_metrics(y_train, yhat_train)
    val_evals = calculate_metrics(y_val, yhat_val)
    
    train_evals['dataset'] = 'train'
    val_evals['dataset'] = 'val'
    
    return pd.concat([train_evals, val_evals], axis=0)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a single model on the test data.
    
    Parameters:
    model: The machine learning model to evaluate.
    X_test: The features in the test dataset.
    y_test: The target variable in the test dataset.

    Returns:
    A dictionary containing various evaluation metrics.
    """

    # Getting the predictions on the test data
    y_pred = model.predict(X_test)

    # Creating a dictionary to store evaluation metrics
    eval_metrics = {}

    # Calculating the mean of y_test to be used in other calculations
    y_mean = y_test.mean()

    # Calculating various evaluation metrics
    eval_metrics['RMSE'] = sqrt(mean_squared_error(y_test, y_pred))
    eval_metrics['R2_Score'] = r2_score(y_test, y_pred)
    eval_metrics['ESS'] = sum((y_pred - y_mean) ** 2)
    eval_metrics['TSS'] = sum((y_test - y_mean) ** 2)
    eval_metrics['MSE'] = mean_squared_error(y_test, y_pred)
    eval_metrics['SSE'] = eval_metrics['MSE'] * len(y_test)
    
    # Adding more metrics as per your requirement

    return eval_metrics

 