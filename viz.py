import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind


def get_continuous_cat_feats(df) -> (list, list):
    '''
    find all continuous and catagorical numerical features
    
    returns: 2 lists of column names
    
    changed continuous defition to > 10
    '''
    cont = []
    cat = []
    
    num_df = df.select_dtypes('number')
    for col in num_df:
        if num_df[col].nunique() > 10:
            cont.append(col)
        else:
            cat.append(col)
    return cont , cat


def plot_categorical_and_continuous_vars(df, cat, cont):
    '''
    full train df not X_train
    plots catagorical and continuous to be used in explore
    TAKES 1 DF, 2 lists
    '''
    for i in cat:
        for l in cont:
            sns.barplot(data=df, x=i , y=l)
            plt.title(f'{l.capitalize()} by {i.capitalize()}')
            plt.xlabel(i.capitalize())
            plt.ylabel(l.capitalize())
            plt.show()
            
    for i in cont:
        for l in cont:
            if i != l:
                sns.scatterplot(data=df, x=i , y=l)
                plt.title(f'{l.capitalize()} by {i.capitalize()}')
                plt.xlabel(i.capitalize())
                plt.ylabel(l.capitalize())
                plt.show()
 

def cont_plots(df, cont):
    '''
    scatter plot for all continous
    '''
    for i in cont:
        for l in cont:
            if i != l:
                sns.scatterplot(data=df, x=i , y=l)
                plt.title(f'{l.capitalize()} by {i.capitalize()}')
                plt.xlabel(i.capitalize())
                plt.ylabel(l.capitalize())
                plt.show()
 
                                    
def count_plots(df, cat):
    '''
    full train df not X_train
    Takes 1 df
    Takes 1 list
    returns count plot of each catagorical feature
    '''
    for i in cat:
        sns.countplot(data=df, x=i)
        plt.title('Distribution of '+ i.capitalize())
        plt.xlabel(i.capitalize())
        plt.ylabel('Count')
        plt.show()
    
    
def pearsonr_viz(X_train,y_train):
    '''
    runs and plots pearsonr for continous contionus
    '''
    print(f'H_0 sqft has no effect on value')
    print(f'H_a sqft has an effect on value')
    
    sqft = X_train.sqft
    corr_coef, p_value = pearsonr(X_train.sqft, y_train)
    
    sns.scatterplot(x=X_train.sqft, y=y_train)
    plt.title(f'Sqft and Value (correlation coefficient: {corr_coef:.4f})')
    plt.xlabel('Sqft')
    plt.ylabel('Value')
    plt.show()
        
    if p_value < 0.05:
        print(f"Reject the null hypothesis: sqft has an effect on value.")
        print()
        print()
    else:
        print(f"Fail to reject the null hypothesis: No significant evidence to suggest that sqft affects value.")
        print()
        print()

            
def ttest_viz(X_train, y_train, cat):
    '''
    run and plots ttest for categorical features against a continuous target
    '''    
    for i in cat:
        print(f'H_0 {i} has no effect on value')
        print(f'H_a {i} has an effect on value')

        unique_categories = X_train[i].unique()

        group1 = y_train[X_train[i] == unique_categories[0]]
        group2 = y_train[X_train[i] == unique_categories[1]]

        t_stat, p_value = ttest_ind(group1, group2)
        
        # box plot
        sns.barplot(x=i, y=y_train, data=X_train)
        plt.title(f'{i.capitalize()} and Value')
        plt.xlabel(i.capitalize())
        plt.ylabel('Value')
        plt.show()
        
        if p_value < 0.05:
            print(f'Reject the null hypothesis: {i} has an effect on value. (p-value: {p_value:.4e})')
            print()
            print()
        else:
            print(f"Fail to reject the null hypothesis: No significant evidence to suggest that {i} affects value. (p-value: {p_value:.4e})")
            print()
            print()
          
       