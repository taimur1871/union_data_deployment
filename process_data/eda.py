# -*- coding: utf-8 -*-
'''
Initial_EDA.ipynb
'''

# imports
import pandas as pd
import numpy as np
import pickle
import os

# import plotting modules
import seaborn as sns

# import logistic regression, SVC, random forests, isolation forests?
# we have target labels so why not see if a model fits on those?
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

"""### EDA Categories"""
def eda(df):
    '''
    function to take a dataframe and get basic stats
    '''
    # determine data types and column names
    categories = df.dtypes.to_dict()
    # determine number of samples in each deduction group
    num_values = df['Deduction Code'].value_counts()

    return categories, num_values

def pair_plot(df):
    # optional function to create pair plots
    pair_plot = sns.pairplot(df, hue='Deduction Code', height=2.5)

    return pair_plot

# function to process training data
def pre_process(df):
    '''
    function to process training dataset
    '''
    # combine code 1 and 2 into one category
    df['category'] = np.where(df['Deduction Code'] == 0, 0,1)

    """### Convert to Dummeis for ML"""

    # get columns and dtypes
    df.drop('Deduction Code',axis=1, inplace=True)

    # create features and labels
    X = df[['Building Description', 'Employee Type Description', 'Zip',
        'Job Start Date', 'Job Pay Rate', 'Yrs Svc']]
    y = df['category']

    # get dummies for X
    X_d = pd.get_dummies(X[['Building Description',
                            'Employee Type Description',
                            ]])

    # get train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_d, y, test_size=0.2, 
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

# function to process prediction data
def process_prediction_df(df):
    '''
    function to process prediction dataset
    '''
    # combine code 1 and 2 into one category
    df['category'] = np.where(df['Deduction Code'] == 0, 0,1)

    '''### Convert to Dummeis for ML'''
    # get columns and dtypes
    df.drop('Deduction Code',axis=1, inplace=True)

    # create features and labels
    X = df[['Building Description', 'Employee Type Description', 'Zip',
        'Job Start Date', 'Job Pay Rate', 'Yrs Svc']]
    y = df['category']

    # get dummies for X
    X_pred = pd.get_dummies(X[['Building Description',
                            'Employee Type Description',
                            ]])

    return X_pred

"""### Test MLP"""
# only setting up MLP for now
def mlp_train(df_train, df_pred, model_save_path):
    # get train and test data
    X_train, X_test, y_train, y_test = pre_process(df_train)

    # get prediction data
    X_pred = process_prediction_df(df_pred)

    # MLP classifier
    mlp_model = mlp(max_iter=1000)

    param_grid = [{
        'activation' : ['relu', 'tanh'],
        'solver' : ['sgd', 'adam'],
        'alpha' : [1e-3, 1e-4],
        'learning_rate' : ['constant', 'adaptive'],
        'random_state' : [7],
        'momentum' : [0.5, 0.7, 0.9], 
    }]

    # instantiate GridSearchCV object
    mlp_clf = GridSearchCV(mlp_model, param_grid, n_jobs=1, cv=5)
    # fit the classifier
    mlp_clf.fit(X_train, y_train)

    # make predictions
    results = mlp_clf.predict(X_pred)

    # combine with df_pred
    df_pred['results'] = results

    # view the best hyper-parameters
    optimal_hparams = mlp_clf.best_params_

    # test model performance using optimized hyper-parameters
    y_true, y_pred = y_test, mlp_clf.predict(X_test)

    # get classification report
    results = classification_report(y_true, y_pred)

    # save the model
    filename = os.path.join(model_save_path, 'saved_model')
    pickle.dump(mlp_model, open(filename, 'wb'))

    return df_pred

# function to get predictions
def get_preds(df_test, saved_model):
    # get prediction data
    X_pred = process_prediction_df(df_test)
    
    # open model
    loaded_model = pickle.load(open(saved_model, 'rb'))
    
    # get results
    results = loaded_model.predict_proba(X_pred)

    # combine with df_pred
    #df_test[['result 0, result 1']] = results[:,0], results[:,1]
    df_test['result 0'] = results[:,0]
    df_test['result 1'] = results[:,1]

    return df_test
    