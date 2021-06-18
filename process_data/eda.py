# -*- coding: utf-8 -*-
'''
Initial_EDA.ipynb
'''

# imports
import pandas as pd
import numpy as np

# import plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# import logistic regression, SVC, random forests, isolation forests?
# we have target labels so why not see if a model fits on those?
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

"""### EDA Categories"""
def eda(df):

    '''
    function to take a dataframe and perform basic analysis
    '''
    # determine data types and column names
    categories = df.dtypes.to_dict()
    # determine number of samples in each deduction group
    num_values = df['Deduction Code'].value_counts()

    return categories, num_values

def pair_plot(df):
    # create pair plots
    pair_plot = sns.pairplot(df, hue='Deduction Code', height=2.5)

    return pair_plot

def pre_process(df):
    
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
                                                        random_state=7)

    return X_train, X_test, y_train, y_test

"""### Test MLP"""
# only setting up MLP for now
def mlp_test(df):
    # get train and test data
    X_train, X_test, y_train, y_test = pre_process(df)

    # MLP classifier
    mlp_model = mlp(max_iter=1000)

    # fit the data
    mlp_model.fit(X_train, y_train)

    # get predictions
    mlp_preds = mlp_model.predict(X_test)

    # get model performance data
    confusion_matrix(y_test, mlp_preds)

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

    # view the best hyper-parameters
    print('optimal hyperparameters:\n', mlp_clf.best_params_)

    # test model performance using optimized hyper-parameters
    y_true, y_pred = y_test, mlp_clf.predict(X_test) 
    print('Results on the test set:\n', classification_report(y_true, y_pred))

    # get updated predictions
    mlp_preds = mlp_clf.predict(X_test)

    # get updated model performance data
    confusion_matrix(y_test, mlp_preds)

# function to get predictions
def get_preds(df_test, model):
    # collect samples from df_combined using index from X_test and store in df_
    df_ = model.predict(df_test)
    