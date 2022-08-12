# library doc string
'''
Module for performing an end-to-end machine learning pipeline, with functions to:
1) Import data
2) Perform exploratory data analysis (descriptive statistics and visualizations) and store plots
3) Convert categorical columns into numeric data with proportion of churn per category as values
4) Split data into training and test sets
5) Produce classification reports and store as images
6) Plot feature importance
7) Train models, storing parameters and performance metrics

Author: George Boorman
Date: August 2022
'''

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import logging
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

path_df = "data/bank_data.csv"
image_path = "images/eda"
results_path = "images/results"
logs_path = "logs"
models_path = "models"

def import_data(pth):
    '''
    returns DataFrame for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas DataFrame
    '''	
    import pandas as pd
    df = pd.read_csv(path_df, index_col=["Unnamed: 0"])
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df.head(), "/n", df.shape, \
        "\n", df.isnull().sum(), "\n", \
                df.describe()


def perform_eda(df, results_path):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20,10))
    df['Churn'].hist()
    plt.savefig(f'{results_path}/churn_histogram.png')
    df['Customer_Age'].hist()
    plt.savefig(f'{results_path}/customer_age_histogram.png')
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{results_path}/marital_status_bar_plot.png')
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{results_path}/Total_Trans_Ct_kde_plot.png')
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{results_path}/Total_Trans_Ct_kde_plot.png')
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.show()
    plt.savefig(f'{results_path}/correlation_heatmap.png')

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        cat_columns = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
        logging.info("Categories: {}".format(cat_columns)
        quant_columns = df.columns.symmetric_difference(cat_cols)
        logging.info("Quant columns: {}".format(quant_columns))
        return "Categories: {}".format(cat_columns), \
                "\n", "Quant columns: {}".format(quant_columns)

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass