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
import os
import logging
os.environ['QT_QPA_PLATFORM']='offscreen'

# Set up logging for all functions
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%{name}s - %{levelname}s - %{message}s')

def import_data(pth):
    '''
    returns DataFrame for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas DataFrame
    '''	
    import pandas as pd
	try:
        # Read in data without extra column
        df = pd.read_csv(pth, index_col=["Unnamed: 0"])
        assert df.isinstance(pd.DataFrame)
        logging.info("Successfully read in file as pandas DataFrame")
    except ValueError:
        logging.error("The function requires a path to a csv file")


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
	pass


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
    pass


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

if __name__ == "__main__":
    import_data("/data/bank_data.csv")
    import_data([1,2,3])
    import_data("README.md")