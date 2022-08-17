# library doc string
'''
Module for testing machine learning pipeline functions from churn_library:
1) test_import_data: tests import_data()
2) test_perform_eda: tests perform_eda()
3) test_encoder_helper: tests encoder_helper()
4) test_perform_feature_engineering: tests perform_feature_engineering()
5) test_train_models: test train_models()

Author: George Boorman
Date: August 2022
'''

import os
import logging
from tempfile import TemporaryFile
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
from variables import *

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_df = import_data(PATH_DF)
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
        logging.info("SUCCESS: DataFrame is not empty")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    data_df = import_data(PATH_DF)

    files = ["churn_histogram.png", "correlation_heatmap.png",
             "customer_age_histogram.png", "marital_status_bar_plot.png",
             "Total_Trans_Ct_kde_plot.png"]
    actual_files = set(os.listdir(IMAGE_PATH))
    try:
        perform_eda(data_df, IMAGE_PATH)
        assert actual_files.intersection(set(files))
        logging.info(f'SUCCESS: Files were successfully saved in {IMAGE_PATH}')
    except AssertionError:
        logging.error(f'Not all files were successfully saved in {IMAGE_PATH}')
    try:
        assert [os.stat(str(IMAGE_PATH + file)).st_size > 0 for file in files]
        logging.info("SUCCESS: The images are not empty")
    except FileNotFoundError:
        logging.error("One or more of the files are empty")

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    # Perform prerequisite functions
    data_df = import_data(PATH_DF)

    # List of columns to check exist after function has been executed
    new_cols = set(['Attrition_Flag_Churn',
                    'Gender_Churn',
                    'Education_Level_Churn',
                    'Marital_Status_Churn',
                    'Income_Category_Churn',
                    'Card_Category_Churn'])

    try:
        data_df = encoder_helper(data_df, CATEGORY_LST)
        expected_columns = set(data_df.columns.tolist())
        assert [col for col in new_cols if col in expected_columns]
        logging.info("SUCCESS: New columns successfully created")
    except NameError:
        logging.error(
            "One or more of the encoded columns are missing from the DataFrame")
    try:
        # Check if values match expected values
        assert set(data_df[new_cols]).intersection(encoded_values)
        logging.info(
            "SUCCESS: Categorical values have been encoded for churn proportion")
    except AssertionError:
        logging.error("One of more of the columns contain incorrect values")

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    # Perform prerequisite functions
    data_df = import_data(PATH_DF)
    data_df = encoder_helper(data_df, CATEGORY_LST)

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(data_df)
        assert [
            expected.equals(actual) for expected in [
                X_train_expected,
                X_test_expected,
                y_train_expected,
                y_test_expected] for actual in [
                X_train,
                X_test,
                y_train,
                y_test]]
        logging.info(
            "SUCCESS: Data has been split into training and test sets")
    except AssertionError:
        logging.error("The training and test set values aren't as expected")

def test_train_models(train_models):
    '''
    test train_models
    '''
    data_df = import_data(PATH_DF)
    data_df = encoder_helper(data_df, CATEGORY_LST)
    X_train, X_test, y_train, y_test = perform_feature_engineering(data_df)

    models = ["logistic_model.pkl", "rfc_model.pkl"]
    actual_models = set(os.listdir(MODELS_PATH))
    try:
        cv_rfc, lrc, y_train_preds_rf, y_test_preds_rf, \
            y_train_preds_lr, y_test_preds_lr = train_models(X_train, X_test,
                                   y_train, y_test, MODELS_PATH, RESULTS_PATH)
        # Check models are saved and match file names
        assert actual_models.issuperset(set(models))
        logging.info(
            f'SUCCESS: Models have been trained and saved in {MODELS_PATH}')
    except AssertionError:
        logging.error(
            f'The models have not been successfully stored in {MODELS_PATH}')
    try:
        assert os.path.exists(str(RESULTS_PATH + 'roc_curve.png'))
        logging.info(f'SUCCESS: ROC Curve plot saved in {RESULTS_PATH}')
    except AssertionError:
        logging.error("ROC Curve plot has not been saved correctly")

if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
