# Predict Customer Churn

- [Udacity ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) project to convert a notebook into a Python module for **predicting customer churn**. 

## Project Description
This project takes [`churn_notebook.ipynb`](https://github.com/georgeboorman/udacity_ml_devops_engineer/blob/main/churn_notebook.ipynb) and converts it into a series of functions to execute an end-to-end machine learning pipeline.

The aim of the project is to create functions and implement a testing framework, while adhering to PEP8 guidelines and producing production-ready code through the use of logging and testing. 

A test file for all functions in [`churn_libary.py`](https://github.com/georgeboorman/udacity_ml_devops_engineer/blob/main/churn_library.py) has been created called [`churn_script_logging_and_tests.py`](https://github.com/georgeboorman/udacity_ml_devops_engineer/blob/main/churn_script_logging_and_tests.py). Additionally, a file called [`variables.py`](https://github.com/georgeboorman/udacity_ml_devops_engineer/blob/main/variables.py) has been created to store variables used in the functions and tests.

## Files and data description
Overview of the files and data present in the root directory. 

The root directory contains:
* `churn_notebook.ipynb` - A notebook to develop machine learning models for predicting churn from [this dataset](https://github.com/georgeboorman/udacity_ml_devops_engineer/blob/main/data/bank_data.csv).
* `churn_library.py` - A Python file containing functions to perform the end-to-end machine learning pipeline, including EDA, feature engineering, model training, and the storage of models and performance metrics.
* `churn_script_logging_and_tests.py` - A Python file to test the functions in `churn_library.py`, storing tests results in a file called `churn_library.log`.
* `requirements_py3.6.txt` - A list of required packages and versions to execute this project using Python 3.6.
* `requirements_py3.8.txt` - A list of required packages and versions to execute this project using Python 3.8.
* `variables.py` - A Python file containing variables used to store outputs from `churn_library.py` and test results in `churn_script_logging_and_tests.py`.

The `data` directory contains:
* `bank_data.csv` - A credit card customers dataset used for the project, available from Kaggle [here](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code)

The `expected_outputs` directory contains:
* `encoded_values.csv` - A csv file containing the columns and values that should be added to the DataFrame on successful execution of `encoder_helper()` from `churn_library.py`.
* `X_train.csv` - A csv file containing the expected training features following successful execution of `perform_feature_engineering()` from `churn_library.py`.
* `X_test.csv` - A csv file containing the expected training labels following successful execution of `perform_feature_engineering()` from `churn_library.py`.
* `y_train.csv` - A csv file containing the expected test features following successful execution of `perform_feature_engineering()` from `churn_library.py`.
* `y_test.csv` - A csv file containing the expected test labels following successful execution of `perform_feature_engineering()` from `churn_library.py`.

The `models` directory contains:
* `logistic_model.pkl` - A pickle file containing the trained Logistic Regression Classifier.
* `rfc_model.pkl` - A pickle file containing a tuned Random Forest Classifier.

The `logs` directory contains:
* `churn_library.log` - A log file with information about tests run in `churn_script_logging_and_tests.py`.

## Running Files
Clone the repo by entering `git clone https://github.com/georgeboorman/udacity_ml_devops_engineer` in your terminal.

Install the packages by running `python3 pip install -r requirements_py3.6txt` if using Python 3.6, or `python3 pip install -r requirements_py3.8txt` if using Python 3.8 or above.

To execute `churn_library.py` run `python3 churn_library.py`.

    - This will save files in the models, images/eda, and images/results directories.

To execute `churn_script_logging_and_tests.py` run `python3 churn_script_logging_and_tests.py`.
    
    - This will update `churn_library.log`.

To load the models locally run the following code in the terminal or a Python file:
```
rfc_model = joblib.load('./models/rfc_model.pkl')
lr_model = joblib.load('./models/logistic_model.pkl')
```
