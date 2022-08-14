# Import functions
import pandas as pd
import numpy as np

# Variables used in churn_library or to test functions
path_df = "data/bank_data.csv"
image_path = "images/eda/"
results_path = "images/results/"
logs_path = "logs/"
models_path = "models/"
expected_outputs_path = "expected_outputs/"
category_lst = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

# Expected values from encoder_helper()
encoded_values = pd.read_csv(f'{expected_outputs_path}encoded_values.csv')

# Expected values from perform_feature_engineering()
X_train_expected = pd.read_csv(f'{expected_outputs_path}X_train.csv')
X_test_expected = pd.read_csv(f'{expected_outputs_path}X_test.csv')
y_train_expected = pd.read_csv(f'{expected_outputs_path}y_train.csv')
y_test_expected = pd.read_csv(f'{expected_outputs_path}y_test.csv')