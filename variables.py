# Import pandas
import pandas as pd

# Variables used in churn_library or to test functions
PATH_DF = "data/bank_data.csv"
IMAGE_PATH = "images/eda/"
RESULTS_PATH = "images/results/"
LOGS_PATH = "logs/"
MODELS_PATH = "models/"
EXPECTED_OUTPUTS_PATH = "expected_outputs/"
CATEGORY_LST = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

# Expected values from encoder_helper()
encoded_values = pd.read_csv(f'{EXPECTED_OUTPUTS_PATH}encoded_values.csv')

# Expected values from perform_feature_engineering()
X_train_expected = pd.read_csv(f'{EXPECTED_OUTPUTS_PATH}X_train.csv')
X_test_expected = pd.read_csv(f'{EXPECTED_OUTPUTS_PATH}X_test.csv')
y_train_expected = pd.read_csv(f'{EXPECTED_OUTPUTS_PATH}y_train.csv')
y_test_expected = pd.read_csv(f'{EXPECTED_OUTPUTS_PATH}y_test.csv')
