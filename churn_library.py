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
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

path_df = "data/bank_data.csv"
image_path = "images/eda/"
results_path = "images/results/"
logs_path = "logs/"
models_path = "models/"
category_lst = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

def import_data(pth):
    '''
    returns DataFrame for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas DataFrame
    '''	
    df = pd.read_csv(path_df, index_col=["Unnamed: 0"])
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df

def perform_eda(df, image_path):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20,10))
    df['Churn'].hist()
    plt.savefig(f'{image_path}churn_histogram.png')
    df['Customer_Age'].hist()
    plt.savefig(f'{image_path}customer_age_histogram.png')
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{image_path}marital_status_bar_plot.png')
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{image_path}/Total_Trans_Ct_kde_plot.png')
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{image_path}Total_Trans_Ct_kde_plot.png')
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(f'{image_path}correlation_heatmap.png')

def encoder_helper(df, category_lst, response="_Churn"):
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
    # category_lst = df.select_dtypes("object").columns
    for col in category_lst:
        col_mean_churn = df.groupby(col).mean()['Churn']
        df[f'{col}{response}'] = df[col].map(col_mean_churn)
    return df


#     try:
#         cat_columns = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
#         logging.info("Categories: {}".format(cat_columns)
#         quant_columns = df.columns.symmetric_difference(cat_cols)
#         logging.info("Quant columns: {}".format(quant_columns))
#         return "Categories: {}".format(cat_columns), \
#                 "\n", "Quant columns: {}".format(quant_columns)

def perform_feature_engineering(df, response=None):
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

    # Create features DataFrame
    X = pd.DataFrame()
    y = df['Churn']
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
        'Income_Category_Churn', 'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test, models_path, results_path):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              cv_rfc: Cross-validated tuned Random Forest Classifier (RFC)
              lrc: Logistic Regression Classifier (LRC)
              y_train_preds_rf: RFC's predicted churn values from training data
              y_test_preds_rf: RFC's predicted churn values from test data
              y_train_preds_lr: LRC's predicted churn values from training data
              y_test_preds_lr: LRC's predicted churn values from test data
    '''
    # Random forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
        }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save models
    joblib.dump(cv_rfc.best_estimator_, f'./{models_path}rfc_model.pkl')
    joblib.dump(lrc, f'./{models_path}logistic_model.pkl')

    # ROC Curve for models
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(f'{results_path}roc_curve.png')

    return cv_rfc, lrc, y_train_preds_rf, y_test_preds_rf, \
        y_train_preds_lr, y_test_preds_lr

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
    # Random forest test and train classification reports
    rf_test_report = classification_report(y_test, y_test_preds_rf)
    rf_train_report = classification_report(y_train, y_train_preds_rf)

    # Plot rfc classification report
    # plt.rc('figure', figsize=(5, 5))
    plt.figure(figsize=(12,8))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')

    # Save rfc_classification_plot
    plt.savefig(f'{results_path}rfc_classification_report.png')

    # LRC test and train classification report
    lr_test_report = classification_report(y_test, y_test_preds_lr)
    lr_train_report = classification_report(y_train, y_train_preds_lr)
    plt.figure(figsize=(12,8))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')

    # Save lrc_classification_plot
    plt.savefig(f'{results_path}lrc_classification_report.png')


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
    # Feature contribution by churn class
    plt.figure(figsize=(20,15))
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)

    # Save rfc_shap_summary_plot
    plt.savefig(f'{results_path}rfc_shap_summary_plot.png')

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save rfc_feature_importance plot
    plt.savefig(f'{results_path}rfc_feature_importance.png')

if __name__ == "__main__":
    path_df = "data/bank_data.csv"
    image_path = "images/eda/"
    results_path = "images/results/"
    logs_path = "logs/"
    models_path = "models/"
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]
    df = import_data(path_df)
    perform_eda(df, image_path)
    df = encoder_helper(df, category_lst, response="_Churn")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    cv_rfc, lrc, y_train_preds_rf, y_test_preds_rf, \
        y_train_preds_lr, y_test_preds_lr = train_models(X_train, X_test,
         y_train, y_test, models_path, results_path)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    feature_importance_plot(cv_rfc, X_test, results_path)