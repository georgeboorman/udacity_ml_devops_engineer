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
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

PATH_DF = "data/bank_data.csv"
IMAGE_PATH = "images/eda/"
RESULTS_PATH = "images/results/"
LOGS_PATH = "logs/"
MODELS_PATH = "models/"
category_lst = [
    'Attrition_Flag',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def import_data(PATH_DF):
    '''
    returns DataFrame for the csv found at pth

    input:
            PATH_DF: a path to the csv
    output:
            df: pandas DataFrame
    '''
    df = pd.read_csv(PATH_DF, index_col=["Unnamed: 0"])
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df, IMAGE_PATH):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            IMAGE_PATH: path for storing images

    output:
            None
    '''
    # Churn histogram
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(f'{IMAGE_PATH}churn_histogram.png')

    # Customer age histogram
    df['Customer_Age'].hist()
    plt.savefig(f'{IMAGE_PATH}customer_age_histogram.png')

    # Marital status histogram
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{IMAGE_PATH}marital_status_bar_plot.png')

    # Total Trans Ct histogram and kde plot
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{IMAGE_PATH}/Total_Trans_Ct_kde_plot.png')

    # Correlation heatmap
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{IMAGE_PATH}correlation_heatmap.png')


def encoder_helper(df, category_lst, response="_Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                [optional argument for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Mapping mean churn by category
    for col in category_lst:
        col_mean_churn = df.groupby(col).mean()['Churn']
        df[f'{col}{response}'] = df[col].map(col_mean_churn)
    return df


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name
                [optional argument for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Create features DataFrame
    features = pd.DataFrame()
    targets = df['Churn']
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    features[keep_cols] = df[keep_cols]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test, MODELS_PATH, RESULTS_PATH):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              MODELS_PATH: path to save models as pickle files
              RESULTS_PATH: path to save ROC curve
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
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Train and test set churn predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    # Train and test set churn predictions
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save models
    joblib.dump(cv_rfc.best_estimator_, f'./{MODELS_PATH}rfc_model.pkl')
    joblib.dump(lrc, f'./{MODELS_PATH}logistic_model.pkl')

    # ROC Curve for models
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(f'{RESULTS_PATH}roc_curve.png')

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
             lr_test_report: logistic regression test predictions classificaiton report
             lr_train_report: logistic regression train predictions classificaiton report
             rfc_test_report: random forest test predictions classificaiton report
             rfc_train_report: random forest train predictions classificaiton report
    '''
    # Random forest test and train classification reports
    rf_test_report = classification_report(y_test, y_test_preds_rf)
    rf_train_report = classification_report(y_train, y_train_preds_rf)

    # Plot rfc classification report
    # plt.rc('figure', figsize=(5, 5))
    plt.figure(figsize=(12, 8))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    # Save rfc_classification_plot
    plt.savefig(f'{RESULTS_PATH}rfc_classification_report.png')

    # LRC test and train classification report
    lr_test_report = classification_report(y_test, y_test_preds_lr)
    lr_train_report = classification_report(y_train, y_train_preds_lr)
    plt.figure(figsize=(12, 8))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    # Save lrc_classification_plot
    plt.savefig(f'{RESULTS_PATH}lrc_classification_report.png')

    return rf_test_report, rf_train_report, \
        lr_test_report, lr_train_report


def feature_importance_plot(model, X_data, RESULTS_PATH):
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
    plt.figure(figsize=(20, 15))
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)

    # Save rfc_shap_summary_plot
    plt.savefig(f'{RESULTS_PATH}rfc_shap_summary_plot.png')

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save rfc_feature_importance plot
    plt.savefig(f'{RESULTS_PATH}rfc_feature_importance.png')

if __name__ == "__main__":
    PATH_DF = "data/bank_data.csv"
    IMAGE_PATH = "images/eda/"
    RESULTS_PATH = "images/results/"
    LOGS_PATH = "logs/"
    MODELS_PATH = "models/"
    category_lst = [
        'Attrition_Flag',
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    data_df = import_data(PATH_DF)
    perform_eda(data_df, IMAGE_PATH)
    data_df = encoder_helper(data_df, category_lst, response="_Churn")
    train_features, test_features, train_labels, test_labels = perform_feature_engineering(
        data_df)
    random_forest_cv, log_reg, rf_pred_train_labels, rf_pred_test_labels, \
        log_reg_pred_train_labels, log_reg_pred_test_labels = train_models(train_features,
                                                         test_features,
                                                         train_labels,
                                                         test_labels,
                                                         MODELS_PATH,
                                                         RESULTS_PATH)
    classification_report_image(train_labels,
                                test_labels,
                                log_reg_pred_train_labels,
                                rf_pred_train_labels,
                                log_reg_pred_test_labels,
                                rf_pred_test_labels)
    feature_importance_plot(random_forest_cv, test_features, RESULTS_PATH)
