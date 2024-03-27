# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20

@author: Rosa
"""
# import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# pylint: disable=no-value-for-parameter

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

     output:
            eda_df: pandas DataFrame
    '''

    print('Check for null values:')
    df.isnull().sum()
    df.describe()
    print('Clients histograms')
    plt.figure(figsize=(5,5))
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_hist.png',bbox_inches='tight')
    plt.figure(figsize=(5,5))
    df['Gender'].hist()
    plt.savefig('./images/eda/customer_gender_hist.png',bbox_inches='tight')

    # Copy DataFrame
    df_eda = df.copy(deep=True)

    # Churn
    df_eda['Churn'] = df_eda['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

     # Churn Distribution
    plt.figure(figsize=(20, 10))
    df_eda['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    df_eda['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df_eda['Total_Trans_Ct'],kde=True);
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')

    return df_eda

def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from
    the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                                               be used for naming variables
                                               or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Copy DataFrmae
    encoder_df = df.copy(deep=True)

    for category in category_lst:
        # Calculate the proportion of the response for each category
        category_proportions = encoder_df.groupby(category)[response].mean()

        # Map the proportions back onto the original DataFrame
        encoder_df[category + '_' +
                   response] = df[category].map(category_proportions)

    return encoder_df

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
                                                 could be used for naming
                                                 variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # get categorical df
    category_lst = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = encoder_helper(df, category_lst, response)

    # X and y values (features and target features)
    X = pd.DataFrame()
    y = df_encoded[response]

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit',
                 'Total_Revolving_Bal','Avg_Open_To_Buy',
                 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                 'Avg_Utilization_Ratio', 'Gender_Churn',
                 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df_encoded[keep_cols]

    # Train and Test split
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)

    return (x_train, x_test, y_train, y_test)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder
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
    # Classification report for random forest
    plt.figure(figsize=(15, 10))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')

    # LogisticRegression
    plt.figure(figsize=(15, 10))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Sort feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(15, 10))
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.ylabel('Importance')
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.title("Feature Importance")
    plt.savefig(output_pth + 'feature_importances.png')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Compute ROC curve for logistic regresion
    plt.figure(figsize=(15, 10))
    plot_roc_curve(lrc, x_test, y_test, alpha=0.8)
    plt.savefig('./images/results/lrc_roc_curve_result.png')

    # Compute ROC curve for random forest
    plt.figure(figsize=(15, 10))
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, alpha=0.8)
    plt.savefig('./images/results/rfc_roc_curve_result.png')

    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute and feature importance for random forest
    feature_importance_plot(model=cv_rfc,
                            x_data=x_test,
                            output_pth='./images/results/')

if __name__ == "__main__":

    df = import_data(r'./data/bank_data.csv')
    df_eda = perform_eda(df)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(df=df_eda,
                                                                   response='Churn')
    # Model training,prediction and evaluation
    train_models(x_train=X_TRAIN,
                 x_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST)
