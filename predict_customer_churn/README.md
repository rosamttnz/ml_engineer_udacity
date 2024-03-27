# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the 1st project of the ML DevOps nanodegree. In this project, I will implement my learnings on Clean Code Princples to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Project Structure
predict_customer_churn/
├── data/
│ └── bank_data.csv
├── images/
│ ├── eda/
│ │ ├── churn_distribution.png
│ │ ├── customer_age_distribution.png
│ │ ├── customer_age_hist.png
│ │ ├── customer_gender_hist.png
│ │ ├── heatmap.png
│ │ ├── marital_status_distribution.png
│ │ └── total_transaction_distribution.png
│ └── results/
│ ├── feature_importances.png
│ ├── logistic_results.png
│ ├── lrc_roc_curve_result.png
│ ├── rf_results.png
│ └── rfc_roc_curve_result.png
├── logs/
│ └── churn_library.log
├── models/
│ ├── logistic_model.pkl
│ └── rfc_model.pkl
├── Guide.ipynb
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
└── requirements_py3.9.txt

## Running Files
To run this project, ensure you have Python 3.8+ installed on your system. Follow these steps:

1. Clone the repository to your local machine:
    git clone https://github.com/rosamttnz/Predict-Customer-Churn-with-Clean-Code.git
2. Navigate into the project directory:
    cd Predict-Customer-Churn-with-Clean-Code
3. Install the required dependencies:
    pip install -r requirements.txt

## Usage
To start analyzing the customer churn data and running the predictive models, follow these steps:

- **Data Analysis:** Open `churn_notebook.ipynb` with Jupyter Notebook for exploratory data analysis visuals.
- **Running Models:** Execute `churn_library.py` to train models and evaluate their performance.
python churn_library.py
- **Testing:** Run unit tests to ensure the reliability of the codebase with `churn_script_logging_and_tests.py`
python test_churn_library.py


