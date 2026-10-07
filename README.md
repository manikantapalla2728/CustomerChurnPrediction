# Customer Churn Prediction using Machine Learning

## Project Overview

Customer Churn is one of the most critical challenges that a subscription-based organization faces; some examples include telecom operators, Internet Service Providers (ISPs), and Software as a Service (SaaS). In other words, customer churn refers to customers terminating their use of the services provided by an organization. The early prediction of customer churn helps the organization act proactively to minimize losses.

The purpose of this project is to develop a machine learning model for customer churn prediction based on the Telco Customer Churn dataset. Various machine learning models have been developed, trained, tested, and compared in order to find the best performing one.

**Project maintainer:** Manikanta Palla

---

## Project Structure

```text
.
├── data/
│   └── Churn-Prediction-dataset.csv
├── models/
│   ├── DecisionTreeModel.pkl
│   ├── KNN.pkl
│   ├── Logistic_Regression.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
├── notebooks/
│   └── Churn_predictions.ipynb
├── streamlit-app/
│   ├── app.py
│   └── pages/
│       ├── datasetpage.py
│       └── trynow.py
├── requirements.txt
└── README.md
```

## Setup and Run

Run these commands from the project root with Python and pip installed:

```bash
python -m venv .venv
```

Activate the environment, then install the project dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Start the Streamlit application from the project root:

```bash
streamlit run streamlit-app/app.py
```

The app dashboard displays dataset summaries and model evaluation results. Use **Try Now** to submit customer details for a prediction with one of the saved models, or **View Dataset** to inspect the CSV data. To rerun the model experiments, open `notebooks/Churn_predictions.ipynb` in Jupyter.

The saved model files are loaded by the prediction page and should remain in the `models/` directory. The comparison below uses the model configurations described in this README, retrained and evaluated on the same stratified 80/20 split (random state 42). `TotalCharges` missing values are filled using the training-set median. ROC-AUC is calculated from predicted probabilities. Retraining with different data or settings can change the results.

---

## Objectives

* Perform data preprocessing and feature engineering.
* Handle categorical and numerical features using machine learning pipelines.
* Train multiple machine learning classification models.
* Compare model performance using multiple evaluation metrics.
* Perform hyperparameter tuning to improve model performance.
* Select the best-performing model based on business-relevant metrics.

---

## Dataset Information

The dataset contains customer information including:

* Customer demographics
* Account information
* Subscription details
* Internet services
* Contract types
* Payment methods
* Monthly charges
* Total charges
* Customer churn status

Target Variable:

* Churn

  * Yes = Customer left the service
  * No = Customer stayed with the service

---

## Data Preprocessing

Several preprocessing steps were performed before training the models:

### Handling Missing Values

Missing values were identified and handled appropriately to ensure data quality.

### Categorical Feature Encoding

Different encoding techniques were used depending on the feature type:

* Ordinal Encoding for binary categorical features.
* One-Hot Encoding for multi-category features.

### Feature Scaling

Numerical features were standardized using StandardScaler to ensure consistent feature ranges and improve model performance.

### Pipeline Implementation

Scikit-Learn Pipelines were used to automate preprocessing and model training. This approach prevents data leakage and ensures reproducibility.

---

## Machine Learning Models Used

The following classification algorithms were implemented and evaluated:

### 1. Logistic Regression

Linear classifier chosen for benchmarking was Logistic Regression. Though Logistic Regression is very simple, it performed very well in terms of prediction accuracy and generalization capability.


### 2. K-Nearest Neighbors (KNN)

For KNN, manual parameter tuning was performed using various values of:

* n_neighbors
* distance metric
* leaf size

Optimal results were obtained using:

* n_neighbors = 45
* metric = euclidean

### 3. Decision Tree

Decision Tree hyperparameters were manually tuned to improve performance and reduce overfitting.

Best configuration:

* criterion = gini
* max_depth = 7
* min_samples_split = 20
* min_samples_leaf = 20

### 4. Random Forest

Random Forest was implemented as an ensemble learning technique and tuned using:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

Best configuration:

* n_estimators = 300
* max_depth = 10
* min_samples_split = 10
* min_samples_leaf = 2

### 5. XGBoost

An XGBoost classifier is trained at the end of `notebooks/Churn_predictions.ipynb` using the same preprocessing pipeline and train/test split as the other models. The notebook reports accuracy, precision, recall, F1, and ROC-AUC, then saves the complete pipeline to `models/XGBoost.pkl` for use in the Streamlit prediction form. Its configuration uses 300 estimators, a maximum tree depth of 4, and a learning rate of 0.05.

---

## Model Evaluation Metrics

The following evaluation metrics were used:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* ROC-AUC Score

Since customer churn prediction is an imbalanced classification problem, special attention was given to:

* Recall
* F1 Score
* ROC-AUC Score

rather than relying solely on accuracy.

---

## Model Comparison

| Model                    | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------------ | -------- | --------- | ------ | -------- | ------- |
| Random Forest (Tuned)    | 0.8070   | 0.6759    | 0.5241 | 0.5904   | 0.8433  |
| Logistic Regression      | 0.8034   | 0.6520    | 0.5561 | 0.6003   | 0.8428  |
| XGBoost                  | 0.8020   | 0.6547    | 0.5374 | 0.5903   | 0.8416  |
| Decision Tree (Tuned)    | 0.7956   | 0.6243    | 0.5775 | 0.6000   | 0.8215  |
| KNN (45, Euclidean)      | 0.7842   | 0.6882    | 0.3422 | 0.4571   | 0.7899  |

---

## Results and Findings

On this fixed test split:

* Random Forest had the highest accuracy (80.70%) and ROC-AUC (0.8433).
* Logistic Regression had the highest F1 score (0.6003), narrowly ahead of the Decision Tree (0.6000).
* The Decision Tree had the highest recall (0.5775), which may be useful when finding as many likely-to-churn customers as possible is the priority.
* KNN had the highest precision (0.6882) but the lowest recall (0.3422) among these models.
* XGBoost scored 80.20% accuracy and 0.8416 ROC-AUC, close to the Logistic Regression and Random Forest results.

---

## Final Model Selection

Logistic Regression is retained as the balanced-F1 choice in this comparison. There is no single winner on every metric: use Random Forest when accuracy or ROC-AUC is the priority, or Decision Tree when recall is the priority. Choose the operating model based on the cost of missed churners versus unnecessary retention outreach.

Logistic Regression Performance:

* Accuracy: 80.34%
* Precision: 65.20%
* Recall: 55.61%
* F1 Score: 60.03%
* ROC-AUC Score: 84.28%

These figures are from the same stratified test split and use predicted probabilities for ROC-AUC, so they can be compared consistently across the models in the table.

This is a single holdout evaluation, not cross-validation. Small score differences should not be treated as proof that one model will perform better on future customer data; validate the selected model on new data before using it for business decisions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Streamlit
* XGBoost

---

## Future Improvements

Potential future enhancements include:

* LightGBM implementation
* CatBoost implementation
* Feature selection techniques


---

## Conclusion

This project demonstrates an end-to-end workflow for customer churn prediction, including data preprocessing, model training, evaluation, and a Streamlit interface. The evaluated models trade off accuracy, recall, precision, and ROC-AUC differently, so model selection should reflect the business cost of missed churners and unnecessary retention outreach. The reported metrics provide a reproducible baseline on one held-out split; further validation is needed before deployment.