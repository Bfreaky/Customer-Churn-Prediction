# Customer Churn Prediction (SQL + Machine Learning)

An end-to-end project that pulls customer data from a SQL database and
predicts which customers are likely to churn (leave) using a machine
learning classification model. This combines two core data science skills
— SQL and ML — into one complete pipeline.

## What This Project Shows
- Storing and querying data with SQL (SQLite)
- Loading SQL query results into Pandas
- Feature engineering (encoding categorical variables)
- Train/test split with stratification
- Training a classification model (Random Forest)
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrix
- Feature importance analysis (explaining *why* the model predicts churn)

## Tech Used
- Python 3, Pandas, SQLite, Scikit-learn

## Files
- `generate_data.py` — creates synthetic customer data and stores it in `customers.db`
- `churn_model.py` — pulls data via SQL, trains model, evaluates results

## How to Run
1. Install requirements:
   ```
   pip install pandas scikit-learn
   ```
2. Generate the database:
   ```
   python generate_data.py
   ```
3. Train and evaluate the model:
   ```
   python churn_model.py
   ```

## Pipeline Overview
```
SQLite Database → SQL Query → Pandas DataFrame → Feature Engineering
→ Train/Test Split → Random Forest Model → Evaluation → Feature Importance
```

## Sample Results
- Model Accuracy: ~78%
- Top churn driver identified: customer tenure (newer customers churn more)
- Second driver: monthly charges (higher charges → higher churn risk)



## Author
[Ayush Raj] — B.Tech CSE, 3rd Year (Data Analytics Specialization)
