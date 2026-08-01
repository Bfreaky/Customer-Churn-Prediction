"""
Customer Churn Prediction
Pulls customer data from a SQL database, trains a classification model,
and evaluates it. Demonstrates the full analyst-to-data-scientist pipeline:
SQL -> Pandas -> Scikit-learn -> Evaluation.
"""

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---- Step 1: Pull data using SQL ----
conn = sqlite3.connect("customers.db")
query = """
SELECT customer_id, tenure_months, monthly_charges, contract_type,
       support_calls, churn
FROM customers
"""
df = pd.read_sql_query(query, conn)
conn.close()

print("Data pulled from SQL database.")
print(df.head(), "\n")

# ---- Step 2: Prepare features ----
le = LabelEncoder()
df["contract_type_encoded"] = le.fit_transform(df["contract_type"])

features = ["tenure_months", "monthly_charges", "contract_type_encoded", "support_calls"]
X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Step 3: Train model ----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Step 4: Evaluate ----
y_pred = model.predict(X_test)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- Step 5: Feature importance ----
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance (what drives churn most):")
for feat, score in importance.items():
    print(f"  {feat}: {score:.3f}")
