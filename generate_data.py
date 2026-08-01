"""
Generates a synthetic telecom/subscription customer dataset and stores it
in a SQLite database (simulating pulling data from a real company database).
"""

import numpy as np
import pandas as pd
import sqlite3

np.random.seed(7)
n = 800

tenure_months = np.random.randint(1, 72, n)
monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.5, 0.3, 0.2])
support_calls = np.random.poisson(2, n)

# Churn probability logic: short tenure + month-to-month + many support calls => higher churn
churn_prob = (
    0.5 * (tenure_months < 12).astype(int)
    + 0.3 * (contract_type == "Month-to-Month").astype(int)
    + 0.1 * (support_calls > 3).astype(int)
    + 0.1 * (monthly_charges > 90).astype(int)
)
churn_prob = np.clip(churn_prob, 0, 0.9)
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    "customer_id": range(1, n + 1),
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "contract_type": contract_type,
    "support_calls": support_calls,
    "churn": churn
})

conn = sqlite3.connect("customers.db")
df.to_sql("customers", conn, if_exists="replace", index=False)
conn.close()

print(f"Database created: customers.db ({n} customer records)")
print(df.head())
print(f"\nChurn rate: {df['churn'].mean()*100:.1f}%")
