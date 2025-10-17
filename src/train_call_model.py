import pandas as pd
import numpy as np

# Step 1: Generate Synthetic Call Data
np.random.seed(42)
n = 1000  # number of records

data = {
    "calls_last_24h": np.random.randint(0, 20, n),
    "calls_last_7d": np.random.randint(0, 100, n),
    "avg_duration_seconds": np.random.randint(10, 600, n),
    "ratio_missed_calls": np.random.rand(n),
    "is_in_contacts": np.random.randint(0, 2, n),
    "label": np.random.randint(0, 2, n)
}

df = pd.DataFrame(data)
df.to_csv("data/synthetic_call_data.csv", index=False)
print("✅ Synthetic call dataset saved to data/synthetic_call_data.csv")

# Step 2: Train the Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

joblib.dump(model, "models/call_spam_model.pkl")
print("\n💾 Model saved as models/call_spam_model.pkl")
