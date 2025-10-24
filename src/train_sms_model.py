import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ✅ Load CSV (with encoding fix for Windows/Excel CSV)
df = pd.read_csv("../data/spam.csv", encoding="latin1")

# ✅ Convert correct columns
# v1 = ham/spam → label
# v2 = actual message text
df["label"] = df["v1"].map({"ham": 0, "spam": 1})
df["text"] = df["v2"]

print(df.head())  # Just to confirm

# ✅ Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ Predictions
y_pred = model.predict(X_test)

# ✅ Performance
print("\n📊 SMS Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

# ✅ Save model & vectorizer
joblib.dump(model, "../models/sms_spam_model.pkl")
joblib.dump(vectorizer, "../models/sms_vectorizer.pkl")
print("\n💾 Model saved in /models folder ✅")
