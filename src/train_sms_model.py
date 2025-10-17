import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Sample SMS dataset
data = {
    "text": [
        "Win money now!", "Call me later", "You won a prize", "Hello, how are you?",
        "Free entry to lottery", "Meeting at 5pm", "Congrats! Claim your reward", "Are you coming?"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 SMS Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))

# Save model and vectorizer
joblib.dump(model, "models/sms_spam_model.pkl")
joblib.dump(vectorizer, "models/sms_vectorizer.pkl")
print("\n💾 SMS model and vectorizer saved in models/")