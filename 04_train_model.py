"""Train a scam/spam text message detector and save it to disk."""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load data -----------------------------------------------------------
if os.path.exists("spam.csv"):
    # The public "SMS Spam Collection" format: columns v1 (label), v2 (text)
    print("Using spam.csv (bigger dataset)")
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
else:
    print("Using 03_sample_data.csv (small demo dataset)")
    df = pd.read_csv("03_sample_data.csv")

df = df.dropna()
df["is_scam"] = (df["label"] == "spam").astype(int)
print(f"Loaded {len(df)} messages, {df['is_scam'].sum()} scam/spam")

# 2. Split into training and testing sets ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["is_scam"],
    test_size=0.25, random_state=42, stratify=df["is_scam"],
)

# 3. Build the model: text -> numbers -> prediction --------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# 4. Evaluate ----------------------------------------------------------
predictions = model.predict(X_test)
print("\nResults on messages the model has never seen:\n")
print(classification_report(y_test, predictions, target_names=["legitimate", "scam"]))

# 5. Show the words the model finds most suspicious --------------------
vec = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
words = vec.get_feature_names_out()
top = clf.coef_[0].argsort()[-10:][::-1]
print("Top scam-signal words:", [words[i] for i in top])

# 6. Save the trained model -------------------------------------------
joblib.dump(model, "scam_model.joblib")
print("\nModel saved as scam_model.joblib")
