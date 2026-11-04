# Scam Text Message Detector

A small predictive AI project that classifies text messages as **scam/spam** or **legitimate (ham)**.

## How it works
1. `03_sample_data.csv` holds labeled example messages.
2. `04_train_model.py` turns the text into numbers (TF-IDF), trains a Logistic Regression model, prints how well it performs, and saves the model.
3. `05_predict.py` loads the saved model so you can paste in any text and get a verdict with a scam probability.

## How to run (GitHub Codespaces)
```
pip install -r 02_requirements.txt
python 04_train_model.py
python 05_predict.py
```

## Using a bigger dataset (optional)
Download the "SMS Spam Collection" dataset (spam.csv) from Kaggle or the UCI Machine Learning Repository, upload it to this repo as `spam.csv`, and run `04_train_model.py` again. It will detect the file and use it automatically.

## Notes
- The included sample data is tiny and for learning only. Real accuracy needs more data.
- Scams are rarer than normal texts in real life, so check precision and recall, not just accuracy.
